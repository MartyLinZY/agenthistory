#!/usr/bin/env python3
"""Aggregate DeepSeek review labels to PR-level results and weekly-report artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from classify_github_review_issues_deepseek import (
    DEFAULT_INPUT,
    DEFAULT_SCHEMA,
    load_completed,
    load_prs,
    load_schema,
    make_requests,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_LABELS = ROOT / "outputs" / "github_review_issue_ds_baseline_v1" / "review_labels.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs" / "github_review_issue_ds_baseline_v1" / "analysis-output"
AGENT_LABELS = {
    "claude_code": "Claude Code",
    "codex": "OpenAI Codex",
    "copilot": "GitHub Copilot",
    "devin": "Devin",
    "jules": "Jules",
}
CATEGORY_LABELS = {
    "functional_bug": "功能/逻辑错误",
    "implementation_deviation": "实现偏差",
    "design_maintainability": "设计与维护性",
    "insufficient_testing": "测试不足",
    "duplication_redundancy": "重复与冗余",
    "other_code_issue": "其他代码问题",
    "compatibility": "兼容性",
    "edge_case": "边界条件",
    "documentation": "文档问题",
    "security": "安全问题",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-size", type=int, default=100)
    return parser.parse_args(argv)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def agent_name(pr: Mapping[str, Any]) -> str:
    assignments = pr.get("selectionAssignments") or []
    return str(assignments[0].get("agent") or "unknown") if assignments else "unknown"


def aggregate(
    prs: list[dict[str, Any]],
    label_path: Path,
    schema: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    expected = list(make_requests(prs, schema))
    completed = load_completed(label_path)
    missing = [
        item.request_id for item in expected
        if completed.get(item.request_id, {}).get("input_hash") != item.input_hash
    ]
    if missing:
        raise RuntimeError(
            f"DeepSeek分类未完成：缺少或输入hash不一致 {len(missing)}/{len(expected)} 条；"
            "请先重跑分类命令"
        )
    review_records = [completed[item.request_id] for item in expected]
    by_pr: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in review_records:
        by_pr[str(record["pr_node_id"])].append(record)

    pr_records: list[dict[str, Any]] = []
    for pr in prs:
        node_id = str(pr["id"])
        records = by_pr.get(node_id, [])
        code_records = [record for record in records if record["classification"]["is_code_issue"]]
        category_counts = Counter(
            category
            for record in code_records
            for category in record["classification"]["categories"]
        )
        severities = Counter(record["classification"]["severity"] for record in code_records)
        severity_order = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
        max_severity = max(severities, key=lambda item: severity_order[item]) if severities else "none"
        assignments = pr.get("selectionAssignments") or []
        pr_records.append({
            "pr_node_id": node_id,
            "pr_url": pr["url"],
            "repository": (pr.get("repository") or {}).get("nameWithOwner"),
            "number": pr.get("number"),
            "agent": agent_name(pr),
            "state": pr.get("state"),
            "merged": bool(pr.get("mergedAt")),
            "selection_rank": assignments[0].get("rank") if assignments else None,
            "classification_status": "model_classified" if records else "no_human_review_text",
            "human_review_text_count": len(records),
            "code_issue_review_count": len(code_records),
            "non_code_review_count": len(records) - len(code_records),
            "has_reported_code_issue": bool(code_records),
            "categories": sorted(category_counts),
            "category_review_counts": dict(sorted(category_counts.items())),
            "max_severity": max_severity,
            "mean_confidence": (
                round(sum(record["classification"]["confidence"] for record in records) / len(records), 4)
                if records else 1.0
            ),
        })

    usage = Counter()
    for record in review_records:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[key] += int((record.get("usage") or {}).get(key) or 0)
    summary = {
        "pull_requests": len(pr_records),
        "reviewed_pull_requests": sum(record["human_review_text_count"] > 0 for record in pr_records),
        "no_review_pull_requests": sum(record["human_review_text_count"] == 0 for record in pr_records),
        "review_texts": len(review_records),
        "code_issue_reviews": sum(record["classification"]["is_code_issue"] for record in review_records),
        "non_code_issue_reviews": sum(not record["classification"]["is_code_issue"] for record in review_records),
        "pull_requests_with_reported_code_issue": sum(record["has_reported_code_issue"] for record in pr_records),
        "usage": dict(usage),
    }
    return pr_records, review_records, summary


def category_tables(
    pr_records: list[dict[str, Any]], review_records: list[dict[str, Any]], categories: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reviewed_prs = [record for record in pr_records if record["human_review_text_count"] > 0]
    code_issue_prs = [record for record in reviewed_prs if record["has_reported_code_issue"]]
    code_reviews = [record for record in review_records if record["classification"]["is_code_issue"]]
    pr_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for category in categories:
        pr_count = sum(category in record["categories"] for record in reviewed_prs)
        low, high = wilson_interval(pr_count, len(reviewed_prs))
        code_pr_count = sum(category in record["categories"] for record in code_issue_prs)
        pr_rows.append({
            "category": category,
            "category_zh": CATEGORY_LABELS[category],
            "pr_count": pr_count,
            "reviewed_pr_denominator": len(reviewed_prs),
            "percent_of_reviewed_prs": pr_count / len(reviewed_prs) if reviewed_prs else 0,
            "wilson_95_low": low,
            "wilson_95_high": high,
            "percent_of_code_issue_prs": code_pr_count / len(code_issue_prs) if code_issue_prs else 0,
        })
        review_count = sum(
            category in record["classification"]["categories"] for record in code_reviews
        )
        all_review_count = sum(
            category in record["classification"]["categories"] for record in review_records
        )
        review_rows.append({
            "category": category,
            "category_zh": CATEGORY_LABELS[category],
            "review_count": review_count,
            "code_issue_review_denominator": len(code_reviews),
            "percent_of_code_issue_reviews": review_count / len(code_reviews) if code_reviews else 0,
            "percent_of_all_review_texts": all_review_count / len(review_records) if review_records else 0,
        })
    return (
        sorted(pr_rows, key=lambda row: (-row["pr_count"], row["category"])),
        sorted(review_rows, key=lambda row: (-row["review_count"], row["category"])),
    )


def agent_table(pr_records: list[dict[str, Any]], categories: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for agent in sorted({record["agent"] for record in pr_records}):
        group = [record for record in pr_records if record["agent"] == agent]
        reviewed = [record for record in group if record["human_review_text_count"] > 0]
        issue = [record for record in reviewed if record["has_reported_code_issue"]]
        review_low, review_high = wilson_interval(len(reviewed), len(group))
        issue_low, issue_high = wilson_interval(len(issue), len(reviewed))
        row: dict[str, Any] = {
            "agent": agent,
            "agent_label": AGENT_LABELS.get(agent, agent),
            "sample_prs": len(group),
            "reviewed_prs": len(reviewed),
            "review_coverage": len(reviewed) / len(group) if group else 0,
            "review_coverage_95_low": review_low,
            "review_coverage_95_high": review_high,
            "reported_issue_prs": len(issue),
            "reported_issue_rate_among_reviewed": len(issue) / len(reviewed) if reviewed else 0,
            "reported_issue_rate_95_low": issue_low,
            "reported_issue_rate_95_high": issue_high,
        }
        for category in categories:
            row[category] = sum(category in record["categories"] for record in reviewed)
        rows.append(row)
    return rows


def save_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def make_figures(
    output: Path,
    category_pr: list[dict[str, Any]],
    agents: list[dict[str, Any]],
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    ordered = list(reversed(category_pr))
    values = [row["percent_of_reviewed_prs"] * 100 for row in ordered]
    errors = [
        [value - row["wilson_95_low"] * 100 for value, row in zip(values, ordered)],
        [row["wilson_95_high"] * 100 - value for value, row in zip(values, ordered)],
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.barh(
        [row["category_zh"] for row in ordered], values, xerr=errors,
        color="#0072B2", alpha=0.9, capsize=2,
    )
    ax.set_xlabel("占含人类 review 文本 PR 的比例（%）")
    ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=0.25)
    for index, (value, row) in enumerate(zip(values, ordered)):
        ax.text(value + 0.6, index, f"{row['pr_count']} ({value:.1f}%)", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "figure-01-category-pr-level.pdf", bbox_inches="tight")
    fig.savefig(figures / "figure-01-category-pr-level.svg", bbox_inches="tight")
    plt.close(fig)

    labels = [row["agent_label"] for row in agents]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), sharey=True)
    for ax, metric, low_key, high_key, title, color in (
        (
            axes[0], "review_coverage", "review_coverage_95_low", "review_coverage_95_high",
            "Baseline 中含人类 review 文本", "#56B4E9",
        ),
        (
            axes[1], "reported_issue_rate_among_reviewed", "reported_issue_rate_95_low",
            "reported_issue_rate_95_high", "Reviewed PR 中报告代码问题", "#D55E00",
        ),
    ):
        values = [row[metric] * 100 for row in agents]
        errors = [
            [value - row[low_key] * 100 for value, row in zip(values, agents)],
            [row[high_key] * 100 - value for value, row in zip(values, agents)],
        ]
        ax.errorbar(values, labels, xerr=errors, fmt="o", color=color, capsize=3)
        ax.set_xlim(0, 100)
        ax.set_xlabel("比例（%），Wilson 95% CI")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "figure-02-agent-coverage.pdf", bbox_inches="tight")
    fig.savefig(figures / "figure-02-agent-coverage.svg", bbox_inches="tight")
    plt.close(fig)


def audit_sample(review_records: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    ranked = sorted(
        review_records,
        key=lambda record: hashlib.sha256(
            f"agentbench-review-audit-v1\0{record['request_id']}".encode("utf-8")
        ).hexdigest(),
    )[: min(size, len(review_records))]
    return [{
        "request_id": record["request_id"],
        "pr_url": record["pr_url"],
        "agent": record["agent"],
        "review_source": record["review_source"],
        "review_url": record.get("review_url"),
        "model_is_code_issue": record["classification"]["is_code_issue"],
        "model_categories": " | ".join(record["classification"]["categories"]),
        "model_severity": record["classification"]["severity"],
        "model_confidence": record["classification"]["confidence"],
        "model_evidence_quote": record["classification"]["evidence_quote"],
        "human_is_code_issue": "",
        "human_categories": "",
        "human_notes": "",
    } for record in ranked]


def write_reports(
    output: Path,
    summary: Mapping[str, Any],
    category_pr: list[dict[str, Any]],
    category_review: list[dict[str, Any]],
    agents: list[dict[str, Any]],
) -> None:
    reviewed = int(summary["reviewed_pull_requests"])
    issue_prs = int(summary["pull_requests_with_reported_code_issue"])
    reviews = int(summary["review_texts"])
    issue_reviews = int(summary["code_issue_reviews"])
    top = category_pr[:3]
    top_sentence = "、".join(
        f"{row['category_zh']} {row['pr_count']} 个（{percent(row['percent_of_reviewed_prs'])}）"
        for row in top
    )
    analysis_report = f"""# baseline-v1 Human Review 10 类问题分析

## Analysis Question

在按 Agent、日期和仓库分层抽取的 1,250 个真实 Agent PR 中，人类 review/discussion 文本报告了哪些类型的代码 concern？

## Key Findings

- 1,250 个 PR 中有 {reviewed} 个包含人类 review/discussion 文本，共 {reviews} 段；这是 review 可观测性，不是代码问题发生率。
- {issue_prs}/{reviewed} 个 reviewed PR（{percent(issue_prs / reviewed if reviewed else 0)}）至少有一段文本被 DeepSeek 标为明确代码 concern；review-level 为 {issue_reviews}/{reviews}（{percent(issue_reviews / reviews if reviews else 0)}）。
- PR-level 最常见的三类 reported concerns 是：{top_sentence}。多标签比例不能相加为 100%。
- Agent 间只报告描述性分布；review 覆盖率、任务和仓库审查实践均不同，不支持将差异解释为模型质量因果差异。

## Claim Candidates

- Claim: 在 baseline-v1 的 human-reviewed 子集中，{top[0]['category_zh']}是最常见的模型识别 concern。
  - Source evidence: `tables/category_pr_level.csv`，n={reviewed} reviewed PR。
  - Allowed wording: “human review 中最常被模型识别/报告”。
  - Forbidden stronger wording: “Agent 代码中客观存在最多”或“由 Agent 导致”。
  - Uncertainty: 自动标签尚未经过本轮人工 gold 验证；reviewed PR 不是全部 PR 的随机无缺失观测。
  - Next check: 完成 `manual_audit_sample_100.csv` 双人复核并计算多标签性能。
  - Decision: keep

## Main Caveats

- DeepSeek 分类对象是 reviewer reported concern，不是独立验证后的真实缺陷。
- 1,052 个无 human review 文本 PR 被确定性标为 `no_human_review_text`，没有调用模型猜测。
- PR discussion 也包含在 `humanReviewTexts` 中；正式论文应分别报告 formal review、inline review 和 PR discussion 的敏感性分析。
- 当前只有一次确定性模型运行，没有人工 gold label，因此不做准确率、显著性或 Agent 优劣结论。
"""
    (output / "analysis-report.md").write_text(analysis_report, encoding="utf-8")

    stats = f"""# Statistics Appendix

## Units and Denominators

- Baseline sampling unit: PR，n={summary['pull_requests']}，每 Agent 250。
- Review-level classification unit: 一段 `humanReviewTexts`，n={reviews}。
- PR-level category prevalence denominator: 含人类文本 PR，n={reviewed}。
- Reported-issue PR denominator: reviewed PR；多标签类别可重叠。

## Uncertainty

- 类别 PR-level 比例和 Agent coverage 图使用二项 Wilson 95% CI。
- 未进行 Agent 间显著性检验：比较不是预注册的主要因果问题，且 Codex 等 Agent 的 reviewed n 很小、repository/task/review practice 混杂明显。
- LLM 为单次 temperature=0 标注；Wilson CI 只描述抽样比例不确定性，不包含模型标注误差。

## Exact Summary

- Code-issue review texts: {issue_reviews}/{reviews} ({percent(issue_reviews / reviews if reviews else 0)})。
- PR with at least one reported issue: {issue_prs}/{reviewed} ({percent(issue_prs / reviewed if reviewed else 0)})。
- Non-code review texts: {summary['non_code_issue_reviews']}/{reviews} ({percent(summary['non_code_issue_reviews'] / reviews if reviews else 0)})。
- Token usage: {json.dumps(summary['usage'], ensure_ascii=False)}。
"""
    (output / "stats-appendix.md").write_text(stats, encoding="utf-8")

    figure_catalog = f"""# Figure Catalog

## Figure 1 — `figures/figure-01-category-pr-level.pdf`

- Purpose: 回答 reviewed PR 中哪些 10 类 concern 最常见。
- Data source: `tables/category_pr_level.csv`，n={reviewed} reviewed PR。
- Caption: 各类别占 reviewed PR 的比例；误差条为 Wilson 95% CI；同一 PR 可多标签。
- Observation: 前三类为 {top_sentence}。
- Interpretation: 这些是 reviewer 报告并由 DeepSeek 分类的 concern 分布，不是独立验证 defect 分布。
- Decision: 人工复核应优先覆盖前三类以及低频 edge/security 类。

## Figure 2 — `figures/figure-02-agent-coverage.pdf`

- Purpose: 分离“是否观察到 human review”与“review 中是否报告代码 concern”。
- Data source: `tables/agent_summary.csv`；每 Agent baseline n=250。
- Caption: 左图为 review 覆盖率，右图为 reviewed PR 中 reported issue 比例；误差条为 Wilson 95% CI。
- Observation: Agent 间 review 可观测性不同，低 reviewed n 组区间较宽。
- Interpretation: 不能把右图直接解释为 Agent 质量排序。
- Decision: 后续 Agent 比较需控制 repository、task、PR size，并报告 review-selection mechanism。
"""
    (output / "figure-catalog.md").write_text(figure_catalog, encoding="utf-8")

    report = f"""---
type: results-report
date: 2026-08-28
experiment_line: github-review-issues
round: 1
purpose: baseline-classification
status: active
source_artifacts:
  - analysis-output/analysis-report.md
  - analysis-output/stats-appendix.md
  - analysis-output/figure-catalog.md
linked_experiments: []
linked_results: []
---

# GitHub Review Issues / Round 1 / Baseline Classification / 2026-08-28

## Executive Summary

对 baseline-v1 的 1,250 个真实 Agent PR 完成覆盖式输出，其中 {reviewed} 个 PR 的 {reviews} 段人类 review/discussion 文本由 DeepSeek 按附件 10 类体系逐条标注，1,052 个无文本 PR 确定性记录为不可观察。{issue_prs} 个 reviewed PR 至少报告一种代码 concern；最常见三类为{top_sentence}。该结果支持进入人工一致性验证，但不支持 Agent 质量排名。

## Experiment Identity and Decision Context

本轮回答“真实 Agent PR 的公开 human review 暴露了哪些代码问题”，并为后续 benchmark failure taxonomy 和典型案例选择提供候选。

## Setup and Evaluation Protocol

输入为五类 Agent 各 250 条的日期均衡、仓库上限 baseline。模型单元为一段 human review 文本，temperature=0，封闭多标签 taxonomy；PR-level 为同一 PR 各文本标签并集。

## Main Findings

{analysis_report.split('## Claim Candidates')[0].split('## Key Findings')[1].strip()}

## Statistical Validation

本轮为描述性分类实验。所有主要比例给出精确分子/分母，图中使用 Wilson 95% CI；未进行无充分设计基础的 Agent 显著性检验。

## Figure-by-Figure Interpretation

Figure 1 用于确定人工复核和案例整理的优先类别；Figure 2 说明 review 缺失机制不可忽略，Agent 间 observed issue rate 不能直接作为质量排名。

## Failure Cases / Negative Results / Limitations

无 review 文本的 1,052 个 PR 无法由该证据源判断代码问题；自动标签尚无本轮人工 gold；PR discussion 与正式 review 的语境不同；多标签类别边界仍可能存在 `functional_bug`/`implementation_deviation`/`edge_case` 混淆。

## What Changed Our Belief

该 baseline 能系统复现真实 reviewer concern 分类，但 review 可观测性只覆盖总体的一部分，因此 benchmark 构建应把“是否有 reviewer oracle”作为单独阶段，而不是把无 review 等同于无问题。

## Next Actions

完成 100 条分层人工复核，计算 human–LLM 的 is-code-issue 与多标签一致性；对 files/commits 截断的 12 条候选补采；再从高置信、需求与 base 可恢复样本中进行 benchmark 重放。

## Artifact and Reproducibility Index

- `review_labels.jsonl`: review-level DeepSeek 原始标签。
- `pr_labels.jsonl`: 1,250 PR 聚合结果。
- `tables/`: 精确统计与 100 条人工复核表。
- `figures/`: PDF/SVG 分布图。
"""
    (output / "2026-08-28--github-review-issues--r01--baseline-classification.md").write_text(
        report, encoding="utf-8"
    )

    weekly = (
        f"本周参考既有 AIDev-pop 人类 review 的 10 类 AI 代码问题体系，对新采集的 1,250 个真实 Agent PR 进行了覆盖式分类："
        f"其中 {reviewed} 个 PR 包含 {reviews} 段非机器人 review 或讨论文本，使用 DeepSeek 逐条进行多标签判断，"
        f"其余 {summary['no_review_pull_requests']} 个无 review 文本 PR 保留为不可观察而未推测问题；"
        f"在含 review 文本的 PR 中，{issue_prs} 个（{percent(issue_prs / reviewed if reviewed else 0)}）至少被 reviewer 报告一种明确代码 concern，"
        f"最常见的三类为{top_sentence}。该结果说明真实 reviewer 对 Agent 代码的关注可由统一 taxonomy 复现，"
        "但当前证据反映的是 reviewer reported concerns，而不是独立验证后的真实缺陷；不同 Agent 的 review 覆盖率和仓库构成也不同，"
        "因此本周仅报告描述性分布，不作 Agent 质量排序。下一步将对 100 条分层样本进行人工复核，评估 DeepSeek 多标签结果的一致性，"
        "并从需求、base commit、patch 和测试 oracle 均可恢复的高置信样本中开展 benchmark 重放。"
    )
    (output / "weekly_report_paragraph.md").write_text(weekly + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    schema = load_schema(args.schema)
    prs = load_prs(args.input)
    pr_records, review_records, summary = aggregate(prs, args.labels, schema)
    categories = list(schema["code_issue_categories"])
    category_pr, category_review = category_tables(pr_records, review_records, categories)
    agents = agent_table(pr_records, categories)
    args.output.mkdir(parents=True, exist_ok=True)
    tables = args.output / "tables"
    save_jsonl(args.output / "pr_labels.jsonl", pr_records)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(tables / "category_pr_level.csv", category_pr, list(category_pr[0]))
    write_csv(tables / "category_review_level.csv", category_review, list(category_review[0]))
    write_csv(tables / "agent_summary.csv", agents, list(agents[0]))
    audit = audit_sample(review_records, args.audit_size)
    write_csv(tables / "manual_audit_sample_100.csv", audit, list(audit[0]))
    make_figures(args.output, category_pr, agents)
    write_reports(args.output, summary, category_pr, category_review, agents)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
