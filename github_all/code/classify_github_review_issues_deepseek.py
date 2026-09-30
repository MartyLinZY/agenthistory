#!/usr/bin/env python3
"""Classify baseline-v1 human review texts with a closed 10-category taxonomy.

The unit sent to DeepSeek is one human review/discussion text. Results are
append-only JSONL and resume by request/input hash. PR-level aggregation,
including deterministic no-review records for all 1,250 PRs, is handled by
``analyze_github_review_issue_results.py``. The API key is read only from
``DEEPSEEK_API_KEY``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import requests


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "outputs" / "github_agent_prs_baseline_202506_202508_v1" / "prs.jsonl"
DEFAULT_SCHEMA = ROOT / "github_review_issue_schema_10cats.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "github_review_issue_ds_baseline_v1" / "review_labels.jsonl"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
LABELER_VERSION = "1.0.0"
PROMPT_VERSION = "1.0.0"
USER_AGENT = "agentbench-review-issue-labeler/1.0"
_THREAD_LOCAL = threading.local()


class ClassificationError(RuntimeError):
    """Raised when a request can be retried or its output violates the contract."""


@dataclass(frozen=True)
class ReviewRequest:
    request_id: str
    pr_node_id: str
    pr_url: str
    repository: str
    number: int
    agent: str
    review_id: str
    review_source: str
    review_url: str | None
    input_hash: str
    evidence: dict[str, Any]
    prompt: str


def clean_text(value: Any, limit: int = 6_000) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def normalize_for_quote(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--limit", type=int, default=20, help="review文本冒烟数量，默认20")
    scope.add_argument("--all-reviewed", action="store_true", help="处理全部含文本review")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须大于0")
    if args.workers < 1:
        parser.error("--workers 必须大于0")
    return args


def load_schema(path: Path) -> dict[str, Any]:
    schema = json.loads(path.read_text(encoding="utf-8"))
    categories = schema.get("code_issue_categories")
    if not schema.get("schema_version") or not isinstance(categories, dict) or len(categories) != 10:
        raise ValueError(f"无效的10类schema: {path}")
    return schema


def load_prs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not value.get("id") or not value.get("url"):
                raise ValueError(f"{path}:{line_number} 缺少PR id/url")
            rows.append(value)
    if not rows:
        raise ValueError(f"输入为空: {path}")
    return rows


def agent_name(pr: Mapping[str, Any]) -> str:
    assignments = pr.get("selectionAssignments") or []
    return str(assignments[0].get("agent") or "unknown") if assignments else "unknown"


def review_contexts(pr: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    for item in pr.get("reviews") or []:
        if item.get("id"):
            contexts[str(item["id"])] = {"review_state": item.get("state")}
    for item in pr.get("comments") or []:
        if item.get("id"):
            contexts.setdefault(str(item["id"]), {})
    for thread in pr.get("reviewThreads") or []:
        thread_context = {
            "path": thread.get("path"),
            "line": thread.get("line"),
            "original_line": thread.get("originalLine"),
            "is_outdated": thread.get("isOutdated"),
            "is_resolved": thread.get("isResolved"),
        }
        for comment in thread.get("comments") or []:
            if not comment.get("id"):
                continue
            contexts[str(comment["id"])] = {
                **thread_context,
                "diff_hunk": clean_text(comment.get("diffHunk"), 2_500),
            }
    return contexts


def build_evidence(pr: Mapping[str, Any], review: Mapping[str, Any], index: int) -> dict[str, Any]:
    repository = pr.get("repository") or {}
    body = clean_text(review.get("body"), 6_000)
    review_id = str(review.get("id") or hashlib.sha256(
        f"{review.get('source')}\0{body}\0{index}".encode("utf-8")
    ).hexdigest()[:20])
    context = review_contexts(pr).get(review_id, {})
    return {
        "pr": {
            "node_id": str(pr["id"]),
            "url": str(pr["url"]),
            "repository": str(repository.get("nameWithOwner") or ""),
            "number": int(pr.get("number") or 0),
            "title": clean_text(pr.get("title"), 500),
            "body_context": clean_text(pr.get("body"), 1_000),
            "agent_signal": agent_name(pr),
        },
        "review": {
            "id": review_id,
            "source": str(review.get("source") or "unknown"),
            "url": review.get("url"),
            "author": (review.get("author") or {}).get("login"),
            "author_association": review.get("authorAssociation"),
            "created_at": review.get("createdAt"),
            "body": body,
            "context": context,
        },
        "evidence_boundary": (
            "Only the review body is evidence that a reviewer reported a concern. "
            "PR metadata and diff context may disambiguate references but cannot create a concern."
        ),
    }


def output_contract(schema: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "is_code_issue": True,
        "categories": list(schema["code_issue_categories"]),
        "severity": "low | medium | high | unknown",
        "confidence": 0.0,
        "evidence_quote": "verbatim substring from review.body",
        "rationale": "one short evidence-bounded sentence",
    }


def build_prompt(evidence: Mapping[str, Any], schema: Mapping[str, Any]) -> str:
    return (
        "你是实证软件工程研究的review文本标注员。只输出严格JSON，不使用外部知识。\n"
        "任务是判断human reviewer在这段文本中是否明确报告了代码concern，并做多标签分类。"
        "分类对象是reported concern，不是独立验证后的真实缺陷。不得从PR正文、diff或常识推测review未指出的问题。\n"
        f"10类定义：{json.dumps(schema['code_issue_categories'], ensure_ascii=False, separators=(',', ':'))}\n"
        f"non_code_issue定义：{schema['sentinel']['non_code_issue']}\n"
        f"严重度定义：{json.dumps(schema['severity'], ensure_ascii=False, separators=(',', ':'))}\n"
        f"边界规则：{json.dumps(schema['boundary_rules'], ensure_ascii=False, separators=(',', ':'))}\n"
        "若is_code_issue=false，categories必须为[\"non_code_issue\"]；若为true，categories至少含一个10类标签且不得含non_code_issue。"
        "evidence_quote必须逐字来自review.body。\n"
        f"输出契约：{json.dumps(output_contract(schema), ensure_ascii=False, separators=(',', ':'))}\n"
        f"输入证据：{json.dumps(evidence, ensure_ascii=False, separators=(',', ':'))}"
    )


def make_requests(prs: Iterable[Mapping[str, Any]], schema: Mapping[str, Any]) -> Iterator[ReviewRequest]:
    for pr in prs:
        repository = str((pr.get("repository") or {}).get("nameWithOwner") or "")
        for index, review in enumerate(pr.get("humanReviewTexts") or []):
            evidence = build_evidence(pr, review, index)
            review_id = evidence["review"]["id"]
            request_id = f"{pr['id']}:{review.get('source') or 'unknown'}:{review_id}"
            canonical = json.dumps(
                {
                    "prompt_version": PROMPT_VERSION,
                    "schema": schema,
                    "evidence": evidence,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            yield ReviewRequest(
                request_id=request_id,
                pr_node_id=str(pr["id"]),
                pr_url=str(pr["url"]),
                repository=repository,
                number=int(pr.get("number") or 0),
                agent=agent_name(pr),
                review_id=review_id,
                review_source=str(review.get("source") or "unknown"),
                review_url=review.get("url"),
                input_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                evidence=evidence,
                prompt=build_prompt(evidence, schema),
            )


def parse_model_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ClassificationError(f"模型输出不是有效JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ClassificationError("模型输出顶层必须是对象")
    return value


def validate_result(
    value: Mapping[str, Any], schema: Mapping[str, Any], review_body: str
) -> dict[str, Any]:
    is_code_issue = value.get("is_code_issue")
    if not isinstance(is_code_issue, bool):
        raise ClassificationError("is_code_issue必须是布尔值")
    categories = value.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ClassificationError("categories必须是非空数组")
    aliases = {
        "documentation_issue": "documentation",
        "style_convention": "other_code_issue",
        "performance": "other_code_issue",
    }
    normalized_categories = list(dict.fromkeys(aliases.get(str(item), str(item)) for item in categories))
    allowed = set(schema["code_issue_categories"])
    if is_code_issue:
        if "non_code_issue" in normalized_categories or not set(normalized_categories) <= allowed:
            raise ClassificationError(f"代码问题categories越界: {normalized_categories}")
    elif normalized_categories != ["non_code_issue"]:
        raise ClassificationError("非代码问题categories必须严格为[non_code_issue]")
    severity = str(value.get("severity") or "")
    if severity not in schema["severity"]:
        raise ClassificationError(f"非法severity: {severity}")
    try:
        confidence = float(value.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ClassificationError("confidence必须是数字") from exc
    if not 0 <= confidence <= 1:
        raise ClassificationError("confidence必须在[0,1]")
    quote = clean_text(value.get("evidence_quote"), 400)
    if not quote or normalize_for_quote(quote) not in normalize_for_quote(review_body):
        raise ClassificationError("evidence_quote不是review.body的逐字子串")
    rationale = clean_text(value.get("rationale"), 500)
    if not rationale:
        raise ClassificationError("rationale不能为空")
    return {
        "is_code_issue": is_code_issue,
        "categories": normalized_categories,
        "severity": severity,
        "confidence": round(confidence, 4),
        "evidence_quote": quote,
        "rationale": rationale,
        "normalizations": [
            f"{original}->{aliases[str(original)]}"
            for original in categories
            if str(original) in aliases
        ],
    }


def get_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT, "Content-Type": "application/json"})
        _THREAD_LOCAL.session = session
    return session


def call_model(
    item: ReviewRequest,
    schema: Mapping[str, Any],
    api_key: str,
    base_url: str,
    model: str,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "只输出满足给定契约的JSON对象。"},
            {"role": "user", "content": item.prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = get_session().post(
                base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=timeout,
            )
            if response.status_code in {408, 409, 429} or response.status_code >= 500:
                raise ClassificationError(f"模型服务临时错误 HTTP {response.status_code}")
            if response.status_code != 200:
                raise RuntimeError(
                    f"模型服务 HTTP {response.status_code}: {clean_text(response.text, 500)}"
                )
            envelope = response.json()
            choice = (envelope.get("choices") or [{}])[0]
            if choice.get("finish_reason") == "length":
                raise ClassificationError("模型输出被max_tokens截断")
            content = (choice.get("message") or {}).get("content") or ""
            return {
                "classification": validate_result(
                    parse_model_json(content), schema, item.evidence["review"]["body"]
                ),
                "usage": envelope.get("usage") or {},
                "response_id": envelope.get("id"),
            }
        except (requests.RequestException, ValueError, ClassificationError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(20.0, 2**attempt + random.random()))
    raise ClassificationError(str(last_error or "未知模型调用错误"))


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("status") == "ok" and item.get("request_id"):
                completed[str(item["request_id"])] = item
    return completed


def output_record(item: ReviewRequest, model: str, response: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "ok",
        "labeler_version": LABELER_VERSION,
        "prompt_version": PROMPT_VERSION,
        "schema_version": "1.0.0",
        "request_id": item.request_id,
        "input_hash": item.input_hash,
        "pr_node_id": item.pr_node_id,
        "pr_url": item.pr_url,
        "repository": item.repository,
        "number": item.number,
        "agent": item.agent,
        "review_id": item.review_id,
        "review_source": item.review_source,
        "review_url": item.review_url,
        "review_text_sha256": hashlib.sha256(
            item.evidence["review"]["body"].encode("utf-8")
        ).hexdigest(),
        "model": model,
        "labeled_at": datetime.now(timezone.utc).isoformat(),
        "classification": response["classification"],
        "usage": response.get("usage") or {},
        "api_response_id": response.get("response_id"),
    }


def finish_futures(
    futures: dict[Future[dict[str, Any]], ReviewRequest],
    done: set[Future[dict[str, Any]]],
    output_handle: Any,
    failure_handle: Any,
    model: str,
    counts: dict[str, int],
) -> None:
    for future in done:
        item = futures.pop(future)
        try:
            record = output_record(item, model, future.result())
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_handle.flush()
            counts["ok"] += 1
        except Exception as exc:
            failure_handle.write(json.dumps({
                "status": "error",
                "request_id": item.request_id,
                "input_hash": item.input_hash,
                "pr_node_id": item.pr_node_id,
                "review_id": item.review_id,
                "model": model,
                "error": clean_text(exc, 1_000),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False) + "\n")
            failure_handle.flush()
            counts["failed"] += 1
        finished = counts["ok"] + counts["failed"]
        if finished <= 10 or finished % 50 == 0:
            print(
                f"本轮完成 {finished}；成功 {counts['ok']}，失败 {counts['failed']}，"
                f"复用 {counts['already_completed']}",
                flush=True,
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    schema = load_schema(args.schema)
    prs = load_prs(args.input)
    all_requests = list(make_requests(prs, schema))
    selected = all_requests if args.all_reviewed else all_requests[: args.limit]
    completed = load_completed(args.output)
    pending = [
        item for item in selected
        if completed.get(item.request_id, {}).get("input_hash") != item.input_hash
    ]
    if args.dry_run:
        print(json.dumps({
            "pull_requests": len(prs),
            "pull_requests_with_human_review_text": sum(bool(pr.get("humanReviewTexts")) for pr in prs),
            "review_text_requests_total": len(all_requests),
            "selected_requests": len(selected),
            "already_completed": len(selected) - len(pending),
            "pending": len(pending),
            "max_prompt_chars": max((len(item.prompt) for item in selected), default=0),
            "model": args.model,
            "dry_run": True,
        }, ensure_ascii=False, indent=2))
        return 0
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("未设置DEEPSEEK_API_KEY；密钥只从环境变量读取")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failure_path = args.output.with_name(args.output.stem + "_failures.jsonl")
    counts = {
        "selected": len(selected),
        "already_completed": len(selected) - len(pending),
        "pending": len(pending),
        "ok": 0,
        "failed": 0,
    }
    with args.output.open("a", encoding="utf-8") as output_handle, failure_path.open(
        "a", encoding="utf-8"
    ) as failure_handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures: dict[Future[dict[str, Any]], ReviewRequest] = {}
        max_in_flight = max(1, args.workers * 2)
        for item in pending:
            futures[pool.submit(
                call_model,
                item,
                schema,
                api_key,
                args.base_url,
                args.model,
                args.max_tokens,
                args.timeout,
                args.retries,
            )] = item
            if len(futures) >= max_in_flight:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                finish_futures(futures, done, output_handle, failure_handle, args.model, counts)
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            finish_futures(futures, done, output_handle, failure_handle, args.model, counts)
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
