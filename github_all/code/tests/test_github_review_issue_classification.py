from __future__ import annotations

import json

import pytest

import analyze_github_review_issue_results as analysis
import classify_github_review_issues_deepseek as labeling


def schema() -> dict:
    return labeling.load_schema(labeling.DEFAULT_SCHEMA)


def sample_prs() -> list[dict]:
    return [
        {
            "id": "PR_1",
            "url": "https://github.com/o/r/pull/1",
            "number": 1,
            "title": "Fix parser",
            "body": "Handle empty input",
            "state": "MERGED",
            "mergedAt": "2025-06-02T00:00:00Z",
            "repository": {"nameWithOwner": "o/r"},
            "selectionAssignments": [{"agent": "codex", "rank": 1}],
            "humanReviewTexts": [
                {
                    "id": "R_1",
                    "source": "inline_review_comment",
                    "body": "This crashes on empty input and needs a regression test.",
                    "url": "https://github.com/o/r/pull/1#discussion_r1",
                    "author": {"login": "alice"},
                }
            ],
            "reviewThreads": [
                {
                    "id": "T_1",
                    "path": "parser.py",
                    "line": 7,
                    "comments": [
                        {
                            "id": "R_1",
                            "body": "This crashes on empty input and needs a regression test.",
                            "diffHunk": "@@ -1 +1 @@\n-parse(x)\n+parse(x.strip())",
                        }
                    ],
                }
            ],
            "reviews": [],
            "comments": [],
        },
        {
            "id": "PR_2",
            "url": "https://github.com/o/s/pull/2",
            "number": 2,
            "title": "Docs",
            "body": "Update docs",
            "state": "OPEN",
            "mergedAt": None,
            "repository": {"nameWithOwner": "o/s"},
            "selectionAssignments": [{"agent": "jules", "rank": 1}],
            "humanReviewTexts": [],
            "reviewThreads": [],
            "reviews": [],
            "comments": [],
        },
    ]


def test_requests_preserve_review_and_inline_context() -> None:
    requests = list(labeling.make_requests(sample_prs(), schema()))
    assert len(requests) == 1
    request = requests[0]
    assert request.agent == "codex"
    assert request.evidence["review"]["context"]["path"] == "parser.py"
    assert "This crashes on empty input" in request.prompt
    assert "不得从PR正文" in request.prompt


def test_validate_multilabel_and_non_code_contract() -> None:
    body = "This crashes on empty input and needs a regression test."
    result = labeling.validate_result(
        {
            "is_code_issue": True,
            "categories": ["functional_bug", "edge_case", "insufficient_testing"],
            "severity": "medium",
            "confidence": 0.9,
            "evidence_quote": "crashes on empty input",
            "rationale": "The reviewer reports an empty-input crash and missing test.",
        },
        schema(),
        body,
    )
    assert result["categories"] == [
        "functional_bug", "edge_case", "insufficient_testing"
    ]
    non_code = labeling.validate_result(
        {
            "is_code_issue": False,
            "categories": ["non_code_issue"],
            "severity": "unknown",
            "confidence": 0.8,
            "evidence_quote": "needs a regression test",
            "rationale": "Synthetic contract test.",
        },
        schema(),
        body,
    )
    assert non_code["categories"] == ["non_code_issue"]


def test_validate_rejects_invented_quote_and_mixed_sentinel() -> None:
    with pytest.raises(labeling.ClassificationError, match="逐字子串"):
        labeling.validate_result(
            {
                "is_code_issue": True,
                "categories": ["functional_bug"],
                "severity": "high",
                "confidence": 0.9,
                "evidence_quote": "invented evidence",
                "rationale": "Invalid quote.",
            },
            schema(),
            "Actual review text.",
        )
    with pytest.raises(labeling.ClassificationError, match="越界"):
        labeling.validate_result(
            {
                "is_code_issue": True,
                "categories": ["functional_bug", "non_code_issue"],
                "severity": "medium",
                "confidence": 0.9,
                "evidence_quote": "Actual review text",
                "rationale": "Mixed sentinel.",
            },
            schema(),
            "Actual review text.",
        )


def test_aggregate_covers_reviewed_and_no_review_pr(tmp_path) -> None:
    requests = list(labeling.make_requests(sample_prs(), schema()))
    item = requests[0]
    record = {
        "status": "ok",
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
        "classification": {
            "is_code_issue": True,
            "categories": ["functional_bug", "edge_case", "insufficient_testing"],
            "severity": "medium",
            "confidence": 0.9,
            "evidence_quote": "crashes on empty input",
            "rationale": "reported concern",
        },
        "usage": {"total_tokens": 100},
    }
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps(record) + "\n")
    prs, reviews, summary = analysis.aggregate(sample_prs(), labels, schema())
    assert len(prs) == 2
    assert len(reviews) == 1
    assert summary["pull_requests"] == 2
    assert summary["reviewed_pull_requests"] == 1
    assert summary["no_review_pull_requests"] == 1
    assert summary["pull_requests_with_reported_code_issue"] == 1
    no_review = next(value for value in prs if value["pr_node_id"] == "PR_2")
    assert no_review["classification_status"] == "no_human_review_text"
    assert no_review["categories"] == []


def test_wilson_interval_is_bounded() -> None:
    low, high = analysis.wilson_interval(5, 10)
    assert 0 < low < 0.5 < high < 1
    assert analysis.wilson_interval(0, 0) == (0.0, 0.0)
