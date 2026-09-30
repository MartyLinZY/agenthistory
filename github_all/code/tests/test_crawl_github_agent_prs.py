from __future__ import annotations

import json
from datetime import datetime, timezone

import crawl_github_agent_prs as crawler
import pytest


def test_search_query_uses_half_open_window() -> None:
    start = datetime(2025, 6, 1, tzinfo=timezone.utc)
    end = datetime(2025, 6, 2, tzinfo=timezone.utc)
    query = crawler.build_search_query("head:codex/", start, end, "owner/repo")
    assert "is:pr" in query
    assert "head:codex/" in query
    assert "repo:owner/repo" in query
    assert "created:2025-06-01T00:00:00Z..2025-06-01T23:59:59Z" in query


def test_iter_windows_covers_period_without_overlap() -> None:
    start = crawler.parse_datetime("2025-06-01")
    end = crawler.parse_datetime("2025-06-01T03:30:00Z")
    windows = list(crawler.iter_windows(start, end, 1))
    assert len(windows) == 4
    assert windows[0][0] == start
    assert windows[-1][1] == end
    assert all(left[1] == right[0] for left, right in zip(windows, windows[1:]))


def test_state_store_deduplicates_pr_and_preserves_matches(tmp_path) -> None:
    store = crawler.StateStore(tmp_path / "state.sqlite3")
    try:
        start = crawler.parse_datetime("2025-06-01")
        end = crawler.parse_datetime("2025-06-02")
        first_id = store.add_window("codex", "head:codex/", None, start, end)
        second_id = store.add_window("cursor", "head:cursor/", None, start, end)
        node = {
            "id": "PR_1",
            "number": 7,
            "url": "https://github.com/o/r/pull/7",
            "createdAt": "2025-06-01T01:00:00Z",
            "repository": {"nameWithOwner": "o/r"},
        }
        first = dict(store.connection.execute("SELECT * FROM windows WHERE id = ?", (first_id,)).fetchone())
        second = dict(store.connection.execute("SELECT * FROM windows WHERE id = ?", (second_id,)).fetchone())
        store.add_search_nodes(first, [node])
        store.add_search_nodes(second, [node])
        assert store.counts()["unique_prs"] == 1
        assert store.counts()["matches"] == 2
        assert store.search_matches("PR_1") == [
            {"agent": "codex", "signal": "head:codex/"},
            {"agent": "cursor", "signal": "head:cursor/"},
        ]
    finally:
        store.close()


def test_human_review_texts_exclude_bots_and_empty_bodies() -> None:
    pr = {
        "reviews": [
            {"id": "r1", "body": "Please cover the empty input.", "author": {"__typename": "User", "login": "alice"}},
            {"id": "r2", "body": "automated", "author": {"__typename": "Bot", "login": "review-bot"}},
        ],
        "comments": [
            {"id": "c1", "body": "", "author": {"__typename": "User", "login": "bob"}},
            {"id": "c2", "body": "This breaks Windows.", "author": {"__typename": "User", "login": "bob"}},
            {"id": "c3", "body": "Imported human review.", "author": {"__typename": "Mannequin", "login": "legacy-user"}},
        ],
        "reviewThreads": [
            {"id": "t1", "comments": [
                {"id": "i1", "body": "Duplicate implementation.", "author": {"__typename": "User", "login": "carol"}},
                {"id": "i2", "body": "fixed", "author": {"__typename": "Bot", "login": "helper[bot]"}},
            ]}
        ],
    }
    texts = crawler.extract_human_review_texts(pr)
    assert [item["source"] for item in texts] == [
        "review", "pr_comment", "pr_comment", "inline_review_comment"
    ]
    assert [item["id"] for item in texts] == ["r1", "c2", "c3", "i1"]


def test_human_review_texts_can_exclude_pr_author() -> None:
    pr = {
        "reviews": [
            {"id": "self", "body": "self note", "author": {"__typename": "User", "login": "owner"}},
            {"id": "peer", "body": "please fix", "author": {"__typename": "User", "login": "reviewer"}},
        ],
        "comments": [],
        "reviewThreads": [],
    }
    assert [item["id"] for item in crawler.extract_human_review_texts(
        pr, exclude_login="OWNER"
    )] == ["peer"]


def test_review_candidate_gate_does_not_filter_change_size_or_file_type() -> None:
    pr = {
        "state": "MERGED",
        "mergedAt": "2026-02-01T00:00:00Z",
        "reviewDecision": "APPROVED",
        "baseRefOid": "base",
        "headRefOid": "head",
        "author": {"__typename": "User", "login": "author"},
        "changedFiles": 500,
        "additions": 9000,
        "deletions": 4000,
        "files": [{"path": "README.md"}],
        "reviews": [{
            "id": "r1", "body": "Looks good after the fix.",
            "author": {"__typename": "User", "login": "reviewer"},
        }],
        "comments": [],
        "reviewThreads": [],
    }
    result = crawler.evaluate_review_candidate(pr)
    assert result["eligible"] is True
    assert result["reasons"] == []
    assert result["metrics"]["changedFiles"] == 500
    assert result["metrics"]["churn"] == 13000


def test_review_candidate_gate_rejects_active_unresolved_thread() -> None:
    pr = {
        "state": "MERGED", "mergedAt": "2026-02-01T00:00:00Z",
        "reviewDecision": "APPROVED", "baseRefOid": "base", "headRefOid": "head",
        "author": {"__typename": "User", "login": "author"},
        "files": [], "reviews": [], "comments": [],
        "reviewThreads": [{
            "id": "t1", "isResolved": False, "isOutdated": False,
            "comments": [{
                "id": "c1", "body": "This is still broken.",
                "author": {"__typename": "User", "login": "reviewer"},
            }],
        }],
    }
    result = crawler.evaluate_review_candidate(pr)
    assert result["eligible"] is False
    assert "active_unresolved_review_thread" in result["reasons"]


def test_github_tokens_are_recursively_redacted() -> None:
    token = "ghp_" + "a" * 36
    value = {
        "body": f"prefix {token} suffix",
        "nested": [{"token": token}],
        "unchanged": 7,
    }
    redacted = crawler.redact_github_tokens(value)
    assert token not in json.dumps(redacted)
    assert redacted["body"] == "prefix [REDACTED_GITHUB_TOKEN] suffix"
    assert redacted["nested"] == [{"token": "[REDACTED_GITHUB_TOKEN]"}]
    assert redacted["unchanged"] == 7


def test_state_store_rejects_mixed_experiment_configuration(tmp_path) -> None:
    store = crawler.StateStore(tmp_path / "state.sqlite3")
    try:
        store.ensure_configuration({"period": "a", "agents": ["codex"]})
        store.ensure_configuration({"agents": ["codex"], "period": "a"})
        try:
            store.ensure_configuration({"period": "b", "agents": ["codex"]})
        except ValueError as exc:
            assert "--output" in str(exc)
        else:
            raise AssertionError("mixed crawl configuration should be rejected")
    finally:
        store.close()


def test_state_store_prevents_concurrent_writer(tmp_path) -> None:
    first = crawler.StateStore(tmp_path / "state.sqlite3")
    try:
        with pytest.raises(RuntimeError, match="另一个爬虫进程"):
            crawler.StateStore(tmp_path / "state.sqlite3")
    finally:
        first.close()


def test_patch_failure_preserves_graphql_detail(tmp_path) -> None:
    class PatchFailureCrawler(crawler.AgentPRCrawler):
        def fetch_pr(self, node_id):
            return {"id": node_id, "reviews": [], "comments": [], "reviewThreads": []}

        def _archive_patch(self, row):
            raise RuntimeError("patch too large")

    store = crawler.StateStore(tmp_path / "state.sqlite3")
    try:
        window_id = store.add_window(
            "codex", "head:codex/", None,
            crawler.parse_datetime("2025-06-01"),
            crawler.parse_datetime("2025-06-02"),
        )
        window = dict(store.connection.execute(
            "SELECT * FROM windows WHERE id = ?", (window_id,)
        ).fetchone())
        store.add_search_nodes(window, [{
            "id": "PR_1", "number": 1, "url": "https://github.com/o/r/pull/1",
            "createdAt": "2025-06-01T00:00:00Z", "repository": {"nameWithOwner": "o/r"},
        }])
        worker = PatchFailureCrawler(
            client=object(), store=store, output_dir=tmp_path, include_patch=True
        )
        worker.collect_details()
        row = store.connection.execute(
            "SELECT status, payload_json, patch_path FROM prs WHERE node_id = 'PR_1'"
        ).fetchone()
        assert row["status"] == "complete"
        assert row["patch_path"] is None
        assert json.loads(row["payload_json"])["patchCollection"] == {
            "status": "error", "error": "patch too large"
        }
        assert store.connection.execute(
            "SELECT COUNT(*) FROM failures WHERE stage = 'patch'"
        ).fetchone()[0] == 1
    finally:
        store.close()


def test_stratified_sample_balances_days_and_caps_repositories(tmp_path) -> None:
    store = crawler.StateStore(tmp_path / "state.sqlite3")
    try:
        start = crawler.parse_datetime("2025-06-01")
        for day in range(1, 4):
            window_id = store.add_window(
                "codex",
                "head:codex/",
                None,
                crawler.parse_datetime(f"2025-06-0{day}"),
                crawler.parse_datetime(f"2025-06-0{day + 1}"),
            )
            window = dict(store.connection.execute(
                "SELECT * FROM windows WHERE id = ?", (window_id,)
            ).fetchone())
            nodes = []
            for index in range(6):
                repository = "dominant/repo" if index < 4 else f"other{day}/{index}"
                nodes.append({
                    "id": f"PR_{day}_{index}",
                    "number": day * 10 + index,
                    "url": f"https://github.com/{repository}/pull/{day * 10 + index}",
                    "createdAt": f"2025-06-0{day}T0{index}:00:00Z",
                    "repository": {"nameWithOwner": repository},
                })
            store.add_search_nodes(window, nodes)
            store.update_window(window_id, status="complete", issue_count=6, fetched_count=6)

        summary = store.freeze_stratified_sample(
            cohort="baseline-v1",
            agents=["codex"],
            per_agent=6,
            seed="fixed-seed",
            max_per_repository=2,
        )
        assert summary["unique_prs"] == 6
        assert summary["by_agent"] == [{
            "agent": "codex", "selected": 6, "repositories": 5, "days": 3
        }]
        selected = store.connection.execute(
            """
            SELECT p.repository, substr(p.created_at, 1, 10) AS day
            FROM selections s JOIN prs p ON p.node_id = s.node_id
            WHERE s.cohort = 'baseline-v1'
            """
        ).fetchall()
        assert max(
            sum(1 for row in selected if row["repository"] == repository)
            for repository in {row["repository"] for row in selected}
        ) <= 2
        assert {row["day"] for row in selected} == {
            "2025-06-01", "2025-06-02", "2025-06-03"
        }
        assert [row["node_id"] for row in store.pending_prs(cohort="baseline-v1")] == sorted(
            [row["node_id"] for row in store.pending_prs(cohort="baseline-v1")],
            key=lambda node_id: next(
                (row["repository"], row["number"])
                for row in store.pending_prs(cohort="baseline-v1")
                if row["node_id"] == node_id
            ),
        )
    finally:
        store.close()


def test_sample_cannot_freeze_before_search_finishes(tmp_path) -> None:
    store = crawler.StateStore(tmp_path / "state.sqlite3")
    try:
        store.add_window(
            "codex", "head:codex/", None,
            crawler.parse_datetime("2025-06-01"),
            crawler.parse_datetime("2025-06-02"),
        )
        try:
            store.freeze_stratified_sample("baseline-v1", ["codex"], 1, "seed", 1)
        except RuntimeError as exc:
            assert "未完成" in str(exc)
        else:
            raise AssertionError("sample must not freeze before discovery completes")
    finally:
        store.close()


class SearchClient:
    def __init__(self, issue_count: int) -> None:
        self.issue_count = issue_count

    def graphql(self, query, variables):
        return {
            "search": {
                "issueCount": self.issue_count,
                "pageInfo": {"hasNextPage": False, "endCursor": None},
                "nodes": [],
            }
        }

    def clone(self):
        return SearchClient(self.issue_count)

    def close(self):
        return None


class ReviewClient:
    def clone(self):
        return ReviewClient()

    def close(self):
        return None


class ReviewSelectionCrawler(crawler.AgentPRCrawler):
    def fetch_review_evidence(self, client, node_id):
        eligible = node_id in {"PR_2", "PR_3", "PR_4"}
        return {
            "eligible": eligible,
            "reasons": [] if eligible else ["no_non_author_human_review_text"],
            "humanReviewTexts": [{"id": f"review-{node_id}", "body": "fix verified"}]
            if eligible else [],
            "metrics": {},
        }


def test_review_enriched_selection_only_freezes_eligible_prs(tmp_path) -> None:
    store = crawler.StateStore(tmp_path / "state.sqlite3")
    try:
        window_id = store.add_window(
            "codex", "head:codex/", None,
            crawler.parse_datetime("2026-01-01"),
            crawler.parse_datetime("2026-01-02"),
        )
        window = dict(store.connection.execute(
            "SELECT * FROM windows WHERE id = ?", (window_id,)
        ).fetchone())
        nodes = [
            {
                "id": f"PR_{index}", "number": index,
                "url": f"https://github.com/o/r{index}/pull/{index}",
                "createdAt": f"2026-01-01T0{index}:00:00Z",
                "repository": {"nameWithOwner": f"o/r{index}"},
            }
            for index in range(1, 5)
        ]
        store.add_search_nodes(window, nodes)
        store.update_window(window_id, status="complete", issue_count=4, fetched_count=4)
        worker = ReviewSelectionCrawler(
            client=ReviewClient(), store=store, output_dir=tmp_path
        )
        summary = worker.freeze_review_enriched_sample(
            cohort="reviewed-v1", agents=["codex"], per_agent=2,
            seed="fixed", max_per_repository=1, review_workers=2, max_attempts=3,
        )
        assert summary["by_agent"][0]["selected"] == 2
        selected = store.connection.execute(
            "SELECT node_id FROM selections WHERE cohort='reviewed-v1' ORDER BY rank"
        ).fetchall()
        assert len(selected) == 2
        assert {row["node_id"] for row in selected} <= {"PR_2", "PR_3", "PR_4"}
        assert store.connection.execute(
            "SELECT COUNT(*) FROM review_eligibility WHERE status='eligible'"
        ).fetchone()[0] >= 2
    finally:
        store.close()


def test_dense_search_window_is_split(tmp_path) -> None:
    store = crawler.StateStore(tmp_path / "state.sqlite3")
    try:
        worker = crawler.AgentPRCrawler(
            client=SearchClient(901),
            store=store,
            output_dir=tmp_path,
            split_threshold=900,
            min_window_seconds=60,
        )
        worker.initialize_windows(
            [crawler.AgentSpec("codex", ("head:codex/",), 1)],
            crawler.parse_datetime("2025-06-01T00:00:00Z"),
            crawler.parse_datetime("2025-06-01T01:00:00Z"),
            [None],
        )
        window = store.next_window()
        assert window is not None
        worker._crawl_window(window)
        statuses = {
            row["status"]: row["n"]
            for row in store.connection.execute(
                "SELECT status, COUNT(*) AS n FROM windows GROUP BY status"
            )
        }
        assert statuses == {"pending": 2, "split": 1}
    finally:
        store.close()


def test_parallel_discovery_writes_results_serially(tmp_path) -> None:
    store = crawler.StateStore(tmp_path / "state.sqlite3")
    try:
        worker = crawler.AgentPRCrawler(
            client=SearchClient(0), store=store, output_dir=tmp_path
        )
        worker.initialize_windows(
            [crawler.AgentSpec("codex", ("head:codex/",), 1)],
            crawler.parse_datetime("2025-06-01T00:00:00Z"),
            crawler.parse_datetime("2025-06-01T04:00:00Z"),
            [None],
        )
        worker.discover(search_workers=3)
        assert [dict(row) for row in store.connection.execute(
            "SELECT status, COUNT(*) AS n FROM windows GROUP BY status"
        )] == [{"status": "complete", "n": 4}]
    finally:
        store.close()


def test_export_is_deterministic_jsonl(tmp_path) -> None:
    store = crawler.StateStore(tmp_path / "state.sqlite3")
    try:
        start = crawler.parse_datetime("2025-06-01")
        end = crawler.parse_datetime("2025-06-02")
        identifier = store.add_window("codex", "head:codex/", None, start, end)
        window = dict(store.connection.execute("SELECT * FROM windows WHERE id = ?", (identifier,)).fetchone())
        store.add_search_nodes(window, [{
            "id": "PR_1", "number": 7, "url": "https://github.com/o/r/pull/7",
            "createdAt": "2025-06-01T01:00:00Z", "repository": {"nameWithOwner": "o/r"},
        }])
        store.complete_pr("PR_1", {"id": "PR_1", "number": 7}, None)
        store.export(tmp_path / "out", {"schemaVersion": "test"})
        rows = [json.loads(line) for line in (tmp_path / "out" / "prs.jsonl").read_text().splitlines()]
        assert rows == [{"id": "PR_1", "number": 7}]
        metadata = json.loads((tmp_path / "out" / "run_metadata.json").read_text())
        assert metadata["counts"]["prs"] == {"complete": 1}
        assert metadata["credentialRedaction"] == {
            "failureOccurrences": 0,
            "patchOccurrences": 0,
            "payloadOccurrences": 0,
        }
    finally:
        store.close()


def test_export_sanitizes_existing_payload_and_patch(tmp_path) -> None:
    token = "github_pat_" + "z" * 30
    store = crawler.StateStore(tmp_path / "state.sqlite3")
    try:
        start = crawler.parse_datetime("2025-06-01")
        end = crawler.parse_datetime("2025-06-02")
        identifier = store.add_window("codex", "head:codex/", None, start, end)
        window = dict(store.connection.execute(
            "SELECT * FROM windows WHERE id = ?", (identifier,)
        ).fetchone())
        store.add_search_nodes(window, [{
            "id": "PR_1", "number": 7, "url": "https://github.com/o/r/pull/7",
            "createdAt": "2025-06-01T01:00:00Z", "repository": {"nameWithOwner": "o/r"},
        }])
        # Simulate a database created before storage-time redaction existed.
        store.connection.execute(
            "UPDATE prs SET status='complete', payload_json=?, patch_path=? WHERE node_id='PR_1'",
            (json.dumps({"body": token}), "patches/o__r/7.patch"),
        )
        store.connection.commit()
        patch = tmp_path / "out" / "patches" / "o__r" / "7.patch"
        patch.parent.mkdir(parents=True)
        patch.write_bytes(f"token={token}".encode())
        store.export(tmp_path / "out", {"schemaVersion": "test"})
        assert token not in (tmp_path / "out" / "prs.jsonl").read_text()
        assert token.encode() not in patch.read_bytes()
        metadata = json.loads((tmp_path / "out" / "run_metadata.json").read_text())
        assert metadata["credentialRedaction"]["payloadOccurrences"] == 1
        assert metadata["credentialRedaction"]["patchOccurrences"] == 1
    finally:
        store.close()
