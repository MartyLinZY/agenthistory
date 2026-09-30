#!/usr/bin/env python3
"""Crawl observable Coding Agent pull requests with GitHub GraphQL.

The crawler separates discovery from evidence collection, persists every state
transition in SQLite, and exports deterministic JSONL snapshots. Tokens are read
only from GITHUB_TOKEN or GH_TOKEN.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import requests


GRAPHQL_URL = "https://api.github.com/graphql"
REST_URL = "https://api.github.com"
SEARCH_RESULT_CAP = 1_000
DEFAULT_SPLIT_THRESHOLD = 900
DEFAULT_OUTPUT = Path("outputs/github_agent_prs")
CORE_AGENTS = ("codex", "copilot", "claude_code", "jules", "devin")
GITHUB_TOKEN_PATTERN = re.compile(
    r"(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"
)
GITHUB_TOKEN_BYTES_PATTERN = re.compile(
    rb"(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"
)
REDACTED_GITHUB_TOKEN = "[REDACTED_GITHUB_TOKEN]"


@dataclass(frozen=True)
class AgentSpec:
    name: str
    signals: tuple[str, ...]
    window_hours: float
    group: str = "core"


DEFAULT_AGENT_SPECS: dict[str, AgentSpec] = {
    "codex": AgentSpec("codex", ("head:codex/",), 1),
    "copilot": AgentSpec("copilot", ("head:copilot/",), 5),
    "claude_code": AgentSpec(
        "claude_code",
        (
            '"Co-Authored-By: Claude" in:body',
            '"Generated with Claude Code" in:body',
        ),
        5,
    ),
    "jules": AgentSpec("jules", ("author:google-labs-jules[bot]",), 24),
    "devin": AgentSpec("devin", ("author:devin-ai-integration[bot]",), 24),
    "cursor": AgentSpec("cursor", ("head:cursor/",), 5, group="extension"),
}


SEARCH_QUERY = """
query SearchAgentPRs($query: String!, $cursor: String) {
  rateLimit { cost remaining resetAt }
  search(query: $query, type: ISSUE, first: 100, after: $cursor) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on PullRequest {
        id
        number
        url
        createdAt
        repository { id nameWithOwner url }
      }
    }
  }
}
"""


PR_DETAIL_QUERY = """
query AgentPRDetail($id: ID!, $pageSize: Int!) {
  rateLimit { cost remaining resetAt }
  node(id: $id) {
    ... on PullRequest {
      id number url title body state createdAt updatedAt closedAt mergedAt
      additions deletions changedFiles
      baseRefName baseRefOid headRefName headRefOid
      isCrossRepository isDraft maintainerCanModify mergeable mergeStateStatus reviewDecision
      author { __typename login url }
      mergedBy { __typename login url }
      repository {
        id nameWithOwner url isArchived isFork isPrivate
        stargazerCount forkCount diskUsage
        primaryLanguage { name }
        owner { __typename login url }
      }
      headRepository { id nameWithOwner url isArchived isFork isPrivate }
      mergeCommit { oid url committedDate }
      labels(first: 100) { nodes { id name color description } }
      commits(first: $pageSize) {
        totalCount pageInfo { hasNextPage endCursor }
        nodes {
          commit {
            oid url authoredDate committedDate
            messageHeadline messageBody additions deletions changedFilesIfAvailable
            author { name email date user { __typename login url } }
            committer { name email date user { __typename login url } }
            parents(first: 10) { totalCount nodes { oid } }
          }
        }
      }
      files(first: $pageSize) {
        totalCount pageInfo { hasNextPage endCursor }
        nodes { path additions deletions changeType }
      }
      comments(first: $pageSize) {
        totalCount pageInfo { hasNextPage endCursor }
        nodes {
          id url body createdAt updatedAt authorAssociation
          author { __typename login url }
        }
      }
      reviews(first: $pageSize) {
        totalCount pageInfo { hasNextPage endCursor }
        nodes {
          id url state body createdAt updatedAt submittedAt authorAssociation
          author { __typename login url }
          commit { oid }
        }
      }
      reviewThreads(first: $pageSize) {
        totalCount pageInfo { hasNextPage endCursor }
        nodes {
          id isResolved isOutdated path line originalLine startLine diffSide
          resolvedBy { __typename login url }
          comments(first: $pageSize) {
            totalCount pageInfo { hasNextPage endCursor }
            nodes {
              id url body createdAt updatedAt authorAssociation
              author { __typename login url }
              replyTo { id }
              pullRequestReview { id }
              commit { oid }
              originalCommit { oid }
              diffHunk
            }
          }
        }
      }
      closingIssuesReferences(first: $pageSize) {
        totalCount pageInfo { hasNextPage endCursor }
        nodes {
          id number url title body state createdAt closedAt
          author { __typename login url }
          repository { id nameWithOwner url }
        }
      }
    }
  }
}
"""


REVIEW_ELIGIBILITY_QUERY = """
query ReviewEligibility($id: ID!, $pageSize: Int!) {
  rateLimit { cost remaining resetAt }
  node(id: $id) {
    ... on PullRequest {
      id state mergedAt reviewDecision additions deletions changedFiles
      baseRefOid headRefOid
      author { __typename login url }
      files(first: 100) {
        totalCount
        nodes { path additions deletions changeType }
      }
      comments(first: $pageSize) {
        totalCount pageInfo { hasNextPage endCursor }
        nodes {
          id url body createdAt updatedAt authorAssociation
          author { __typename login url }
        }
      }
      reviews(first: $pageSize) {
        totalCount pageInfo { hasNextPage endCursor }
        nodes {
          id url state body createdAt updatedAt submittedAt authorAssociation
          author { __typename login url }
          commit { oid }
        }
      }
      reviewThreads(first: $pageSize) {
        totalCount pageInfo { hasNextPage endCursor }
        nodes {
          id isResolved isOutdated path line originalLine startLine diffSide
          resolvedBy { __typename login url }
          comments(first: $pageSize) {
            totalCount pageInfo { hasNextPage endCursor }
            nodes {
              id url body createdAt updatedAt authorAssociation
              author { __typename login url }
              replyTo { id }
              pullRequestReview { id }
              commit { oid }
              originalCommit { oid }
              diffHunk
            }
          }
        }
      }
    }
  }
}
"""


CONNECTION_QUERIES: dict[str, str] = {
    "commits": """
query MoreCommits($id: ID!, $pageSize: Int!, $cursor: String) {
  rateLimit { cost remaining resetAt }
  node(id: $id) { ... on PullRequest {
    commits(first: $pageSize, after: $cursor) {
      totalCount pageInfo { hasNextPage endCursor }
      nodes { commit {
        oid url authoredDate committedDate messageHeadline messageBody
        additions deletions changedFilesIfAvailable
        author { name email date user { __typename login url } }
        committer { name email date user { __typename login url } }
        parents(first: 10) { totalCount nodes { oid } }
      } }
    }
  } }
}
""",
    "files": """
query MoreFiles($id: ID!, $pageSize: Int!, $cursor: String) {
  rateLimit { cost remaining resetAt }
  node(id: $id) { ... on PullRequest {
    files(first: $pageSize, after: $cursor) {
      totalCount pageInfo { hasNextPage endCursor }
      nodes { path additions deletions changeType }
    }
  } }
}
""",
    "comments": """
query MoreComments($id: ID!, $pageSize: Int!, $cursor: String) {
  rateLimit { cost remaining resetAt }
  node(id: $id) { ... on PullRequest {
    comments(first: $pageSize, after: $cursor) {
      totalCount pageInfo { hasNextPage endCursor }
      nodes {
        id url body createdAt updatedAt authorAssociation
        author { __typename login url }
      }
    }
  } }
}
""",
    "reviews": """
query MoreReviews($id: ID!, $pageSize: Int!, $cursor: String) {
  rateLimit { cost remaining resetAt }
  node(id: $id) { ... on PullRequest {
    reviews(first: $pageSize, after: $cursor) {
      totalCount pageInfo { hasNextPage endCursor }
      nodes {
        id url state body createdAt updatedAt submittedAt authorAssociation
        author { __typename login url } commit { oid }
      }
    }
  } }
}
""",
    "reviewThreads": """
query MoreReviewThreads($id: ID!, $pageSize: Int!, $cursor: String) {
  rateLimit { cost remaining resetAt }
  node(id: $id) { ... on PullRequest {
    reviewThreads(first: $pageSize, after: $cursor) {
      totalCount pageInfo { hasNextPage endCursor }
      nodes {
        id isResolved isOutdated path line originalLine startLine diffSide
        resolvedBy { __typename login url }
        comments(first: $pageSize) {
          totalCount pageInfo { hasNextPage endCursor }
          nodes {
            id url body createdAt updatedAt authorAssociation
            author { __typename login url }
            replyTo { id } pullRequestReview { id }
            commit { oid } originalCommit { oid } diffHunk
          }
        }
      }
    }
  } }
}
""",
    "closingIssuesReferences": """
query MoreClosingIssues($id: ID!, $pageSize: Int!, $cursor: String) {
  rateLimit { cost remaining resetAt }
  node(id: $id) { ... on PullRequest {
    closingIssuesReferences(first: $pageSize, after: $cursor) {
      totalCount pageInfo { hasNextPage endCursor }
      nodes {
        id number url title body state createdAt closedAt
        author { __typename login url }
        repository { id nameWithOwner url }
      }
    }
  } }
}
""",
}


THREAD_COMMENTS_QUERY = """
query MoreThreadComments($id: ID!, $pageSize: Int!, $cursor: String) {
  rateLimit { cost remaining resetAt }
  node(id: $id) { ... on PullRequestReviewThread {
    comments(first: $pageSize, after: $cursor) {
      totalCount pageInfo { hasNextPage endCursor }
      nodes {
        id url body createdAt updatedAt authorAssociation
        author { __typename login url }
        replyTo { id } pullRequestReview { id }
        commit { oid } originalCommit { oid } diffHunk
      }
    }
  } }
}
"""


HEAD_CHECKS_QUERY = """
query HeadChecks($id: ID!, $pageSize: Int!, $cursor: String) {
  rateLimit { cost remaining resetAt }
  node(id: $id) { ... on PullRequest {
    headRef {
      id name
      target { ... on Commit {
        oid
        statusCheckRollup {
          state
          contexts(first: $pageSize, after: $cursor) {
            totalCount pageInfo { hasNextPage endCursor }
            nodes {
              __typename
              ... on CheckRun {
                id name status conclusion detailsUrl startedAt completedAt
                checkSuite { id app { name slug } }
              }
              ... on StatusContext {
                id context state description targetUrl createdAt
                creator { __typename login url }
              }
            }
          }
        }
      } }
    }
  } }
}
"""


class CrawlError(RuntimeError):
    """A recoverable crawl failure."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_datetime(value: str) -> datetime:
    text = value.strip()
    if len(text) == 10:
        text += "T00:00:00Z"
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0)


def github_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iter_windows(start: datetime, end: datetime, hours: float) -> Iterator[tuple[datetime, datetime]]:
    if start >= end:
        raise ValueError("start must be earlier than end")
    step = timedelta(hours=hours)
    if step.total_seconds() < 1:
        raise ValueError("window_hours must be at least one second")
    current = start
    while current < end:
        next_end = min(current + step, end)
        yield current, next_end
        current = next_end


def window_id(agent: str, signal: str, repository: str | None, start: str, end: str) -> str:
    raw = "\0".join((agent, signal, repository or "", start, end))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def build_search_query(
    signal: str,
    start: datetime,
    end: datetime,
    repository: str | None = None,
    extra: str = "",
) -> str:
    if start >= end:
        raise ValueError("start must be earlier than end")
    inclusive_end = end - timedelta(seconds=1)
    terms = [
        "is:pr",
        "is:public",
        signal.strip(),
        f"created:{github_timestamp(start)}..{github_timestamp(inclusive_end)}",
    ]
    if repository:
        terms.append(f"repo:{repository}")
    if extra.strip():
        terms.append(extra.strip())
    return " ".join(term for term in terms if term)


def is_human_actor(actor: Mapping[str, Any] | None) -> bool:
    if not actor:
        return False
    login = str(actor.get("login") or "").lower()
    actor_type = str(actor.get("__typename") or "").lower()
    return bool(login) and actor_type != "bot" and not login.endswith("[bot]")


def extract_human_review_texts(
    pr: Mapping[str, Any], exclude_login: str | None = None
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    excluded = (exclude_login or "").strip().lower()

    def append(source: str, item: Mapping[str, Any]) -> None:
        body = str(item.get("body") or "").strip()
        author = item.get("author")
        login = str((author or {}).get("login") or "").lower() if isinstance(author, Mapping) else ""
        if (
            body
            and isinstance(author, Mapping)
            and is_human_actor(author)
            and (not excluded or login != excluded)
        ):
            result.append({
                "source": source,
                "id": item.get("id"),
                "url": item.get("url"),
                "author": dict(author),
                "authorAssociation": item.get("authorAssociation"),
                "createdAt": item.get("createdAt") or item.get("submittedAt"),
                "body": body,
            })

    for review in pr.get("reviews") or []:
        append("review", review)
    for comment in pr.get("comments") or []:
        append("pr_comment", comment)
    for thread in pr.get("reviewThreads") or []:
        for comment in thread.get("comments") or []:
            enriched = dict(comment)
            enriched["threadId"] = thread.get("id")
            append("inline_review_comment", enriched)
    return result


SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cs", ".css", ".dart", ".ex", ".exs", ".go",
    ".h", ".hpp", ".html", ".java", ".js", ".jsx", ".kt", ".kts", ".lua",
    ".m", ".mm", ".php", ".pl", ".py", ".r", ".rb", ".rs", ".scala",
    ".sh", ".sql", ".swift", ".ts", ".tsx", ".vue", ".xml",
}


def is_source_or_test_path(path: str) -> bool:
    normalized = path.strip().lower()
    if not normalized:
        return False
    suffix = Path(normalized).suffix
    return suffix in SOURCE_SUFFIXES or any(
        marker in f"/{normalized}" for marker in ("/test/", "/tests/", "/spec/", "/specs/")
    )


def evaluate_review_candidate(pr: Mapping[str, Any]) -> dict[str, Any]:
    author = pr.get("author") if isinstance(pr.get("author"), Mapping) else {}
    author_login = str(author.get("login") or "")
    human_texts = extract_human_review_texts(pr, exclude_login=author_login)
    changed_files = int(pr.get("changedFiles") or 0)
    churn = int(pr.get("additions") or 0) + int(pr.get("deletions") or 0)
    files = pr.get("files") or []
    active_threads = [
        thread for thread in (pr.get("reviewThreads") or [])
        if not bool(thread.get("isResolved")) and not bool(thread.get("isOutdated"))
    ]
    reasons: list[str] = []
    if pr.get("state") != "MERGED" or not pr.get("mergedAt"):
        reasons.append("not_merged")
    if pr.get("reviewDecision") != "APPROVED":
        reasons.append("not_approved")
    if not human_texts:
        reasons.append("no_non_author_human_review_text")
    if active_threads:
        reasons.append("active_unresolved_review_thread")
    if not pr.get("baseRefOid") or not pr.get("headRefOid"):
        reasons.append("missing_base_or_head_oid")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "humanReviewTexts": human_texts,
        "metrics": {
            "changedFiles": changed_files,
            "churn": churn,
            "activeUnresolvedReviewThreads": len(active_threads),
            "sourceOrTestFiles": sum(
                is_source_or_test_path(str(item.get("path") or "")) for item in files
            ),
        },
    }


def redact_github_tokens(value: Any) -> Any:
    """Recursively redact GitHub token-shaped strings from collected content."""
    if isinstance(value, str):
        return GITHUB_TOKEN_PATTERN.sub(REDACTED_GITHUB_TOKEN, value)
    if isinstance(value, Mapping):
        return {key: redact_github_tokens(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_github_tokens(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_github_tokens(item) for item in value)
    return value


def load_agent_specs(path: Path | None) -> dict[str, AgentSpec]:
    if path is None:
        return dict(DEFAULT_AGENT_SPECS)
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("agents") if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise ValueError("agent config must be a list or contain an 'agents' list")
    result: dict[str, AgentSpec] = {}
    for entry in entries:
        name = str(entry["name"]).strip()
        signals = tuple(str(value).strip() for value in entry["signals"] if str(value).strip())
        if not name or not signals:
            raise ValueError("every agent requires a non-empty name and signals")
        result[name] = AgentSpec(
            name=name,
            signals=signals,
            window_hours=float(entry.get("window_hours", 24)),
            group=str(entry.get("group", "custom")),
        )
    return result


class StateStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock_handle = path.with_suffix(path.suffix + ".lock").open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_handle.close()
            raise RuntimeError(f"另一个爬虫进程正在使用状态库: {path}") from exc
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS windows (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                signal TEXT NOT NULL,
                repository TEXT,
                start_at TEXT NOT NULL,
                end_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                cursor TEXT,
                issue_count INTEGER,
                fetched_count INTEGER NOT NULL DEFAULT 0,
                truncated INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS prs (
                node_id TEXT PRIMARY KEY,
                number INTEGER NOT NULL,
                url TEXT NOT NULL,
                repository TEXT NOT NULL,
                created_at TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                payload_json TEXT,
                patch_path TEXT,
                error TEXT,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS matches (
                node_id TEXT NOT NULL,
                window_id TEXT NOT NULL,
                agent TEXT NOT NULL,
                signal TEXT NOT NULL,
                PRIMARY KEY (node_id, window_id),
                FOREIGN KEY (node_id) REFERENCES prs(node_id),
                FOREIGN KEY (window_id) REFERENCES windows(id)
            );
            CREATE TABLE IF NOT EXISTS failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stage TEXT NOT NULL,
                subject TEXT NOT NULL,
                error TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS selections (
                cohort TEXT NOT NULL,
                node_id TEXT NOT NULL,
                agent TEXT NOT NULL,
                rank INTEGER NOT NULL,
                selected_at TEXT NOT NULL,
                PRIMARY KEY (cohort, agent, node_id),
                FOREIGN KEY (node_id) REFERENCES prs(node_id)
            );
            CREATE TABLE IF NOT EXISTS review_eligibility (
                node_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                human_text_count INTEGER,
                evidence_json TEXT,
                error TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (node_id) REFERENCES prs(node_id)
            );
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_windows_status ON windows(status, start_at);
            CREATE INDEX IF NOT EXISTS idx_prs_status ON prs(status, repository, number);
            CREATE INDEX IF NOT EXISTS idx_selections_cohort ON selections(cohort, agent, rank);
            CREATE INDEX IF NOT EXISTS idx_review_eligibility_status
                ON review_eligibility(status, attempts);
            """
        )
        self.connection.commit()

    def ensure_configuration(self, metadata: Mapping[str, Any]) -> None:
        canonical = json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        existing = self.connection.execute(
            "SELECT value FROM metadata WHERE key = 'crawl_configuration'"
        ).fetchone()
        if existing is not None and existing["value"] != canonical:
            raise ValueError(
                "输出目录已绑定到另一组采集参数；请复用原参数，或为新范围指定新的 --output"
            )
        self.connection.execute(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES ('crawl_configuration', ?)",
            (canonical,),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()
        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
        self._lock_handle.close()

    def add_window(
        self,
        agent: str,
        signal: str,
        repository: str | None,
        start: datetime,
        end: datetime,
    ) -> str:
        start_text, end_text = github_timestamp(start), github_timestamp(end)
        identifier = window_id(agent, signal, repository, start_text, end_text)
        self.connection.execute(
            """
            INSERT OR IGNORE INTO windows
            (id, agent, signal, repository, start_at, end_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (identifier, agent, signal, repository, start_text, end_text, utc_now()),
        )
        self.connection.commit()
        return identifier

    def next_window(self) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT * FROM windows
            WHERE status IN ('pending', 'running', 'partial', 'failed')
            ORDER BY start_at, agent, signal, repository
            LIMIT 1
            """
        ).fetchone()

    def next_windows(self, limit: int) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT * FROM windows
            WHERE status IN ('pending', 'running', 'partial', 'failed')
            ORDER BY start_at, agent, signal, repository
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    def update_window(self, identifier: str, **values: Any) -> None:
        allowed = {"status", "cursor", "issue_count", "fetched_count", "truncated", "error"}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unsupported window columns: {sorted(unknown)}")
        values["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in values)
        self.connection.execute(
            f"UPDATE windows SET {assignments} WHERE id = ?",  # columns are allow-listed
            (*values.values(), identifier),
        )
        self.connection.commit()

    def add_search_nodes(self, window: Mapping[str, Any], nodes: Iterable[Mapping[str, Any]]) -> int:
        added = 0
        now = utc_now()
        with self.connection:
            for node in nodes:
                repository = (node.get("repository") or {}).get("nameWithOwner")
                if not node.get("id") or not repository or not node.get("number") or not node.get("url"):
                    continue
                before = self.connection.total_changes
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO prs
                    (node_id, number, url, repository, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (node["id"], node["number"], node["url"], repository, node.get("createdAt"), now),
                )
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO matches(node_id, window_id, agent, signal)
                    VALUES (?, ?, ?, ?)
                    """,
                    (node["id"], window["id"], window["agent"], window["signal"]),
                )
                added += int(self.connection.total_changes > before)
        return added

    def search_matches(self, node_id: str) -> list[dict[str, str]]:
        rows = self.connection.execute(
            """
            SELECT DISTINCT agent, signal FROM matches
            WHERE node_id = ? ORDER BY agent, signal
            """,
            (node_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def stratified_candidates(self, agent: str, seed: str) -> list[sqlite3.Row]:
        candidates = self.connection.execute(
            """
            SELECT DISTINCT p.node_id, p.number, p.url, p.repository, p.created_at
            FROM prs p
            JOIN matches m ON m.node_id = p.node_id
            WHERE m.agent = ? AND p.created_at IS NOT NULL
            """,
            (agent,),
        ).fetchall()
        by_day: dict[str, list[sqlite3.Row]] = {}
        for candidate in candidates:
            try:
                day = parse_datetime(candidate["created_at"]).date().isoformat()
            except (TypeError, ValueError):
                continue
            by_day.setdefault(day, []).append(candidate)
        for day, rows in by_day.items():
            rows.sort(
                key=lambda row: hashlib.sha256(
                    f"{seed}\0{agent}\0{day}\0{row['node_id']}".encode("utf-8")
                ).digest()
            )

        ordered: list[sqlite3.Row] = []
        indices = {day: 0 for day in by_day}
        days = sorted(by_day)
        while True:
            progress = False
            for day in days:
                index = indices[day]
                if index < len(by_day[day]):
                    ordered.append(by_day[day][index])
                    indices[day] = index + 1
                    progress = True
            if not progress:
                break
        return ordered

    def freeze_stratified_sample(
        self,
        cohort: str,
        agents: Sequence[str],
        per_agent: int,
        seed: str,
        max_per_repository: int,
    ) -> dict[str, Any]:
        unfinished = self.connection.execute(
            """
            SELECT COUNT(*) FROM windows
            WHERE status NOT IN ('complete', 'split')
            """
        ).fetchone()[0]
        if unfinished:
            raise RuntimeError(f"仍有 {unfinished} 个搜索窗口未完成，不能冻结研究样本")

        existing = self.connection.execute(
            "SELECT COUNT(*) FROM selections WHERE cohort = ?", (cohort,)
        ).fetchone()[0]
        if existing:
            return self.selection_summary(cohort)

        selected_at = utc_now()
        with self.connection:
            for agent in agents:
                candidates = self.connection.execute(
                    """
                    SELECT DISTINCT p.node_id, p.number, p.url, p.repository, p.created_at
                    FROM prs p
                    JOIN matches m ON m.node_id = p.node_id
                    WHERE m.agent = ? AND p.created_at IS NOT NULL
                    """,
                    (agent,),
                ).fetchall()
                by_day: dict[str, list[sqlite3.Row]] = {}
                for candidate in candidates:
                    try:
                        day = parse_datetime(candidate["created_at"]).date().isoformat()
                    except (TypeError, ValueError):
                        continue
                    by_day.setdefault(day, []).append(candidate)
                for day, rows in by_day.items():
                    rows.sort(
                        key=lambda row: hashlib.sha256(
                            f"{seed}\0{agent}\0{day}\0{row['node_id']}".encode("utf-8")
                        ).digest()
                    )

                day_indices = {day: 0 for day in by_day}
                repository_counts: dict[str, int] = {}
                chosen: list[sqlite3.Row] = []
                chosen_ids: set[str] = set()
                days = sorted(by_day)
                while len(chosen) < per_agent:
                    progress = False
                    for day in days:
                        rows = by_day[day]
                        index = day_indices[day]
                        while index < len(rows):
                            candidate = rows[index]
                            index += 1
                            repository = candidate["repository"]
                            if candidate["node_id"] in chosen_ids:
                                continue
                            if repository_counts.get(repository, 0) >= max_per_repository:
                                continue
                            chosen.append(candidate)
                            chosen_ids.add(candidate["node_id"])
                            repository_counts[repository] = repository_counts.get(repository, 0) + 1
                            progress = True
                            break
                        day_indices[day] = index
                        if len(chosen) >= per_agent:
                            break
                    if not progress:
                        break

                for rank, candidate in enumerate(chosen, start=1):
                    self.connection.execute(
                        """
                        INSERT INTO selections(cohort, node_id, agent, rank, selected_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (cohort, candidate["node_id"], agent, rank, selected_at),
                    )
        return self.selection_summary(cohort)

    def selection_assignments(self, node_id: str, cohort: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT cohort, agent, rank FROM selections
            WHERE node_id = ? AND cohort = ? ORDER BY agent, rank
            """,
            (node_id, cohort),
        ).fetchall()
        return [dict(row) for row in rows]

    def selection_summary(self, cohort: str) -> dict[str, Any]:
        rows = self.connection.execute(
            """
            SELECT s.agent,
                   COUNT(*) AS selected,
                   COUNT(DISTINCT p.repository) AS repositories,
                   COUNT(DISTINCT substr(p.created_at, 1, 10)) AS days
            FROM selections s
            JOIN prs p ON p.node_id = s.node_id
            WHERE s.cohort = ?
            GROUP BY s.agent ORDER BY s.agent
            """,
            (cohort,),
        ).fetchall()
        unique_prs = self.connection.execute(
            "SELECT COUNT(DISTINCT node_id) FROM selections WHERE cohort = ?", (cohort,)
        ).fetchone()[0]
        return {"cohort": cohort, "unique_prs": unique_prs, "by_agent": [dict(row) for row in rows]}

    def ensure_search_complete(self) -> None:
        unfinished = self.connection.execute(
            "SELECT COUNT(*) FROM windows WHERE status NOT IN ('complete', 'split')"
        ).fetchone()[0]
        if unfinished:
            raise RuntimeError(f"仍有 {unfinished} 个搜索窗口未完成，不能构建研究样本")

    def review_eligibility_rows(self, node_ids: Sequence[str]) -> dict[str, sqlite3.Row]:
        if not node_ids:
            return {}
        placeholders = ",".join("?" for _ in node_ids)
        rows = self.connection.execute(
            f"SELECT * FROM review_eligibility WHERE node_id IN ({placeholders})",
            list(node_ids),
        ).fetchall()
        return {row["node_id"]: row for row in rows}

    def start_review_checks(self, node_ids: Sequence[str]) -> None:
        if not node_ids:
            return
        now = utc_now()
        with self.connection:
            for node_id in node_ids:
                self.connection.execute(
                    """
                    INSERT INTO review_eligibility(node_id, status, attempts, updated_at)
                    VALUES (?, 'running', 1, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        status='running', attempts=attempts+1, error=NULL, updated_at=excluded.updated_at
                    """,
                    (node_id, now),
                )

    def complete_review_check(self, node_id: str, evidence: Mapping[str, Any]) -> None:
        sanitized = redact_github_tokens(evidence)
        texts = sanitized.get("humanReviewTexts") or []
        status = "eligible" if bool(sanitized.get("eligible")) else "ineligible"
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO review_eligibility
                    (node_id, status, attempts, human_text_count, evidence_json, error, updated_at)
                VALUES (?, ?, 1, ?, ?, NULL, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    status=excluded.status,
                    human_text_count=excluded.human_text_count,
                    evidence_json=excluded.evidence_json,
                    error=NULL,
                    updated_at=excluded.updated_at
                """,
                (
                    node_id,
                    status,
                    len(texts),
                    json.dumps(sanitized, ensure_ascii=False, sort_keys=True),
                    utc_now(),
                ),
            )

    def fail_review_check(self, node_id: str, error: str) -> None:
        message = error[:10_000]
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO review_eligibility(node_id, status, attempts, error, updated_at)
                VALUES (?, 'failed', 1, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    status='failed', error=excluded.error, updated_at=excluded.updated_at
                """,
                (node_id, message, utc_now()),
            )
            self.connection.execute(
                "INSERT INTO failures(stage, subject, error, created_at) VALUES (?, ?, ?, ?)",
                ("review_eligibility", node_id, message, utc_now()),
            )

    def review_evidence(self, node_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT evidence_json FROM review_eligibility WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        if row is None or not row["evidence_json"]:
            return None
        return json.loads(row["evidence_json"])

    def selected_for_agent(self, cohort: str, agent: str) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT s.rank, p.* FROM selections s JOIN prs p ON p.node_id = s.node_id
            WHERE s.cohort = ? AND s.agent = ? ORDER BY s.rank
            """,
            (cohort, agent),
        ).fetchall()

    def add_selection(self, cohort: str, agent: str, node_id: str) -> None:
        rank = self.connection.execute(
            "SELECT COUNT(*) + 1 FROM selections WHERE cohort = ? AND agent = ?",
            (cohort, agent),
        ).fetchone()[0]
        self.connection.execute(
            """
            INSERT OR IGNORE INTO selections(cohort, node_id, agent, rank, selected_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (cohort, node_id, agent, rank, utc_now()),
        )
        self.connection.commit()

    def pending_prs(
        self,
        limit: int | None = None,
        max_attempts: int = 3,
        cohort: str | None = None,
    ) -> list[sqlite3.Row]:
        sql = """
            SELECT DISTINCT p.* FROM prs p
        """
        parameters: list[Any] = []
        if cohort:
            sql += " JOIN selections s ON s.node_id = p.node_id AND s.cohort = ?"
            parameters.append(cohort)
        sql += " WHERE p.status != 'complete' AND p.attempts < ? ORDER BY p.repository, p.number"
        parameters.append(max_attempts)
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(limit)
        return self.connection.execute(sql, parameters).fetchall()

    def start_pr(self, node_id: str) -> None:
        self.connection.execute(
            """
            UPDATE prs SET status = 'running', attempts = attempts + 1,
                           error = NULL, updated_at = ? WHERE node_id = ?
            """,
            (utc_now(), node_id),
        )
        self.connection.commit()

    def complete_pr(self, node_id: str, payload: Mapping[str, Any], patch_path: str | None) -> None:
        sanitized_payload = redact_github_tokens(payload)
        self.connection.execute(
            """
            UPDATE prs SET status = 'complete', payload_json = ?, patch_path = ?,
                           error = NULL, updated_at = ? WHERE node_id = ?
            """,
            (
                json.dumps(sanitized_payload, ensure_ascii=False, sort_keys=True),
                patch_path,
                utc_now(),
                node_id,
            ),
        )
        self.connection.commit()

    def fail(self, stage: str, subject: str, error: str) -> None:
        message = error[:10_000]
        with self.connection:
            self.connection.execute(
                "INSERT INTO failures(stage, subject, error, created_at) VALUES (?, ?, ?, ?)",
                (stage, subject, message, utc_now()),
            )
            if stage == "detail":
                self.connection.execute(
                    "UPDATE prs SET status = 'failed', error = ?, updated_at = ? WHERE node_id = ?",
                    (message, utc_now(), subject),
                )

    def counts(self) -> dict[str, Any]:
        def grouped(table: str) -> dict[str, int]:
            rows = self.connection.execute(
                f"SELECT status, COUNT(*) AS n FROM {table} GROUP BY status"  # fixed internal names
            ).fetchall()
            return {row["status"]: row["n"] for row in rows}

        return {
            "windows": grouped("windows"),
            "prs": grouped("prs"),
            "unique_prs": self.connection.execute("SELECT COUNT(*) FROM prs").fetchone()[0],
            "matches": self.connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0],
            "selections": {
                row["cohort"]: row["n"]
                for row in self.connection.execute(
                    "SELECT cohort, COUNT(*) AS n FROM selections GROUP BY cohort"
                ).fetchall()
            },
            "reviewEligibility": grouped("review_eligibility"),
            "failures": self.connection.execute("SELECT COUNT(*) FROM failures").fetchone()[0],
        }

    def redact_stored_credentials(self, output_dir: Path) -> dict[str, int]:
        payloads_redacted = 0
        failure_messages_redacted = 0
        patch_files_redacted = 0
        with self.connection:
            for row in self.connection.execute(
                "SELECT node_id, payload_json FROM prs WHERE payload_json IS NOT NULL"
            ).fetchall():
                original = row["payload_json"]
                sanitized, count = GITHUB_TOKEN_PATTERN.subn(REDACTED_GITHUB_TOKEN, original)
                if count:
                    self.connection.execute(
                        "UPDATE prs SET payload_json = ? WHERE node_id = ?",
                        (sanitized, row["node_id"]),
                    )
                    payloads_redacted += count
            for row in self.connection.execute(
                "SELECT node_id, evidence_json FROM review_eligibility WHERE evidence_json IS NOT NULL"
            ).fetchall():
                sanitized, count = GITHUB_TOKEN_PATTERN.subn(
                    REDACTED_GITHUB_TOKEN, row["evidence_json"]
                )
                if count:
                    self.connection.execute(
                        "UPDATE review_eligibility SET evidence_json = ? WHERE node_id = ?",
                        (sanitized, row["node_id"]),
                    )
                    payloads_redacted += count
            for row in self.connection.execute("SELECT id, error FROM failures").fetchall():
                sanitized, count = GITHUB_TOKEN_PATTERN.subn(
                    REDACTED_GITHUB_TOKEN, row["error"]
                )
                if count:
                    self.connection.execute(
                        "UPDATE failures SET error = ? WHERE id = ?", (sanitized, row["id"])
                    )
                    failure_messages_redacted += count

        patch_root = output_dir / "patches"
        if patch_root.exists():
            replacement = REDACTED_GITHUB_TOKEN.encode("ascii")
            for patch in patch_root.rglob("*.patch"):
                original = patch.read_bytes()
                sanitized, count = GITHUB_TOKEN_BYTES_PATTERN.subn(replacement, original)
                if count:
                    temporary = patch.with_suffix(".patch.tmp")
                    temporary.write_bytes(sanitized)
                    temporary.replace(patch)
                    patch_files_redacted += count
        return {
            "payloadOccurrences": payloads_redacted,
            "failureOccurrences": failure_messages_redacted,
            "patchOccurrences": patch_files_redacted,
        }

    def export(self, output_dir: Path, metadata: Mapping[str, Any]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        redaction_summary = self.redact_stored_credentials(output_dir)
        exports = {
            "prs.jsonl": self.connection.execute(
                "SELECT payload_json FROM prs WHERE status = 'complete' ORDER BY repository, number"
            ),
            "search_windows.jsonl": self.connection.execute(
                "SELECT * FROM windows ORDER BY start_at, agent, signal, repository"
            ),
            "failures.jsonl": self.connection.execute(
                "SELECT stage, subject, error, created_at FROM failures ORDER BY id"
            ),
            "selections.jsonl": self.connection.execute(
                """
                SELECT s.cohort, s.agent, s.rank, s.selected_at,
                       p.node_id, p.repository, p.number, p.url, p.created_at
                FROM selections s JOIN prs p ON p.node_id = s.node_id
                ORDER BY s.cohort, s.agent, s.rank
                """
            ),
            "review_eligibility.jsonl": self.connection.execute(
                """
                SELECT r.node_id, p.repository, p.number, p.url, p.created_at,
                       r.status, r.attempts, r.human_text_count,
                       r.evidence_json, r.error, r.updated_at
                FROM review_eligibility r JOIN prs p ON p.node_id = r.node_id
                ORDER BY p.repository, p.number
                """
            ),
        }
        for name, rows in exports.items():
            target = output_dir / name
            temporary = target.with_suffix(target.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                for row in rows:
                    if name == "prs.jsonl":
                        handle.write(row["payload_json"] + "\n")
                    elif name == "review_eligibility.jsonl":
                        record = dict(row)
                        evidence = record.pop("evidence_json")
                        record["evidence"] = json.loads(evidence) if evidence else None
                        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    else:
                        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(target)
        full_metadata = dict(metadata)
        full_metadata.update({
            "exported_at": utc_now(),
            "counts": self.counts(),
            "credentialRedaction": redaction_summary,
        })
        target = output_dir / "run_metadata.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(full_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)


class GitHubClient:
    def __init__(
        self,
        token: str,
        timeout: int = 60,
        retries: int = 5,
        wait_on_rate_limit: bool = True,
        session: requests.Session | None = None,
    ) -> None:
        if not token:
            raise ValueError("GitHub token is required")
        self._token = token
        self.timeout = timeout
        self.retries = retries
        self.wait_on_rate_limit = wait_on_rate_limit
        self.session = session or requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "agent-pr-benchmark-research-crawler/1.0",
        })

    def clone(self) -> "GitHubClient":
        return GitHubClient(
            token=self._token,
            timeout=self.timeout,
            retries=self.retries,
            wait_on_rate_limit=self.wait_on_rate_limit,
        )

    def close(self) -> None:
        self.session.close()

    def graphql(self, query: str, variables: Mapping[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.session.post(
                    GRAPHQL_URL,
                    json={"query": query, "variables": dict(variables)},
                    timeout=self.timeout,
                )
                if response.status_code in {429, 502, 503, 504}:
                    raise CrawlError(f"transient HTTP {response.status_code}")
                response.raise_for_status()
                envelope = response.json()
                errors = envelope.get("errors") or []
                if errors:
                    messages = "; ".join(str(item.get("message", item)) for item in errors)
                    if "rate limit" in messages.lower() and self.wait_on_rate_limit:
                        self._wait_for_reset(response.headers.get("X-RateLimit-Reset"))
                        continue
                    raise CrawlError(f"GraphQL errors: {messages}")
                data = envelope.get("data")
                if not isinstance(data, dict):
                    raise CrawlError("GraphQL response has no data object")
                self._respect_rate_limit(data.get("rateLimit"))
                return data
            except (requests.RequestException, ValueError, CrawlError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                delay = min(60.0, (2**attempt) + random.random())
                time.sleep(delay)
        raise CrawlError(str(last_error or "unknown GraphQL failure"))

    def _respect_rate_limit(self, rate: Mapping[str, Any] | None) -> None:
        if not rate or int(rate.get("remaining") or 0) >= 50:
            return
        if self.wait_on_rate_limit:
            self._wait_for_reset(str(rate.get("resetAt") or ""))
        else:
            raise CrawlError(f"GraphQL rate limit low: remaining={rate.get('remaining')}")

    @staticmethod
    def _wait_for_reset(reset: str | None) -> None:
        if not reset:
            time.sleep(60)
            return
        try:
            if reset.isdigit():
                reset_at = datetime.fromtimestamp(int(reset), timezone.utc)
            else:
                reset_at = parse_datetime(reset)
            seconds = max(1.0, (reset_at - datetime.now(timezone.utc)).total_seconds() + 2)
        except (ValueError, OverflowError):
            seconds = 60
        time.sleep(seconds)

    def download_patch(self, repository: str, number: int) -> bytes:
        url = f"{REST_URL}/repos/{repository}/pulls/{number}"
        response = self.session.get(
            url,
            headers={"Accept": "application/vnd.github.patch"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.content


class AgentPRCrawler:
    def __init__(
        self,
        client: GitHubClient,
        store: StateStore,
        output_dir: Path,
        split_threshold: int = DEFAULT_SPLIT_THRESHOLD,
        min_window_seconds: int = 60,
        collection_limit: int = 100,
        include_checks: bool = True,
        include_patch: bool = False,
        search_extra: str = "",
    ) -> None:
        self.client = client
        self.store = store
        self.output_dir = output_dir
        self.split_threshold = split_threshold
        self.min_window_seconds = min_window_seconds
        self.collection_limit = collection_limit
        self.include_checks = include_checks
        self.include_patch = include_patch
        self.search_extra = search_extra

    def initialize_windows(
        self,
        specs: Sequence[AgentSpec],
        start: datetime,
        end: datetime,
        repositories: Sequence[str | None],
    ) -> int:
        count = 0
        for spec in specs:
            for signal in spec.signals:
                for repository in repositories:
                    for left, right in iter_windows(start, end, spec.window_hours):
                        self.store.add_window(spec.name, signal, repository, left, right)
                        count += 1
        return count

    def discover(self, search_workers: int = 1) -> None:
        clients = [self.client] + [self.client.clone() for _ in range(search_workers - 1)]
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=search_workers) as executor:
                while windows := self.store.next_windows(search_workers):
                    futures: dict[concurrent.futures.Future[dict[str, Any]], sqlite3.Row] = {}
                    for index, window in enumerate(windows):
                        self.store.update_window(window["id"], status="running", error=None)
                        futures[executor.submit(self._fetch_window, clients[index], window)] = window
                    first_error: Exception | None = None
                    for future in concurrent.futures.as_completed(futures):
                        window = futures[future]
                        try:
                            data = future.result()
                            self._apply_window_result(window, data)
                        except Exception as exc:  # keep later windows resumable
                            self.store.update_window(
                                window["id"], status="failed", error=str(exc)[:10_000]
                            )
                            self.store.fail("search", window["id"], str(exc))
                            first_error = first_error or exc
                    if first_error is not None:
                        raise first_error
        finally:
            for client in clients[1:]:
                client.close()

    def _crawl_window(self, window: sqlite3.Row) -> None:
        self.store.update_window(window["id"], status="running", error=None)
        data = self._fetch_window(self.client, window)
        self._apply_window_result(window, data)

    def _fetch_window(self, client: GitHubClient, window: sqlite3.Row) -> dict[str, Any]:
        start, end = parse_datetime(window["start_at"]), parse_datetime(window["end_at"])
        query_text = build_search_query(
            window["signal"], start, end, window["repository"], self.search_extra
        )
        return client.graphql(SEARCH_QUERY, {"query": query_text, "cursor": window["cursor"]})

    def _apply_window_result(self, window: sqlite3.Row, data: Mapping[str, Any]) -> None:
        start, end = parse_datetime(window["start_at"]), parse_datetime(window["end_at"])
        search = data["search"]
        issue_count = int(search.get("issueCount") or 0)
        duration = int((end - start).total_seconds())
        if window["cursor"] is None and issue_count > self.split_threshold and duration > self.min_window_seconds:
            midpoint = start + timedelta(seconds=duration // 2)
            self.store.add_window(window["agent"], window["signal"], window["repository"], start, midpoint)
            self.store.add_window(window["agent"], window["signal"], window["repository"], midpoint, end)
            self.store.update_window(
                window["id"], status="split", issue_count=issue_count, error=None
            )
            return

        nodes = [node for node in search.get("nodes") or [] if node]
        self.store.add_search_nodes(window, nodes)
        fetched_count = int(window["fetched_count"] or 0) + len(nodes)
        page_info = search.get("pageInfo") or {}
        has_next = bool(page_info.get("hasNextPage"))
        cap_reached = fetched_count >= SEARCH_RESULT_CAP
        if has_next and not cap_reached:
            self.store.update_window(
                window["id"],
                status="partial",
                cursor=page_info.get("endCursor"),
                issue_count=issue_count,
                fetched_count=fetched_count,
                truncated=0,
                error=None,
            )
            return
        truncated = int(has_next or issue_count > fetched_count)
        self.store.update_window(
            window["id"],
            status="complete",
            cursor=None,
            issue_count=issue_count,
            fetched_count=fetched_count,
            truncated=truncated,
            error=None,
        )

    def fetch_review_evidence(
        self, client: GitHubClient, node_id: str
    ) -> dict[str, Any]:
        data = client.graphql(REVIEW_ELIGIBILITY_QUERY, {"id": node_id, "pageSize": 100})
        node = data.get("node")
        if not isinstance(node, dict):
            raise CrawlError(f"PR node is unavailable during review check: {node_id}")

        for name in ("comments", "reviews", "reviewThreads"):
            connection = node.pop(name, None) or {}
            items = list(connection.get("nodes") or [])
            page_info = connection.get("pageInfo") or {}
            items = self._fetch_remaining_connection(
                node_id,
                name,
                items,
                page_info.get("endCursor"),
                bool(page_info.get("hasNextPage")),
                client=client,
            )
            node[name] = items

        files = node.pop("files", None) or {}
        node["files"] = list(files.get("nodes") or [])
        for thread in node.get("reviewThreads") or []:
            comments = thread.get("comments") or {}
            if isinstance(comments, dict):
                thread["comments"] = self._fetch_remaining_thread_comments(
                    thread["id"],
                    list(comments.get("nodes") or []),
                    (comments.get("pageInfo") or {}).get("endCursor"),
                    bool((comments.get("pageInfo") or {}).get("hasNextPage")),
                    client=client,
                )
        evidence = evaluate_review_candidate(node)
        evidence["checkedAt"] = utc_now()
        evidence["gate"] = {
            "search": "is:merged review:approved",
            "requiresNonAuthorHumanText": True,
            "requiresNoActiveUnresolvedThread": True,
            "requiresBaseAndHeadOid": True,
            "changeSizeFilter": None,
            "fileTypeFilter": None,
        }
        return evidence

    def freeze_review_enriched_sample(
        self,
        cohort: str,
        agents: Sequence[str],
        per_agent: int,
        seed: str,
        max_per_repository: int,
        review_workers: int,
        max_attempts: int,
    ) -> dict[str, Any]:
        self.store.ensure_search_complete()
        clients = [self.client] + [self.client.clone() for _ in range(review_workers - 1)]
        try:
            for agent in agents:
                selected = self.store.selected_for_agent(cohort, agent)
                selected_ids = {row["node_id"] for row in selected}
                repository_counts: dict[str, int] = {}
                for row in selected:
                    repository_counts[row["repository"]] = (
                        repository_counts.get(row["repository"], 0) + 1
                    )
                if len(selected) >= per_agent:
                    continue

                ordered = self.store.stratified_candidates(agent, seed)
                cursor = 0
                batch_size = max(review_workers, review_workers * 4)
                while len(selected_ids) < per_agent and cursor < len(ordered):
                    batch: list[sqlite3.Row] = []
                    while cursor < len(ordered) and len(batch) < batch_size:
                        candidate = ordered[cursor]
                        cursor += 1
                        if candidate["node_id"] in selected_ids:
                            continue
                        if repository_counts.get(candidate["repository"], 0) >= max_per_repository:
                            continue
                        batch.append(candidate)
                    if not batch:
                        continue

                    statuses = self.store.review_eligibility_rows(
                        [row["node_id"] for row in batch]
                    )
                    pending: list[sqlite3.Row] = []
                    for candidate in batch:
                        status = statuses.get(candidate["node_id"])
                        if status is None or status["status"] in {"pending", "running"}:
                            pending.append(candidate)
                        elif status["status"] == "failed":
                            if int(status["attempts"] or 0) >= max_attempts:
                                raise CrawlError(
                                    f"review eligibility exhausted retries for {candidate['node_id']}"
                                )
                            pending.append(candidate)

                    if pending:
                        self.store.start_review_checks([row["node_id"] for row in pending])
                        first_error: Exception | None = None
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=review_workers
                        ) as executor:
                            futures = {
                                executor.submit(
                                    self.fetch_review_evidence,
                                    clients[index % len(clients)],
                                    candidate["node_id"],
                                ): candidate
                                for index, candidate in enumerate(pending)
                            }
                            for future in concurrent.futures.as_completed(futures):
                                candidate = futures[future]
                                try:
                                    self.store.complete_review_check(
                                        candidate["node_id"], future.result()
                                    )
                                except Exception as exc:
                                    self.store.fail_review_check(candidate["node_id"], str(exc))
                                    first_error = first_error or exc
                        if first_error is not None:
                            raise first_error

                    statuses = self.store.review_eligibility_rows(
                        [row["node_id"] for row in batch]
                    )
                    for candidate in batch:
                        if len(selected_ids) >= per_agent:
                            break
                        status = statuses.get(candidate["node_id"])
                        if status is None or status["status"] != "eligible":
                            continue
                        repository = candidate["repository"]
                        if repository_counts.get(repository, 0) >= max_per_repository:
                            continue
                        self.store.add_selection(
                            cohort, agent, candidate["node_id"]
                        )
                        selected_ids.add(candidate["node_id"])
                        repository_counts[repository] = repository_counts.get(repository, 0) + 1
        finally:
            for client in clients[1:]:
                client.close()
        return self.store.selection_summary(cohort)

    def collect_details(
        self,
        limit: int | None = None,
        max_attempts: int = 3,
        cohort: str | None = None,
        require_human_review: bool = False,
    ) -> None:
        for row in self.store.pending_prs(limit=limit, max_attempts=max_attempts, cohort=cohort):
            node_id = row["node_id"]
            self.store.start_pr(node_id)
            try:
                payload = self.fetch_pr(node_id)
                payload["searchMatches"] = self.store.search_matches(node_id)
                if cohort:
                    payload["selectionAssignments"] = self.store.selection_assignments(node_id, cohort)
                payload["humanReviewTexts"] = extract_human_review_texts(
                    payload,
                    exclude_login=str((payload.get("author") or {}).get("login") or "")
                    if require_human_review else None,
                )
                if require_human_review:
                    evidence = self.store.review_evidence(node_id)
                    if not evidence or not evidence.get("eligible"):
                        raise CrawlError(f"selected PR lacks eligible review evidence: {node_id}")
                    payload["humanReviewTexts"] = evidence.get("humanReviewTexts") or []
                    payload["reviewEligibility"] = evidence
                patch_path = None
                if self.include_patch:
                    try:
                        patch_path = self._archive_patch(row)
                        payload["patchCollection"] = {"status": "ok", "path": patch_path}
                    except Exception as patch_error:
                        message = str(patch_error)[:2_000]
                        payload["patchCollection"] = {"status": "error", "error": message}
                        self.store.fail("patch", node_id, message)
                self.store.complete_pr(node_id, payload, patch_path)
            except Exception as exc:
                self.store.fail("detail", node_id, str(exc))

    def fetch_pr(self, node_id: str) -> dict[str, Any]:
        first_page_size = 100 if self.collection_limit == 0 else min(100, self.collection_limit)
        data = self.client.graphql(PR_DETAIL_QUERY, {"id": node_id, "pageSize": first_page_size})
        node = data.get("node")
        if not isinstance(node, dict):
            raise CrawlError(f"PR node is unavailable: {node_id}")

        truncation: dict[str, bool] = {}
        for name in CONNECTION_QUERIES:
            connection = node.pop(name, None) or {}
            items = list(connection.get("nodes") or [])
            page_info = connection.get("pageInfo") or {}
            if self.collection_limit == 0:
                items = self._fetch_remaining_connection(
                    node_id, name, items, page_info.get("endCursor"), bool(page_info.get("hasNextPage"))
                )
            truncation[name] = len(items) < int(connection.get("totalCount") or len(items))
            node[name] = items

        if self.collection_limit == 0:
            for thread in node.get("reviewThreads") or []:
                comments = thread.get("comments") or {}
                if isinstance(comments, dict):
                    thread["comments"] = self._fetch_remaining_thread_comments(
                        thread["id"],
                        list(comments.get("nodes") or []),
                        (comments.get("pageInfo") or {}).get("endCursor"),
                        bool((comments.get("pageInfo") or {}).get("hasNextPage")),
                    )
        else:
            for thread in node.get("reviewThreads") or []:
                comments = thread.get("comments") or {}
                if isinstance(comments, dict):
                    thread["comments"] = list(comments.get("nodes") or [])
                    if len(thread["comments"]) < int(comments.get("totalCount") or len(thread["comments"])):
                        truncation["reviewThreadComments"] = True

        node["checks"] = self._fetch_checks(node_id) if self.include_checks else None
        node["collection"] = {
            "collectedAt": utc_now(),
            "collectionLimit": self.collection_limit,
            "truncated": truncation,
        }
        return node

    def _fetch_remaining_connection(
        self,
        node_id: str,
        name: str,
        items: list[dict[str, Any]],
        cursor: str | None,
        has_next: bool,
        client: GitHubClient | None = None,
    ) -> list[dict[str, Any]]:
        active_client = client or self.client
        while has_next:
            data = active_client.graphql(
                CONNECTION_QUERIES[name], {"id": node_id, "pageSize": 100, "cursor": cursor}
            )
            node = data.get("node") or {}
            connection = node.get(name) or {}
            items.extend(item for item in connection.get("nodes") or [] if item)
            page_info = connection.get("pageInfo") or {}
            next_cursor = page_info.get("endCursor")
            has_next = bool(page_info.get("hasNextPage"))
            if has_next and (not next_cursor or next_cursor == cursor):
                raise CrawlError(f"pagination cursor did not advance for {name} on {node_id}")
            cursor = next_cursor
        return items

    def _fetch_remaining_thread_comments(
        self,
        thread_id: str,
        items: list[dict[str, Any]],
        cursor: str | None,
        has_next: bool,
        client: GitHubClient | None = None,
    ) -> list[dict[str, Any]]:
        active_client = client or self.client
        while has_next:
            data = active_client.graphql(
                THREAD_COMMENTS_QUERY,
                {"id": thread_id, "pageSize": 100, "cursor": cursor},
            )
            connection = (data.get("node") or {}).get("comments") or {}
            items.extend(item for item in connection.get("nodes") or [] if item)
            page_info = connection.get("pageInfo") or {}
            next_cursor = page_info.get("endCursor")
            has_next = bool(page_info.get("hasNextPage"))
            if has_next and (not next_cursor or next_cursor == cursor):
                raise CrawlError(f"thread comment cursor did not advance for {thread_id}")
            cursor = next_cursor
        return items

    def _fetch_checks(self, node_id: str) -> dict[str, Any] | None:
        cursor: str | None = None
        contexts: list[dict[str, Any]] = []
        rollup_state: str | None = None
        commit_oid: str | None = None
        while True:
            data = self.client.graphql(
                HEAD_CHECKS_QUERY, {"id": node_id, "pageSize": 100, "cursor": cursor}
            )
            head_ref = ((data.get("node") or {}).get("headRef") or {})
            commit = head_ref.get("target") or {}
            rollup = commit.get("statusCheckRollup")
            if not isinstance(rollup, dict):
                return {"headCommitOid": commit.get("oid"), "state": None, "contexts": []}
            commit_oid = commit.get("oid")
            rollup_state = rollup.get("state")
            connection = rollup.get("contexts") or {}
            contexts.extend(item for item in connection.get("nodes") or [] if item)
            page_info = connection.get("pageInfo") or {}
            if not page_info.get("hasNextPage"):
                break
            next_cursor = page_info.get("endCursor")
            if not next_cursor or next_cursor == cursor:
                raise CrawlError(f"check cursor did not advance for {node_id}")
            cursor = next_cursor
        return {"headCommitOid": commit_oid, "state": rollup_state, "contexts": contexts}

    def _archive_patch(self, row: Mapping[str, Any]) -> str:
        repository = str(row["repository"])
        number = int(row["number"])
        content = self.client.download_patch(repository, number)
        content = GITHUB_TOKEN_BYTES_PATTERN.sub(
            REDACTED_GITHUB_TOKEN.encode("ascii"), content
        )
        safe_repo = repository.replace("/", "__")
        relative = Path("patches") / safe_repo / f"{number}.patch"
        target = self.output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".patch.tmp")
        temporary.write_bytes(content)
        temporary.replace(target)
        return relative.as_posix()


def read_repositories(values: Sequence[str], repo_file: Path | None) -> list[str | None]:
    repositories = [value.strip() for value in values if value.strip()]
    if repo_file:
        for line in repo_file.read_text(encoding="utf-8").splitlines():
            clean = line.strip()
            if clean and not clean.startswith("#"):
                repositories.append(clean)
    invalid = [value for value in repositories if value.count("/") != 1 or " " in value]
    if invalid:
        raise ValueError(f"invalid owner/name repositories: {invalid}")
    return sorted(set(repositories)) or [None]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="UTC inclusive start, date or ISO-8601")
    parser.add_argument("--end", required=True, help="UTC exclusive end, date or ISO-8601")
    parser.add_argument("--agents", nargs="+", default=list(CORE_AGENTS))
    parser.add_argument("--agent-config", type=Path)
    parser.add_argument("--repository", action="append", default=[], help="owner/name; repeatable")
    parser.add_argument("--repo-file", type=Path, help="one owner/name per line")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=["all", "search", "details", "export"], default="all")
    parser.add_argument("--search-extra", default="", help="additional GitHub search qualifiers")
    parser.add_argument("--split-threshold", type=int, default=DEFAULT_SPLIT_THRESHOLD)
    parser.add_argument("--min-window-seconds", type=int, default=60)
    parser.add_argument(
        "--collection-limit",
        type=int,
        default=100,
        help="max items per nested collection; 0 fetches all pages",
    )
    parser.add_argument("--max-details", type=int, help="detail cap for a smoke run")
    parser.add_argument(
        "--details-per-agent",
        type=int,
        help="freeze and collect this many temporally balanced PRs per Agent",
    )
    parser.add_argument("--selection-name", default="baseline-v1")
    parser.add_argument("--sample-seed", default="agentbench-baseline-v1")
    parser.add_argument("--max-per-repository", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--search-workers", type=int, default=1)
    parser.add_argument("--review-workers", type=int, default=4)
    parser.add_argument(
        "--require-human-review",
        action="store_true",
        help=(
            "build an eligible-only cohort after full search; automatically adds "
            "is:merged review:approved and excludes bot/author-only or unresolved review"
        ),
    )
    parser.add_argument("--include-patch", action="store_true")
    parser.add_argument("--skip-checks", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--no-wait-rate-limit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> tuple[datetime, datetime]:
    start, end = parse_datetime(args.start), parse_datetime(args.end)
    if start >= end:
        raise ValueError("--start must be earlier than --end")
    if not 1 <= args.split_threshold <= SEARCH_RESULT_CAP:
        raise ValueError("--split-threshold must be between 1 and 1000")
    if args.min_window_seconds < 1:
        raise ValueError("--min-window-seconds must be positive")
    if args.collection_limit < 0:
        raise ValueError("--collection-limit cannot be negative")
    if args.max_details is not None and args.max_details < 1:
        raise ValueError("--max-details must be positive")
    if args.details_per_agent is not None and args.details_per_agent < 1:
        raise ValueError("--details-per-agent must be positive")
    if args.max_details is not None and args.details_per_agent is not None:
        raise ValueError("--max-details and --details-per-agent cannot be used together")
    if args.max_per_repository < 1:
        raise ValueError("--max-per-repository must be positive")
    if not 1 <= args.search_workers <= 16:
        raise ValueError("--search-workers must be between 1 and 16")
    if not 1 <= args.review_workers <= 16:
        raise ValueError("--review-workers must be between 1 and 16")
    if args.details_per_agent is not None and not args.selection_name.strip():
        raise ValueError("--selection-name cannot be empty")
    if args.require_human_review and args.details_per_agent is None:
        raise ValueError("--require-human-review requires --details-per-agent")
    return start, end


def build_metadata(
    args: argparse.Namespace,
    specs: Sequence[AgentSpec],
    start: datetime,
    end: datetime,
    repositories: Sequence[str | None],
) -> dict[str, Any]:
    return {
        "schemaVersion": "1.1.0",
        "method": "GitHub GraphQL time-stratified PR search",
        "period": {"startInclusive": github_timestamp(start), "endExclusive": github_timestamp(end)},
        "agents": [asdict(spec) for spec in specs],
        "repositories": list(repositories),
        "searchExtra": args.search_extra,
        "splitThreshold": args.split_threshold,
        "minimumWindowSeconds": args.min_window_seconds,
        "collectionLimit": args.collection_limit,
        "includeChecks": not args.skip_checks,
        "includePatch": args.include_patch,
        "selection": {
            "name": args.selection_name,
            "detailsPerAgent": args.details_per_agent,
            "seed": args.sample_seed,
            "maxPerRepositoryPerAgent": args.max_per_repository,
            "temporalStratum": "UTC calendar day",
            "requireHumanReview": args.require_human_review,
            "reviewWorkers": args.review_workers,
            "eligibilityGate": {
                "search": "is:merged review:approved" if args.require_human_review else None,
                "nonBotNonAuthorText": args.require_human_review,
                "noActiveUnresolvedThread": args.require_human_review,
                "baseAndHeadOid": args.require_human_review,
                "changeSizeFilter": None,
                "fileTypeFilter": None,
            },
        },
        "interpretationLimits": [
            "Search signals demonstrate observable Agent association, not pure Agent authorship.",
            "Merged state and successful checks are evidence fields, not correctness ground truth.",
            "A truncated search window or collection must not be treated as complete data.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        start, end = validate_args(args)
        if args.require_human_review:
            required_qualifiers = "is:merged review:approved"
            args.search_extra = " ".join(
                value for value in (args.search_extra.strip(), required_qualifiers) if value
            )
        all_specs = load_agent_specs(args.agent_config)
        unknown = sorted(set(args.agents) - set(all_specs))
        if unknown:
            raise ValueError(f"unknown agents: {unknown}; available={sorted(all_specs)}")
        specs = [all_specs[name] for name in args.agents]
        repositories = read_repositories(args.repository, args.repo_file)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    metadata = build_metadata(args, specs, start, end, repositories)
    planned_windows = sum(
        sum(1 for _ in iter_windows(start, end, spec.window_hours))
        * len(spec.signals)
        * len(repositories)
        for spec in specs
    )
    if args.dry_run:
        print(json.dumps({**metadata, "plannedInitialWindows": planned_windows}, ensure_ascii=False, indent=2))
        return 0

    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if not token and args.mode != "export":
        print("未设置 GITHUB_TOKEN 或 GH_TOKEN；密钥只从环境变量读取", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    store = StateStore(args.output / "crawl_state.sqlite3")
    try:
        store.ensure_configuration(metadata)
        if args.mode == "export":
            store.export(args.output, metadata)
            print(json.dumps(store.counts(), ensure_ascii=False, indent=2))
            return 0
        client = GitHubClient(
            token=token or "",
            timeout=args.timeout,
            retries=args.retries,
            wait_on_rate_limit=not args.no_wait_rate_limit,
        )
        crawler = AgentPRCrawler(
            client=client,
            store=store,
            output_dir=args.output,
            split_threshold=args.split_threshold,
            min_window_seconds=args.min_window_seconds,
            collection_limit=args.collection_limit,
            include_checks=not args.skip_checks,
            include_patch=args.include_patch,
            search_extra=args.search_extra,
        )
        if args.mode in {"all", "search"}:
            crawler.initialize_windows(specs, start, end, repositories)
            crawler.discover(search_workers=args.search_workers)
        if args.mode in {"all", "details"}:
            cohort = None
            if args.details_per_agent is not None:
                if args.require_human_review:
                    summary = crawler.freeze_review_enriched_sample(
                        cohort=args.selection_name,
                        agents=[spec.name for spec in specs],
                        per_agent=args.details_per_agent,
                        seed=args.sample_seed,
                        max_per_repository=args.max_per_repository,
                        review_workers=args.review_workers,
                        max_attempts=args.max_attempts,
                    )
                else:
                    summary = store.freeze_stratified_sample(
                        cohort=args.selection_name,
                        agents=[spec.name for spec in specs],
                        per_agent=args.details_per_agent,
                        seed=args.sample_seed,
                        max_per_repository=args.max_per_repository,
                    )
                print(json.dumps({"selection": summary}, ensure_ascii=False, indent=2))
                cohort = args.selection_name
            crawler.collect_details(
                limit=args.max_details,
                max_attempts=args.max_attempts,
                cohort=cohort,
                require_human_review=args.require_human_review,
            )
        store.export(args.output, metadata)
        print(json.dumps(store.counts(), ensure_ascii=False, indent=2))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
