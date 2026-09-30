# File descriptions

Run `python3 merge_data.py` from the repository root first to restore `raw/prs.jsonl.gz` from the five adjacent `.partNNN` files. All smaller archives remain intact. Git LFS is not required.

| Path | Contents |
| --- | --- |
| `raw/prs.jsonl.gz` | Original PR records, including available files, commits, comments and reviews. |
| `raw/pr_labels_v2.jsonl.gz` | Original supplied annotations, model names and input hashes. |
| `raw/selections.jsonl.gz` | Selection-cohort membership and Agent assignments. |
| `raw/search_windows.jsonl.gz` | Search windows, counts, signals and truncation flags. |
| `raw/failures.jsonl.gz` | Collection failures and retry events. |
| `raw/run_metadata.json` | Original collection configuration. |
| `raw/metadata/raw_archives.json` | Source-file sizes, record counts and original-byte checksums. |
| `code/crawl_github_agent_prs.py` | AgentBench GitHub PR collector. |
| `code/github_agent_signals.example.json` | Agent-signal configuration example. |
| `code/analyze_corpus.py` | Offline analysis of the compressed PR export. |
| `code/analyze_github_delivery.py` | Path classification and analysis helpers. |
| `code/analyze_supplied_labels.py` | Summarizes supplied annotations. |
| `code/validate_raw_independent.py` | Compares raw records with regenerated analysis results. |
| `code/classify_github_review_issues_deepseek.py` | Separate review-text concern classifier; not the generator of the supplied task/phase annotations. |
| `code/github_review_issue_schema_10cats.json` | Review-text classification schema. |
| `code/analyze_github_review_issue_results.py` | Review-text classification analysis. |
| `code/tests/` | Existing code tests. |
