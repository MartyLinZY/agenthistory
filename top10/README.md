# File descriptions

All archives in this directory are smaller than 90 MB and are stored intact. No merging or Git LFS is required for these files.

| Path | Contents |
| --- | --- |
| `raw/<owner>__<repo>/prs.jsonl.gz` | Compressed original PR records for each repository. |
| `raw/*/acquisition_status.json` | Collection status and coverage. |
| `raw/yt-dlp__yt-dlp/rest_recovery_audit.json` | Recovery records and unresolved PR identities. |
| `raw/metadata/repositories.json` | Selected repositories and original repository metadata. |
| `raw/metadata/selection_protocol.json` | Collection and selection settings. |
| `raw/metadata/high_star_search_raw.json` | Original search entries for the selected repositories. |
| `raw/metadata/engineering_evidence.json` | Collected repository-structure evidence. |
| `raw/metadata/initial_counts.json` | Initial collection counts. |
| `raw/metadata/raw_archives.json` | Archive paths and original-byte checksums. |
| `code/build_highstar_dataset.py` | Rebuilds features and analyses directly from the raw archives; takes `--source` and `--output`. |
| `code/add_highstar_monthly_analysis.py` | Generates monthly analyses from a rebuilt dataset. |
| `code/collect_repo_contributions.py` | Collection implementation and Agent-signal rules. |
| `code/analyze_github_delivery.py` | Path classification and analysis helpers. |
| `code/restore_raw.py` | Optional extraction of compressed JSONL. |
| `code/verify_package.py` | Validates a rebuilt dataset specified by `--dataset`. |
