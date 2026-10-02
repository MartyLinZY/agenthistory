# Reproduce the manuscript analyses

Run from the repository root, with Python 3.10 or later:

```bash
python3 -m pip install -r requirements.txt
python3 analysis/run_all.py
```

The pipeline restores and verifies the archived records, then recomputes all statistics offline. It makes no API or model calls and uses the supplied task/lifecycle annotations. Outputs and logs are written to `build/analysis/`; allow approximately 5 GB of free disk space for restoration and intermediate tables. To use another location, pass `--output /path/to/output`. Keep the same output path when resuming with `--from-step` after correcting a failure. A successful run ends with `validation.json` reporting `"status": "passed"`.

## Scripts and paper results

| Script | Reproduced result | Output subdirectory |
| --- | --- | --- |
| Existing `github_all/code/analyze_corpus.py` | All 126,740 retrieved PRs, change scope, discussion indicators and completeness | `corpus/` |
| Existing `top10/code/build_highstar_dataset.py` | Independent 190,094-PR project cohort and original signal features | `projects_base/` |
| `projects.py` | Expanded PR-body signal rule; project, quarterly and monthly shares; equal-project summaries and composition sensitivity | `projects/` |
| `audit_labels.py`, `analyze_labels.py` | Annotation linkage, 126,685 annotated PRs, tasks, lifecycle phases, quarterly summaries and fixed-population discussion comparisons | `labels/` |
| `matched.py` | Repository/agent/quarter matched feature–repair scope; discussion contrasts; changed-line tail sensitivity | `matched/` |
| `extract_interaction.py`, `interaction.py`, `interaction_sensitivity.py` | Author/merger workflows, pre-merge evidence, all six supported pairwise decompositions, paired repair changes and supplementary test-path analyses | `interaction/` |
| `classify.py`, `extract_modules.py`, `modules.py`, `module_sensitivity.py` | File-role and functional-module evidence, module co-change, fixed-composition temporal comparisons and directory-only sensitivity | `modules/` |
| `prepare_figures.py`, `build_*figures.py`, `figure_io.py` | Seven manuscript figures from regenerated statistics; Matplotlib PDF/PNG export | `figures/` |
| `workflow_tables.py` | Five-agent workflow and six-pair decomposition tables in LaTeX | `tables/` |
| `developer_evidence.py` | Recent-quarter counts and scope, requirement-related discussion, source/test/configuration/documentation co-change, testing-label/path overlap | `developer_evidence/` |
| `validate_matched.py`, `validate_interaction.py`, `validate_modules.py`, `validate.py` | Independent numerical reconstruction and comparison with 37 manuscript aggregate tables | Stage directories and `validation.json` |

`projects_base/` contains intermediate results under the original signal rule. Use `projects/` for the paper's project-level results: `projects.py` additionally recognizes `Generated with [Claude Code](http(s) URL)` in PR bodies. This adds 5,773 signal-positive PRs and yields 3,205/76,617 = 4.18% of merged PRs. The intermediate 3.14% figure is not the final manuscript estimate. No commit-message scan is added.

The feature–repair scope and five agent-specific discussion contrasts form a six-test family. The supplementary feature–repair test-path comparison and paired repair-size comparison form a two-test family. The five module comparisons form a separate exploratory family. Holm corrections, 5,000 bootstrap repetitions, recorded seeds, support thresholds, and repository weighting are retained in the scripts.

The 37 small CSVs under `expected/` are the manuscript's aggregate reference outputs, not inputs to the estimators. Validation compares the regenerated tables with numeric tolerances and reconstructs selected results independently from PR records. Public repository names, PR identifiers and paths describe the research subjects. No private filesystem paths or author affiliations are required. The pipeline performs computational checks, not manual semantic adjudication of model annotations; it does not include blank reviewer templates as completed validation.

`verification_summary.json` records a successful full run, including runtime versions, relative script hashes, all 37 checked aggregate tables and the seven generated manuscript figures.
