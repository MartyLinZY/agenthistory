"""Apply the manuscript's expanded body-marker rule and summarize project shares."""
from pathlib import Path
import hashlib
import json
import re
import sys
import pandas as pd
from context import ARTIFACT, OUTPUT, stage, raw_open
sys.path.insert(0, str(ARTIFACT / 'top10/code'))
from collect_repo_contributions import agents

out = stage('projects')
b = pd.read_parquet(OUTPUT / 'projects_base/features/prs.parquet').set_index('id', drop=False)
pattern = re.compile(r'Generated\s+with\s+\[Claude\s+Code\]\(https?://[^\s)]+\)', re.I)
seen, changes, hashes = set(), [], []
for path in sorted((ARTIFACT / 'top10/raw').glob('*/prs.jsonl.gz')):
    digest = hashlib.sha256()
    with raw_open(path) as records:
        for line in records:
            digest.update(line)
            raw = json.loads(line)
            pid = raw['id']
            assert pid not in seen
            seen.add(pid)
            old, row = agents(raw), b.loc[pid]
            assert bool(old) == bool(row.agent_detected)
            assert raw['changedFiles'] == row.changed_files
            assert raw['additions'] + raw['deletions'] == row.changed_lines
            marker = pattern.search(raw.get('body') or '')
            if marker and 'claude_code' not in old:
                expanded = dict(old, claude_code='markdown_claude_body_signature')
                b.loc[pid, 'agents'] = '|'.join(sorted(expanded))
                b.loc[pid, 'evidence'] = json.dumps(expanded)
                b.loc[pid, 'agent_detected'] = True
                changes.append(dict(id=pid, repository=row.repository,
                                    previously_detected=bool(old)))
    hashes.append(dict(path=str(path.relative_to(ARTIFACT)), uncompressed_sha256=digest.hexdigest()))
assert seen == set(b.index) and len(seen) == 190094
assert len(changes) == 5786 and sum(not r['previously_detected'] for r in changes) == 5773
b = b.reset_index(drop=True)
b.to_parquet(out / 'prs_signal_v2.parquet', index=False)
pd.DataFrame(changes).to_csv(out / 'signal_v2_changes.csv', index=False)

def share(g, flag='agent_detected'):
    detected = g[g[flag]]
    return dict(n=len(g), agent_n=len(detected), pr_pct=100 * len(detected) / len(g) if len(g) else None,
                lines=int(g.changed_lines.sum()), agent_lines=int(detected.changed_lines.sum()),
                lines_pct=100 * detected.changed_lines.sum() / g.changed_lines.sum() if g.changed_lines.sum() else None)

repository, quarterly, monthly, sensitivity, fixed = [], [], [], [], []
summary = {}
for cohort in ['created', 'merged']:
    g = b[b[cohort + '_in_window']].copy()
    summary[cohort] = share(g)
    project_rows = [dict(cohort=cohort, repository=name, **share(group)) for name, group in g.groupby('repository')]
    repository.extend(project_rows)
    equal = pd.DataFrame(project_rows)
    summary[cohort].update(repo_equal_pr_pct=equal.pr_pct.mean(), repo_median_pr_pct=equal.pr_pct.median(),
                           repo_equal_lines_pct=equal.lines_pct.mean(), nonzero_repo_denominator=len(equal))
    known, unknown = g[g.source_lines_eligible], g[~g.source_lines_eligible]
    ka, kt = known.loc[known.agent_detected, 'source_churn'].sum(), known.source_churn.sum()
    ua = unknown.loc[unknown.agent_detected, 'changed_lines'].sum()
    uo = unknown.loc[~unknown.agent_detected, 'changed_lines'].sum()
    summary[cohort].update(source_eligible=len(known), source_lower_pct=100*ka/(kt+uo), source_upper_pct=100*(ka+ua)/(kt+ua))
    for scope, mask in [('all', pd.Series(True, index=g.index)),
                        ('exclude_openclaw_vscode', ~g.repository.isin(['openclaw/openclaw', 'microsoft/vscode'])),
                        ('existed_before_2025', g.repository_created_at < '2025-01-01')]:
        sensitivity.append(dict(cohort=cohort, scope=scope, **share(g[mask])))
    for flag, pattern_value in [('body_or_account', 'claude_code|jules|devin'), ('branch_signal', 'codex|copilot')]:
        g[flag] = g.agents.str.contains(pattern_value, na=False)
        sensitivity.append(dict(cohort=cohort, scope=flag, **share(g, flag)))
    for quarter, group in g.groupby(cohort + '_quarter'):
        repairs = group[group.fix_title_candidate]
        quarterly.append(dict(cohort=cohort, quarter=quarter, **share(group), fix_n=len(repairs),
                              fix_agent_n=int(repairs.agent_detected.sum()), fix_pct=100*repairs.agent_detected.mean()))
    g['month'] = g[cohort + '_at'].str[:7]
    for month, group in g.groupby('month'):
        monthly.append(dict(cohort=cohort, repository='ALL_SELECTED', month=month, **share(group)))
    for (name, month), group in g.groupby(['repository', 'month']):
        monthly.append(dict(cohort=cohort, repository=name, month=month, **share(group)))
    pd.DataFrame([dict(cohort=cohort, excluded=name, **share(g[g.repository != name]))
                  for name in sorted(g.repository.unique())]).to_csv(out / f'leave_one_out_{cohort}.csv', index=False)
    recent = g[g[cohort + '_quarter'].isin(['2026Q1', '2026Q2'])]
    cells = recent.groupby(['repository', cohort + '_quarter']).agent_detected.agg(['size', 'sum']).unstack().dropna()
    weights = cells['size'].sum(axis=1); weights /= weights.sum()
    for quarter in ['2026Q1', '2026Q2']:
        fixed.append(dict(cohort=cohort, quarter=quarter, repositories=len(cells),
                          fixed_weight_pr_pct=100*((cells['sum'][quarter]/cells['size'][quarter])*weights).sum()))
for name, records in [('repository_shares', repository), ('quarter_shares', quarterly), ('monthly_shares', monthly),
                      ('sensitivity', sensitivity), ('fixed_repository_weights', fixed)]:
    pd.DataFrame(records).to_csv(out / f'{name}.csv', index=False)
(out / 'results.json').write_text(json.dumps(summary, indent=2, default=lambda x: x.item()))
(out / 'raw_validation.json').write_text(json.dumps(dict(status='passed', rows=len(b), added_labels=len(changes),
    new_positive_prs=5773, raw_hashes=hashes, rule='Original signals plus Markdown Claude Code PR-body signature; no commit scan'), indent=2))
print(json.dumps(summary, indent=2, default=lambda x: x.item()))
