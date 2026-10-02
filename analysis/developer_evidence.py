from context import ARTIFACT, OUTPUT, stage, raw_open
"""Verify descriptive evidence for developer lessons against existing PR records."""
from pathlib import Path
import hashlib
import json
import pandas as pd


OUT = stage('developer_evidence')
SOURCE = OUTPUT / 'labels'
records = SOURCE / 'labeled_metrics.parquet'
modules = OUTPUT / 'modules/pr_modules.parquet'
d = pd.read_parquet(records)
m = pd.read_parquet(modules)
assert len(d) == 126685 and d.id.is_unique
assert len(m) == 125174 and m.id.is_unique and m.files_complete.all()

rows = []
for name, group in [('all_annotated', d), ('feature', d[d.task_type == 'feature']), ('bug_fix', d[d.task_type == 'bug_fix'])]:
    for quarter in ['2026Q1', '2026Q2']:
        g = group[group.quarter == quarter]
        rows.append(dict(population=name, quarter=quarter, prs=len(g),
                         files_median=g.changed_files.median(), lines_median=g.changed_lines.median(),
                         discussion_prs=int(g.human_label.sum()), discussion_pct=100*g.human_label.mean(),
                         cross_file_pct=100*g.cross_file.mean()))
quarterly = pd.DataFrame(rows)
quarterly.to_csv(OUT / 'quarterly_evidence.csv', index=False)
for name, source in [('all_annotated', 'quarter.csv'), ('feature', 'task_type_quarter.csv'), ('bug_fix', 'task_type_quarter.csv')]:
    existing = pd.read_csv(SOURCE / source)
    if 'task_type' in existing:
        existing = existing[existing.task_type == name]
    for row in quarterly[quarterly.population == name].itertuples():
        old = existing[existing.quarter == row.quarter].iloc[0]
        assert row.prs == old['n'] and row.lines_median == old.lines_median
        assert row.discussion_prs == old.human_n and row.files_median == old.files_median

tag = m.phases.fillna('').str.split('|').map(lambda p: 'testing' in p)
overlap = {'testing_tag_complete_prs': int(tag.sum()),
           'with_test_path': int((tag & m.test_path).sum()),
           'without_test_path': int((tag & ~m.test_path).sum())}
assert overlap == dict(testing_tag_complete_prs=68952, with_test_path=35775, without_test_path=33177)
overlap['with_test_path_pct'] = 100 * overlap['with_test_path'] / overlap['testing_tag_complete_prs']
overlap['without_test_path_pct'] = 100 - overlap['with_test_path_pct']
role_cochange = {role: int((m.source_candidate & m[role]).sum())
                for role in ['test_path', 'configuration_path', 'documentation_path']}
assert role_cochange == dict(test_path=33048, configuration_path=21494, documentation_path=28797)
phase = pd.read_csv(SOURCE / 'phase_summary.csv').set_index('phase')
phase_counts = {}
for label in ['implementation', 'requirement_analysis', 'requirement_elicitation']:
    g = d[d.phases.str.split('|').map(lambda p: label in p)]
    assert len(g) == phase.loc[label, 'n']
    phase_counts[label] = dict(prs=len(g), coverage_pct=100*len(g)/len(d),
                               external_discussion_pct=100*g.has_external_text_30d.mean())
files = [records, modules, SOURCE/'quarter.csv', SOURCE/'task_type_quarter.csv', SOURCE/'phase_summary.csv']
report = dict(quarterly=rows, testing_overlap=overlap, phases=phase_counts, source_role_cochange=role_cochange,
              provenance=[dict(path=str(p.relative_to(OUTPUT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files],
              interpretation=dict(
                  observed='PR counts, median change scope, and counts of PRs with discussion increase in the observed Q1/Q2 populations.',
                  not_measured=['Total human effort', 'Functional points delivered', 'Net productivity change',
                                'Requirement-test semantic alignment', 'Repeated test edits caused by misunderstood requirements'],
                  writing='Use missing semantic alignment and repeated test adaptation as a conditional failure mechanism and developer advice, not a measured result.'))
(OUT/'evidence.json').write_text(json.dumps(report, indent=2)+'\n')
print(quarterly.to_string(index=False))
print(json.dumps(overlap, indent=2))
print('Record-level summaries match the existing tables.')
