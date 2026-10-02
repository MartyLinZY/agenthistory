"""Reconcile regenerated statistics with the manuscript's aggregate reference tables."""
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import statsmodels
from context import ARTIFACT, OUTPUT

expected = Path(__file__).resolve().parent / 'expected'
checked = []
for name in json.loads((expected/'manifest.json').read_text()):
    reference, actual = pd.read_csv(expected/name), pd.read_csv(OUTPUT/name)
    if name.endswith('monthly_shares_all_selected.csv'):
        actual = actual.sort_values(['cohort', 'month']).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, reference, check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-10)
    checked.append(name)
a = pd.read_csv(OUTPUT/'corpus/pr_metrics.csv')
d = pd.read_parquet(OUTPUT/'labels/labeled_metrics.parquet')
b = pd.read_parquet(OUTPUT/'projects/prs_signal_v2.parquet')
assert len(a) == a.id.nunique() == 126740
assert len(d) == d.id.nunique() == 126685
assert len(b) == b.id.nunique() == 190094
assert int(a.cross_file.sum()) == 93982
assert int(d.human_label.sum()) == 16974
assert (d.human_label == d.has_exported_text).all()
assert d.model.eq('deepseek-v4-flash').all()
assert d.task_type.value_counts()['feature'] == 53326
assert d.task_type.value_counts()['bug_fix'] == 30370
for cohort, denominator, numerator in [('created',189757,14653),('merged',76617,3205)]:
    group = b[b[cohort+'_in_window']]
    assert len(group) == denominator and group.agent_detected.sum() == numerator
    monthly = pd.read_csv(OUTPUT/'projects/monthly_shares.csv')
    m = monthly[(monthly.cohort == cohort) & (monthly.repository == 'ALL_SELECTED')]
    assert m.n.sum() == denominator and m.agent_n.sum() == numerator
    assert m.lines.sum() == group.changed_lines.sum()
    assert m.agent_lines.sum() == group.loc[group.agent_detected, 'changed_lines'].sum()
merged = b[b.merged_in_window & b.agent_detected]
assert merged.repository.isin(['openclaw/openclaw','microsoft/vscode']).sum() == 2795
pairs = pd.read_csv(OUTPUT/'interaction/workflow_gap_decomposition.csv')
assert len(pairs) == 6
assert np.allclose(pairs.gap_b_minus_a_pp, pairs.composition_component_pp + pairs.within_component_pp)
checks = json.loads((OUTPUT/'interaction/workflow_table_checks.json').read_text())
assert len(checks['summary']) == 5 and len(checks['unsupported_pairs']) == 4
figures = ['matched_task_scope', 'lifecycle_1pct', 'module_temporal_compact',
           'repair_temporal_narrow', 'repair_composition_narrow',
           'highstar_monthly_v2', 'line_share_tail_sensitivity']
for name in figures:
    for extension in ['pdf', 'png']:
        assert (OUTPUT/'figures'/f'{name}.{extension}').stat().st_size > 1000
# Numeric checks are separate from semantic human annotation accuracy.
report = dict(status='passed', aggregate_tables_checked=len(checked), checked_tables=checked,
              all_prs=len(a), annotated_prs=len(d), project_prs=len(b),
              workflow_pairs=len(pairs), merged_agent_prs=3205, manuscript_figures=figures,
              manual_annotation_validation='not performed by this pipeline',
              runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                           scipy=scipy.__version__, statsmodels=statsmodels.__version__),
              source_hashes={str(p.relative_to(ARTIFACT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted((ARTIFACT/'analysis').glob('*.py'))})
(OUTPUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
