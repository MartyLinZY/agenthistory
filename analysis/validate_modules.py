from context import ARTIFACT, OUTPUT, stage, raw_open
"""Independent count/weight reconstruction and manuscript integration checks."""
from pathlib import Path
import json,hashlib,re
import pandas as pd,numpy as np
from statsmodels.stats.multitest import multipletests
from classify import classify,MODULES
P=stage('modules')
f=pd.read_parquet(P/'files.parquet');h=pd.read_parquet(P/'module_file_evidence.parquet');d=pd.read_parquet(P/'pr_modules.parquet')
assert len(d)==125174 and len(f)==896100 and d.id.is_unique and d.files_complete.all()
assert not f.duplicated(['id','path']).any() and not h.duplicated(['id','path','module']).any()
assert set(h.id)<=set(d.id)
assert (d.id.map(f.groupby('id').size()).fillna(0).astype(int)==d.changed_files).all()
assert np.allclose(d.id.map(f.assign(lines=f.additions+f.deletions).groupby('id').lines.sum()).fillna(0),d.changed_lines)
summary=pd.read_csv(P/'module_summary.csv').set_index('category')
for k in MODULES:
    ids=set(h.loc[h.module.eq(k),'id'])
    assert ids==set(d.loc[d[k],'id'])
    assert len(ids)==summary.loc[k,'prs']
    assert np.isclose(100*len(ids)/len(d),summary.loc[k,'pct'])
for col in ['agent','quarter','task_type']:
    table=pd.read_csv(P/(col+'_coverage.csv'))
    for row in table.itertuples():
        eligible=d[d[col].eq(row.group)]
        assert len(eligible)==row.n and eligible[row.category].sum()==row.prs
        assert np.isclose(100*eligible[row.category].mean(),row.pct)
recent=d[d.quarter.isin(['2026Q1','2026Q2'])]
saved=pd.read_csv(P/'temporal_standardized.csv').set_index(['mode','category'])
for mode,cols in [('fixed_agent',['agent']),('fixed_agent_task',['agent','task_type'])]:
    data=recent if len(cols)==1 else recent[recent.task_type.notna()]
    groups=list(data.groupby(cols));total=len(data)
    for k in MODULES:
        rate={'2026Q1':0.,'2026Q2':0.}
        for _,g in groups:
            for q in rate:rate[q]+=100*len(g)/total*g[g.quarter.eq(q)][k].mean()
        assert np.isclose(rate['2026Q1'],saved.loc[(mode,k),'q1_pct'])
        assert np.isclose(rate['2026Q2'],saved.loc[(mode,k),'q2_pct'])
matched=pd.read_csv(P/'matched_temporal.csv');repo=pd.read_csv(P/'matched_repository_deltas.csv')
for m in [1,2,3]:
    g=matched[matched.min_prs_per_quarter.eq(m)];assert len(g)==5
    assert np.allclose(multipletests(g.p_sign,method='holm')[1],g.p_holm)
for k,g in repo.groupby('category'):
    r=matched[(matched.category==k)&matched.min_prs_per_quarter.eq(1)].iloc[0]
    assert len(g)==r.repositories and np.isclose(g.delta_pp.mean(),r.mean_delta_pp)
    assert r.positive+r.negative+r.ties==r.repositories
# Concrete failure cases discovered during path QA.
assert 'storage' not in classify('supabase/functions/organic-orchestrator/index.ts','source_candidate')
assert not classify('SharpOpenGl.Tests/UI/UIFontMetricsTests.cs','source_candidate')
assert 'security' not in classify('templates/decks/sales-security-trust.html','source_candidate')
assert 'storage' in classify('supabase/migrations/003_tighten_rls_policies.sql','source_candidate')
assert 'automation' in classify('.github/workflows/ci.yml','configuration_path')
assert not classify('src/core/lib/model.py','source_candidate')
result=dict(status='passed',eligible_prs=len(d),file_rows=len(f),module_counts_reconstructed=True,group_counts_reconstructed=True,fixed_weights_reconstructed=True,holm_family_size=5)
(P/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
