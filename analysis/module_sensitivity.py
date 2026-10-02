from context import ARTIFACT, OUTPUT, stage, raw_open
"""Support, path co-occurrence, and accounting diagnostics; retain all categories."""
from pathlib import Path
import pandas as pd,numpy as np
from classify import MODULES
P=stage('modules');pr=pd.read_parquet(P/'pr_modules.parquet');recent=pr[pr.quarter.isin(['2026Q1','2026Q2'])]
decomp=[]
for k in MODULES:
    r=recent.groupby(['agent','quarter'])[k].mean().unstack('quarter')
    n=recent.groupby(['agent','quarter']).size().unstack('quarter');w=n/n.sum(axis=0)
    c=100*((w['2026Q2']-w['2026Q1'])*(r['2026Q2']+r['2026Q1'])/2).sum()
    within=100*((r['2026Q2']-r['2026Q1'])*(w['2026Q2']+w['2026Q1'])/2).sum()
    raw=100*(recent[recent.quarter.eq('2026Q2')][k].mean()-recent[recent.quarter.eq('2026Q1')][k].mean())
    assert np.isclose(c+within,raw)
    decomp.append(dict(category=k,raw_delta_pp=raw,agent_composition_pp=c,within_agent_pp=within))
pd.DataFrame(decomp).to_csv(P/'agent_mix_decomposition.csv',index=False)
n.to_csv(P/'agent_quarter_denominators.csv')
h=pd.read_parquet(P/'module_file_evidence.parquet')
sets=h[h.module.isin(['interface','service_api'])].groupby(['id','module']).path.agg(set).unstack('module').dropna()
distinct=int(sets.apply(lambda row:bool(row['interface']-row['service_api']) and bool(row['service_api']-row['interface']),axis=1).sum())
pd.DataFrame([dict(both_prs=len(sets),distinct_exclusive_paths_prs=distinct,all_eligible_prs=len(pr),distinct_pct=100*distinct/len(pr))]).to_csv(P/'interface_service_distinct_paths.csv',index=False)
# Conservative naming sensitivity across agents and standardization.
out=[]
for col in ['agent','quarter']:
    for group,g in pr.groupby(col):
        for m in MODULES:out.append(dict(dimension=col,group=group,category=m,n=len(g),directory_only_pct=100*g[m+'_strict'].mean()))
pd.DataFrame(out).to_csv(P/'directory_only_profiles.csv',index=False)
out=[]
for mode,columns in [('agent',['agent']),('agent_task',['agent','task_type'])]:
    d=recent if mode=='agent' else recent[recent.task_type.notna()]
    n=d.groupby(columns+['quarter']).size().unstack('quarter').dropna();w=n.sum(axis=1);w=w/w.sum()
    for m in MODULES:
        rates=d.groupby(columns+['quarter'])[m+'_strict'].mean().unstack('quarter').loc[w.index]
        v=100*rates.mul(w,axis=0).sum();out.append(dict(mode=mode,category=m,q1_pct=v['2026Q1'],q2_pct=v['2026Q2'],delta_pp=v['2026Q2']-v['2026Q1']))
pd.DataFrame(out).to_csv(P/'directory_only_standardized.csv',index=False)
print(pd.DataFrame(decomp).to_string(index=False));print('Interface and service paths present:',len(sets),'Distinct exclusive path pairs:',distinct)
