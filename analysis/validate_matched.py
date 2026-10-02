from context import ARTIFACT, OUTPUT, stage, raw_open
"""Reconstruct contrasts independently from row-level cached source data."""
from pathlib import Path
from collections import defaultdict
import json, hashlib, re
import numpy as np
import pandas as pd
P=stage('matched')
d=pd.read_parquet(OUTPUT/'labels/labeled_metrics.parquet')
b=pd.read_parquet(OUTPUT/'projects/prs_signal_v2.parquet')
rep=defaultdict(lambda:[0.,0]);nfeature=nrepair=0
for (repo,agent,quarter),g in d[d.task_type.isin(['feature','bug_fix'])].groupby(['repository_id','agent','quarter']):
    f=g.loc[g.task_type=='feature','changed_lines'].to_numpy();r=g.loc[g.task_type=='bug_fix','changed_lines'].to_numpy()
    if not len(f) or not len(r):continue
    delta=np.log((np.median(f)+1)/(np.median(r)+1));n=len(f)+len(r)
    rep[repo][0]+=n*delta;rep[repo][1]+=n;nfeature+=len(f);nrepair+=len(r)
saved=pd.read_csv(P/'scope_repository_agent_quarter_repositories.csv').set_index('repository_id')
assert set(rep)==set(saved.index)
for repo,(delta,n) in rep.items():assert np.isclose(delta/n,saved.loc[repo,'delta'],atol=1e-12)
values=np.array([v[0]/v[1] for v in rep.values()]);assert len(rep)==5432 and nfeature==10484 and nrepair==9595
assert (values>1e-12).sum()==4324 and (values < -1e-12).sum()==1095
assert np.isclose(np.exp(np.median(values)),3.9008195860130406)
tails=pd.read_csv(P/'tail_sensitivity.csv')
for cohort in ['created','merged']:
    f=b[b[cohort+'_in_window']].copy();n=int(np.ceil(len(f)/100));order=f.sort_values(['changed_lines','id'],ascending=[False,True]);largest=order.iloc[:n];rest=order.iloc[n:]
    for mode,g in [('raw',f),('drop_top_1pct',rest),('winsorize_99pct',f)]:
        x=g.changed_lines.to_numpy().astype(float)
        if mode=='winsorize_99pct':x=np.minimum(x,np.quantile(f.changed_lines,.99))
        expected=100*np.sum(x[g.agent_detected.to_numpy()])/np.sum(x)
        row=tails[(tails.cohort==cohort)&(tails['mode']==mode)].iloc[0]
        assert row.prs==len(g) and np.isclose(row.line_share_pct,expected,atol=1e-10)
    if cohort=='merged':
        assert n==767 and int(largest.agent_detected.sum())==36
        assert int(largest.changed_lines.sum())==28455010
        assert int(largest.loc[largest.repository=='tensorflow/tensorflow','changed_lines'].sum())==19191728
discussion=pd.read_csv(P/'discussion_comparisons.csv')
for row in discussion.itertuples():
    frame=d[d.agent==row.agent];scores=defaultdict(lambda:[0.,0]);prs=0
    for (repo,task,q),g in frame.groupby(['repository_id','task_type','quarter']):
        f=g[g.has_external_text_30d];r=g[~g.has_external_text_30d]
        if not len(f) or not len(r):continue
        scores[repo][0]+=100*(f.merged.mean()-r.merged.mean())*len(g);scores[repo][1]+=len(g);prs+=len(g)
    effect=np.mean([v[0]/v[1] for v in scores.values()])
    assert len(scores)==row.matched_repositories and prs==row.matched_prs
    assert np.isclose(effect,row.equal_repository_difference_pp,atol=1e-10)
assert discussion.matched_prs.sum()==2018
result=dict(status='passed',matched_scope_reconstructed=True,discussion_contrasts_reconstructed=True,tail_sensitivity_reconstructed=True)
(P/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
