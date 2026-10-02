from context import ARTIFACT, OUTPUT, stage, raw_open
"""Post-hoc support diagnostics, retained separately from the two planned tests."""
from pathlib import Path
import json,itertools
import numpy as np
import pandas as pd
P=stage('interaction');d=pd.read_parquet(P/'enriched_metrics.parquet')
m=d[d.merged].copy();m['star_tier']=pd.cut(m.stars,[-1,10,1000,np.inf],labels=['0-10','11-1000','>1000'])
z=m.groupby(['agent','star_tier'],observed=True).agg(n=('id','size'),pre=('premerge_external_evidence','sum'),selfmerge=('same_author_merger','sum'));z['pre_pct']=100*z.pre/z.n;z.to_csv(P/'workflow_agent_stars.csv')
known=m[m.author_known & m.merger_known]
ga=known.groupby(['agent','same_author_merger']).premerge_external_evidence.agg(n='size',positive='sum',rate='mean');ga.to_csv(P/'workflow_agent_actor.csv');decomposition=[]
for a,b in itertools.combinations(sorted(known.agent.unique()),2):
    x=ga.loc[a];y=ga.loc[b]
    if set(x.index)!={False,True} or set(y.index)!={False,True}:continue
    pa=x.n/x.n.sum();pb=y.n/y.n.sum();ra=x.rate;rb=y.rate
    gap=(pb*rb).sum()-(pa*ra).sum();composition=((pb-pa)*(rb+ra)/2).sum();within=((rb-ra)*(pb+pa)/2).sum()
    assert np.isclose(gap,composition+within)
    decomposition.append(dict(agent_a=a,agent_b=b,gap_b_minus_a_pp=100*gap,composition_component_pp=100*composition,within_component_pp=100*within,composition_fraction=composition/gap if gap else None))
pd.DataFrame(decomposition).to_csv(P/'workflow_gap_decomposition.csv',index=False)
r=pd.read_csv(P/'repair_temporal_strata.csv');out=[]
for minimum in [2,3]:
    s=r[r[['q1_n','q2_n']].min(axis=1)>=minimum];g=s.groupby('repository_id')[['weighted','weight']].sum();x=g.weighted/g.weight
    out.append(dict(minimum_per_quarter=minimum,repos=len(g),prs=int(s.weight.sum()),median_ratio=float(np.exp(x.median())),positive=int((x>1e-12).sum()),negative=int((x < -1e-12).sum())))
pd.DataFrame(out).to_csv(P/'repair_temporal_sensitivity.csv',index=False)
a=d[d.task_type.isin(['feature','bug_fix'])&d.files_complete&d.touches_source.eq(True)].copy()
a['line_band']=pd.cut(a.changed_lines,[-1,50,200,1000,np.inf],labels=['0-50','51-200','201-1000','>1000']);a['file_band']=pd.cut(a.changed_files,[-1,1,5,np.inf],labels=['0-1','2-5','>5'])
keys=['repository_id','agent','quarter','line_band','file_band'];g=a.groupby(keys+['task_type'],observed=True).size().unstack().dropna()
shared=a.merge(g.index.to_frame(index=False),on=keys,validate='many_to_one');shared[['id','repository_id','agent','quarter','task_type','changed_lines','changed_files','touches_test','line_band','file_band']].to_csv(P/'testing_scope_common_support.csv',index=False)
out=[];rng=np.random.default_rng(20260929)
for label,frame,ks in [('original_context',a,keys[:3]),('common_support_context',shared,keys[:3]),('common_support_size',shared,keys)]:
    g=frame.groupby(ks+['task_type'],observed=True).agg(n=('id','size'),test=('touches_test','sum')).unstack('task_type').dropna();w=g['n'].sum(axis=1);delta=100*(g['test']['feature']/g['n']['feature']-g['test']['bug_fix']/g['n']['bug_fix'])
    s=pd.DataFrame({'w':w,'v':w*delta}).reset_index().groupby('repository_id')[['w','v']].sum();x=(s.v/s.w).to_numpy(float);vals=[]
    for _ in range(50):vals.extend(x[rng.integers(0,len(x),(100,len(x)))].mean(axis=1).tolist())
    lo,hi=np.quantile(vals,[.025,.975]);out.append(dict(comparison=label,repos=len(x),prs=int(w.sum()),difference_pp=x.mean(),ci_low=lo,ci_high=hi))
pd.DataFrame(out).to_csv(P/'testing_scope_support_contrasts.csv',index=False)
print('Workflow strata and post-hoc common-support diagnostics saved.')
