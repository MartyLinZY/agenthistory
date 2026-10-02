from context import ARTIFACT, OUTPUT, stage, raw_open
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
from scipy.stats import binomtest, binom
from scipy.special import logsumexp
from statsmodels.stats.multitest import multipletests
P=stage('interaction');d=pd.read_parquet(P/'enriched_metrics.parquet')
rng=np.random.default_rng(20260928)
def ci(x,median=False):
    x=np.asarray(x,float);out=[]
    for _ in range(50):
        z=x[rng.integers(0,len(x),(100,len(x)))];out.extend((np.median(z,axis=1) if median else z.mean(axis=1)).tolist())
    return np.quantile(out,[.025,.975]).tolist()
def sign(x):
    x=np.asarray(x,float);pos=int((x>1e-12).sum());neg=int((x < -1e-12).sum());n=pos+neg
    logp=min(0.,np.log(2)+logsumexp(binom.logpmf(np.arange(min(pos,neg)+1),n,.5))) if n else 0.
    return dict(positive=pos,negative=neg,tied=len(x)-n,p_sign=float(binomtest(pos,n).pvalue) if n else 1.,log10_p_sign=float(logp/np.log(10)))

# Workflow evidence on all merged PRs; missing actor identities are explicit.
m=d[d.merged].copy();m['fast_5m']=m.merge_minutes.le(5);m['fast_1h']=m.merge_minutes.le(60)
m['postmerge_only_30d']=(m.ext_text_postmerge_30d>0)&(m.ext_text_premerge_30d==0)
m['empty_review_extra']=(m.ext_review_premerge>0)&(m.ext_text_premerge_all==0)
def workflow(g):
    known=g[g.author_known & g.merger_known]
    return dict(n=len(g),repos=g.repository_id.nunique(),known_author_merger=len(known),same_author_merger_n=int(known.same_author_merger.sum()),same_author_merger_pct=100*known.same_author_merger.mean(),median_merge_minutes=g.merge_minutes.median(),q25_minutes=g.merge_minutes.quantile(.25),q75_minutes=g.merge_minutes.quantile(.75),within5m_n=int(g.fast_5m.sum()),within5m_pct=100*g.fast_5m.mean(),within1h_n=int(g.fast_1h.sum()),within1h_pct=100*g.fast_1h.mean(),premerge_text_n=int((g.ext_text_premerge_all>0).sum()),premerge_review_n=int((g.ext_review_premerge>0).sum()),premerge_evidence_n=int(g.premerge_external_evidence.sum()),premerge_evidence_pct=100*g.premerge_external_evidence.mean(),empty_review_extra_n=int(g.empty_review_extra.sum()),text30_n=int((g.ext_text_30d_legacy>0).sum()),postmerge_only_30d_n=int(g.postmerge_only_30d.sum()))
rows=[dict(group='all',**workflow(m))]
for col in ['agent','star_bucket']:
    for key,g in m.groupby(col):rows.append(dict(group=col+':'+str(key),**workflow(g)))
pd.DataFrame(rows).to_csv(P/'merge_workflows.csv',index=False)
cross=[]
for self,g in m[m.author_known&m.merger_known].groupby('same_author_merger'):cross.append(dict(self_merge=bool(self),**workflow(g)))
pd.DataFrame(cross).to_csv(P/'merge_actor_comparison.csv',index=False)

# Testing labels and independent path evidence are different constructs.
a=d[d.task_type.notna() & d.files_complete].copy()
a['testing_label']=a.phases.str.split('|').map(lambda x:'testing' in x)
label=[]
for value,g in a.groupby('testing_label'):
    label.append(dict(testing_label=bool(value),n=len(g),test_path_n=int(g.touches_test.sum()),test_path_pct=100*g.touches_test.mean(),source_path_n=int(g.touches_source.sum())))
pd.DataFrame(label).to_csv(P/'testing_label_paths.csv',index=False)
source=a[a.touches_source.astype(bool)].copy();task=[]
for k,g in source.groupby('task_type'):
    task.append(dict(task=k,n=len(g),test_cochange_n=int(g.touches_test.sum()),test_cochange_pct=100*g.touches_test.mean(),lines_median=g.changed_lines.median()))
pd.DataFrame(task).to_csv(P/'task_test_cochange.csv',index=False)

f=source[source.task_type.isin(['feature','bug_fix'])]
t=f.groupby(['repository_id','agent','quarter','task_type']).agg(n=('id','size'),test=('touches_test','sum')).unstack('task_type').dropna()
delta=100*(t['test']['feature']/t['n']['feature']-t['test']['bug_fix']/t['n']['bug_fix']);w=t['n'].sum(axis=1)
strata=pd.DataFrame({'delta_pp':delta,'weight':w,'n_feature':t['n']['feature'],'n_repair':t['n']['bug_fix']}).reset_index();strata['weighted']=strata.delta_pp*strata.weight
repo=strata.groupby('repository_id')[['weighted','weight']].sum();repo['delta_pp']=repo.weighted/repo.weight
repo.to_csv(P/'testing_matched_repositories.csv');strata.to_csv(P/'testing_matched_strata.csv',index=False)
lo,hi=ci(repo.delta_pp)
testing=dict(repositories=len(repo),strata=len(t),prs=int(w.sum()),features=int(t['n']['feature'].sum()),repairs=int(t['n']['bug_fix'].sum()),difference_pp=repo.delta_pp.mean(),ci_low=lo,ci_high=hi,**sign(repo.delta_pp))
testsensitivity=[]
for minimum in [2,3]:
    s=strata[strata[['n_feature','n_repair']].min(axis=1)>=minimum];g=s.groupby('repository_id')[['weight','weighted']].sum();x=g.weighted/g.weight
    testsensitivity.append(dict(minimum_per_task=minimum,repositories=len(g),prs=int(s.weight.sum()),difference_pp=x.mean(),positive=int((x>1e-12).sum()),negative=int((x < -1e-12).sum())))
pd.DataFrame(testsensitivity).to_csv(P/'testing_support_sensitivity.csv',index=False)

# Paired temporal repairs: keep repo/agent support fixed over the latest quarters.
f=d[(d.task_type=='bug_fix')&d.quarter.isin(['2026Q1','2026Q2'])]
t=f.groupby(['repository_id','agent','quarter']).agg(n=('id','size'),lines=('changed_lines','median')).unstack('quarter').dropna()
delta=np.log1p(t['lines']['2026Q2'])-np.log1p(t['lines']['2026Q1']);w=t['n'].sum(axis=1)
strata=pd.DataFrame({'delta':delta,'weight':w,'q1_n':t['n']['2026Q1'],'q2_n':t['n']['2026Q2']}).reset_index();strata['weighted']=strata.delta*strata.weight
repo=strata.groupby('repository_id')[['weighted','weight']].sum();repo['delta']=repo.weighted/repo.weight
repo.to_csv(P/'repair_temporal_repositories.csv');strata.to_csv(P/'repair_temporal_strata.csv',index=False)
lo,hi=ci(repo.delta,median=True)
repair=dict(repositories=len(repo),strata=len(t),prs=int(w.sum()),q1_n=int(t['n']['2026Q1'].sum()),q2_n=int(t['n']['2026Q2'].sum()),median_ratio=float(np.exp(repo.delta.median())),ratio_ci_low=math.exp(lo),ratio_ci_high=math.exp(hi),**sign(repo.delta))
adj=multipletests([testing['p_sign'],repair['p_sign']],method='holm')[1]
for r,p in zip([testing,repair],adj):r['p_holm']=float(p)
pd.DataFrame([testing]).to_csv(P/'testing_matched_summary.csv',index=False);pd.DataFrame([repair]).to_csv(P/'repair_temporal_summary.csv',index=False)
result=dict(workflow=rows[0],actor=cross,testing_label=label,task_testing=task,testing_matched=testing,testing_sensitivity=testsensitivity,repair_matched=repair,test_family=2,bootstrap_repeats=5000,seed=20260928)
(P/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
