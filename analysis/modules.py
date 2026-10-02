from context import ARTIFACT, OUTPUT, stage, raw_open
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
from classify import classify,MODULES,NAMES,DIRECTORIES,FILE_TOKENS,UI_EXT
P=stage('modules')
files=pd.read_parquet(P/'files.parquet');pr=pd.read_parquet(P/'eligible_prs.parquet')
assert pr.id.is_unique
rules=dict(version=2,module_order=MODULES,names=NAMES,directories={k:sorted(v) for k,v in DIRECTORIES.items()},file_tokens={k:sorted(v) for k,v in FILE_TOKENS.items()},ui_extensions=sorted(UI_EXT),classifier_sha256=hashlib.sha256((Path(__file__).resolve().parent/'classify.py').read_bytes()).hexdigest())
(P/'taxonomy.json').write_text(json.dumps(rules,indent=2))
rows=[];strictrows=[]
for r in files.itertuples(index=False):
    for k,e in classify(r.path,r.file_kind).items():rows.append((r.id,r.path,k,e,r.additions+r.deletions))
    for k in classify(r.path,r.file_kind,strict=True):strictrows.append((r.id,k))
hits=pd.DataFrame(rows,columns=['id','path','module','evidence','lines']);hits.to_parquet(P/'module_file_evidence.parquet',index=False)
flags=pd.crosstab(hits.id,hits.module).gt(0).reindex(columns=MODULES,fill_value=False)
strictflags=pd.crosstab(*zip(*strictrows)).gt(0).reindex(columns=MODULES,fill_value=False)
pr=pr.join(flags,on='id');pr[MODULES]=pr[MODULES].astype('boolean').fillna(False).astype(bool)
for m in MODULES:pr[m+'_strict']=pr.id.map(strictflags[m]).astype('boolean').fillna(False).astype(bool)
roles=pd.crosstab(files.id,files.file_kind).gt(0)
rolecols=list(roles.columns)
pr=pr.join(roles,on='id');pr[rolecols]=pr[rolecols].astype('boolean').fillna(False).astype(bool)
pr['functional_count']=pr[MODULES[:-1]].sum(axis=1)
pr['any_functional']=pr.functional_count.gt(0)
pr.to_parquet(P/'pr_modules.parquet',index=False)
def summary(g,cols):
    return [dict(category=k,n=len(g),prs=int(g[k].sum()),pct=100*g[k].mean()) for k in cols]
pd.DataFrame(summary(pr,rolecols)).to_csv(P/'role_summary.csv',index=False)
pd.DataFrame(summary(pr,MODULES+['any_functional'])).to_csv(P/'module_summary.csv',index=False)
for col in ['agent','quarter','task_type']:
    table=[]
    for name,g in pr.groupby(col):table.extend(dict(group=name,**r) for r in summary(g,MODULES+rolecols))
    pd.DataFrame(table).to_csv(P/(col+'_coverage.csv'),index=False)
source=pr[pr.source_candidate]
pd.DataFrame(summary(source,MODULES+['any_functional'])).to_csv(P/'source_conditional.csv',index=False)
co=[]
for i,a in enumerate(MODULES):
    for b in MODULES[i+1:]:co.append(dict(a=a,b=b,n=int((pr[a]&pr[b]).sum()),pct=100*(pr[a]&pr[b]).mean(),pct_given_a=100*pr.loc[pr[a],b].mean()))
pd.DataFrame(co).to_csv(P/'cochange.csv',index=False)

# Exact descriptions and fixed pooled weights for latest quarters.
recent=pr[pr.quarter.isin(['2026Q1','2026Q2'])];temporal=[];support=[]
for mode,groupcols in [('pooled',[]),('fixed_agent',['agent']),('fixed_agent_task',['agent','task_type'])]:
    d=recent if mode!='fixed_agent_task' else recent[recent.task_type.notna()]
    if groupcols:
        counts=d.groupby(groupcols+['quarter']).size().unstack('quarter').dropna();w=counts.sum(axis=1);w=w/w.sum()
        means=d.groupby(groupcols+['quarter'])[MODULES+rolecols].mean()
        for k in MODULES+rolecols:
            rates=means[k].unstack('quarter').loc[w.index]
            v=rates.mul(w,axis=0).sum()*100
            temporal.append(dict(mode=mode,category=k,q1_pct=v['2026Q1'],q2_pct=v['2026Q2'],delta_pp=v['2026Q2']-v['2026Q1']))
        support.append(dict(mode=mode,cells=len(w),q1_n=int(counts['2026Q1'].sum()),q2_n=int(counts['2026Q2'].sum()),excluded_n=len(d)-int(counts.to_numpy().sum())))
    else:
        v=d.groupby('quarter')[MODULES+rolecols].mean()*100
        for k in MODULES+rolecols:temporal.append(dict(mode=mode,category=k,q1_pct=v.loc['2026Q1',k],q2_pct=v.loc['2026Q2',k],delta_pp=v.loc['2026Q2',k]-v.loc['2026Q1',k]))
pd.DataFrame(temporal).to_csv(P/'temporal_standardized.csv',index=False)
pd.DataFrame(support).to_csv(P/'standardization_support.csv',index=False)

# All five module tests belong to one exploratory family.
t=recent.groupby(['repository_id','agent','quarter'])[MODULES].mean().unstack('quarter').dropna()
n=recent.groupby(['repository_id','agent','quarter']).size().unstack('quarter').loc[t.index]
w=n.sum(axis=1);results=[];details=[];rng=np.random.default_rng(20260928)
for k in MODULES:
    delta=100*(t[k]['2026Q2']-t[k]['2026Q1'])
    s=pd.DataFrame(dict(delta_pp=delta,weight=w,q1_n=n['2026Q1'],q2_n=n['2026Q2'])).reset_index();s['weighted']=s.delta_pp*s.weight
    for minimum in [1,2,3]:
        z=s[s[['q1_n','q2_n']].min(axis=1)>=minimum];g=z.groupby('repository_id')[['weight','weighted']].sum();x=g.weighted/g.weight
        boot=[]
        for _ in range(50):boot.extend(x.to_numpy()[rng.integers(0,len(x),(100,len(x)))].mean(axis=1))
        lo,hi=np.quantile(boot,[.025,.975]);pos=int((x>1e-10).sum());neg=int((x < -1e-10).sum())
        results.append(dict(category=k,min_prs_per_quarter=minimum,repositories=len(x),strata=len(z),prs=int(z.weight.sum()),q1_n=int(z.q1_n.sum()),q2_n=int(z.q2_n.sum()),mean_delta_pp=x.mean(),ci_low=lo,ci_high=hi,positive=pos,negative=neg,ties=len(x)-pos-neg,p_sign=binomtest(pos,pos+neg).pvalue if pos+neg else 1))
        if minimum==1:
            details.append(pd.DataFrame(dict(repository_id=g.index,category=k,delta_pp=x.values,prs=g.weight.values)))
r=pd.DataFrame(results)
for minimum in [1,2,3]:
    mask=r.min_prs_per_quarter.eq(minimum);r.loc[mask,'p_holm']=multipletests(r.loc[mask,'p_sign'],method='holm')[1]
r.to_csv(P/'matched_temporal.csv',index=False);pd.concat(details).to_csv(P/'matched_repository_deltas.csv',index=False)
strict=[]
for k in MODULES:
    for quarter,g in recent.groupby('quarter'):
        strict.append(dict(category=k,quarter=quarter,n=len(g),broad_n=int(g[k].sum()),directory_only_n=int(g[k+'_strict'].sum()),broad_pct=100*g[k].mean(),directory_only_pct=100*g[k+'_strict'].mean()))
pd.DataFrame(strict).to_csv(P/'directory_sensitivity.csv',index=False)
# Fixed reproducible examples support manual inspection; no human accuracy score is inferred.
examples=[]
for k,g in hits.groupby('module'):
    sample=g.drop_duplicates('path').sample(n=min(24,g.path.nunique()),random_state=20260928)
    examples.append(sample.merge(pr[['id','repository','url']],on='id',validate='many_to_one'))
pd.concat(examples).to_csv(P/'path_audit_sample.csv',index=False)
info=dict(complete_prs=len(pr),source_prs=len(source),functional_identified_prs=int(pr.any_functional.sum()),functional_identified_source_pct=100*source.any_functional.mean(),multiple_functional_prs=int(pr.functional_count.ge(2).sum()),no_functional_source_prs=int((pr.source_candidate&~pr.any_functional).sum()),tests=5,bootstrap=5000,seed=20260928,latest_quarter_counts=recent.quarter.value_counts().to_dict())
(P/'results.json').write_text(json.dumps(info,indent=2))
print(json.dumps(info,indent=2));print(pd.read_csv(P/'module_summary.csv').to_string(index=False));print(pd.DataFrame(temporal).query('category in @MODULES').to_string(index=False));print(r.query('min_prs_per_quarter==1').to_string(index=False))
