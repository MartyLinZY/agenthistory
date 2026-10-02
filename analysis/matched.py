from context import ARTIFACT, OUTPUT, stage, raw_open
from pathlib import Path
import json, hashlib, math
import numpy as np
import pandas as pd
from scipy.stats import binomtest, binom
from scipy.special import logsumexp
from statsmodels.stats.multitest import multipletests

P=stage('matched')
SOURCE=OUTPUT/'labels/labeled_metrics.parquet'
d=pd.read_parquet(SOURCE)
b=pd.read_parquet(OUTPUT/'projects/prs_signal_v2.parquet')
assert d.id.is_unique and len(d)==126685 and b.id.is_unique and len(b)==190094
assert not d[['repository_id','agent','quarter','task_type','merged','changed_lines','has_external_text_30d']].isna().any().any()
rng=np.random.default_rng(20260928)
def interval(x, stat='mean', repeats=5000):
    x=np.asarray(x,float); out=[]
    for start in range(0,repeats,100):
        samples=x[rng.integers(0,len(x),size=(min(100,repeats-start),len(x)))]
        out.extend((np.mean(samples,axis=1) if stat=='mean' else np.median(samples,axis=1)).tolist())
    return np.quantile(out,[.025,.975]).tolist()
def sign_summary(x):
    x=np.asarray(x,float); pos=int((x>1e-12).sum()); neg=int((x < -1e-12).sum()); nonzero=pos+neg
    logp=min(0.,float(np.log(2)+logsumexp(binom.logpmf(np.arange(min(pos,neg)+1),nonzero,.5)))) if nonzero else 0.
    return dict(positive=pos,negative=neg,tied=int(len(x)-nonzero),positive_pct=100*pos/len(x),p_sign=float(binomtest(pos,nonzero).pvalue) if nonzero else 1.,log10_p_sign=logp/np.log(10))

scope=[]
for name,keys in [('repository',['repository_id']),('repository_agent_quarter',['repository_id','agent','quarter'])]:
    f=d[d.task_type.isin(['feature','bug_fix'])]
    g=f.groupby(keys+['task_type']).agg(n=('id','size'),lines=('changed_lines','median'),files=('changed_files','median')).unstack('task_type').dropna()
    g['delta']=np.log1p(g['lines']['feature'])-np.log1p(g['lines']['bug_fix'])
    g['weight']=g['n'].sum(axis=1)
    s=pd.DataFrame({'delta':g['delta'],'weight':g['weight']}).reset_index()
    s['weighted_delta']=s.delta*s.weight
    repo=s.groupby('repository_id').agg(weight=('weight','sum'),weighted_delta=('weighted_delta','sum'),strata=('delta','size'))
    repo['delta']=repo.weighted_delta/repo.weight
    repo.to_csv(P/f'scope_{name}_repositories.csv')
    flat=g.copy();flat.columns=['_'.join(str(v) for v in c if v) if isinstance(c,tuple) else c for c in flat.columns];flat.to_csv(P/f'scope_{name}_strata.csv')
    lo,hi=interval(repo.delta,'median')
    scope.append(dict(comparison=name,repositories=len(repo),strata=len(g),prs=int(g['n'].sum().sum()),features=int(g['n']['feature'].sum()),repairs=int(g['n']['bug_fix'].sum()),median_log1p_ratio=float(np.exp(repo.delta.median())),ratio_ci_low=math.exp(lo),ratio_ci_high=math.exp(hi),**sign_summary(repo.delta)))
pd.DataFrame(scope).to_csv(P/'scope_comparisons.csv',index=False)
strict=pd.read_csv(P/'scope_repository_agent_quarter_strata.csv')
scope_support=[]
for name,frame in [('two_per_task',strict[strict[['n_bug_fix','n_feature']].min(axis=1)>=2]),('complete_file_lists',None)]:
    if frame is None:
        g=d[d.files_complete & d.task_type.isin(['feature','bug_fix'])].groupby(['repository_id','agent','quarter','task_type']).agg(n=('id','size'),lines=('changed_lines','median')).unstack('task_type').dropna()
        frame=pd.DataFrame({'delta':np.log1p(g['lines']['feature'])-np.log1p(g['lines']['bug_fix']),'weight':g['n'].sum(axis=1)}).reset_index()
    frame=frame.copy();frame['weighted_delta']=frame.delta*frame.weight
    rep=frame.groupby('repository_id')[['weighted_delta','weight']].sum();x=rep.weighted_delta/rep.weight
    scope_support.append(dict(mode=name,repositories=len(rep),strata=len(frame),prs=int(frame.weight.sum()),median_log1p_ratio=float(np.exp(x.median())),positive_pct=100*(x>0).mean()))
pd.DataFrame(scope_support).to_csv(P/'scope_sensitivity.csv',index=False)

discussion=[];support=[]
for ag,frame in d.groupby('agent'):
    keys=['repository_id','task_type','quarter']
    g=frame.groupby(keys+['has_external_text_30d']).agg(n=('id','size'),merged=('merged','sum')).unstack('has_external_text_30d').dropna()
    delta=100*(g['merged'][True]/g['n'][True]-g['merged'][False]/g['n'][False])
    weight=g['n'].sum(axis=1)
    s=pd.DataFrame({'delta_pp':delta,'weight':weight}).reset_index();s['weighted_delta']=s.delta_pp*s.weight
    repo=s.groupby('repository_id').agg(weight=('weight','sum'),weighted_delta=('weighted_delta','sum'),strata=('weight','size'))
    repo['delta_pp']=repo.weighted_delta/repo.weight
    repo.to_csv(P/f'discussion_{ag}_repositories.csv')
    s.to_csv(P/f'discussion_{ag}_strata.csv',index=False)
    raw=frame.groupby('has_external_text_30d').merged.agg(['size','sum','mean'])
    lo,hi=interval(repo.delta_pp)
    rec=dict(agent=ag,total_n=len(frame),raw_no_discussion_n=int(raw.loc[False,'size']),raw_discussion_n=int(raw.loc[True,'size']),raw_no_discussion_merge_pct=100*raw.loc[False,'mean'],raw_discussion_merge_pct=100*raw.loc[True,'mean'],raw_difference_pp=100*(raw.loc[True,'mean']-raw.loc[False,'mean']),matched_repositories=len(repo),matched_strata=len(g),matched_prs=int(weight.sum()),matched_discussion_n=int(g['n'][True].sum()),matched_no_discussion_n=int(g['n'][False].sum()),equal_repository_difference_pp=repo.delta_pp.mean(),ci_low=lo,ci_high=hi,**sign_summary(repo.delta_pp))
    discussion.append(rec)
    # Fixed design sensitivity: resolved PRs and strata with >= 2 PRs in each group.
    for mode in ['resolved_only','two_per_group']:
        f=frame[frame.state.ne('OPEN')] if mode=='resolved_only' else frame
        q=f.groupby(keys+['has_external_text_30d']).agg(n=('id','size'),merged=('merged','sum')).unstack('has_external_text_30d').dropna()
        if mode=='two_per_group':q=q[q['n'].min(axis=1)>=2]
        diff=100*(q['merged'][True]/q['n'][True]-q['merged'][False]/q['n'][False]);w=q['n'].sum(axis=1)
        x=pd.DataFrame({'weight':w,'weighted_delta':diff*w}).reset_index().groupby('repository_id')[['weight','weighted_delta']].sum()
        support.append(dict(agent=ag,mode=mode,repositories=len(x),strata=len(q),prs=int(w.sum()),difference_pp=(x.weighted_delta/x.weight).mean()))
tests=[scope[1]['p_sign']]+[r['p_sign'] for r in discussion]
adj=multipletests(tests,method='holm')[1]
scope[1]['p_holm']=float(adj[0])
for row,p in zip(discussion,adj[1:]):row['p_holm']=float(p)
rows=[scope[1]]+discussion
order=np.argsort([r['log10_p_sign'] for r in rows]);prev=-np.inf
for rank,i in enumerate(order):
    prev=max(prev,min(0.,rows[i]['log10_p_sign']+np.log10(len(rows)-rank)))
    rows[i]['log10_p_holm']=float(prev)
pd.DataFrame(scope).to_csv(P/'scope_comparisons.csv',index=False)
pd.DataFrame(discussion).to_csv(P/'discussion_comparisons.csv',index=False)
pd.DataFrame(support).to_csv(P/'discussion_sensitivity.csv',index=False)

tails=[];project=[];top_records=[]
for cohort in ['created','merged']:
    f=b[b[cohort+'_in_window']].copy()
    n=int(np.ceil(.01*len(f)));ordered=f.sort_values(['changed_lines','id'],ascending=[False,True]);largest=ordered.iloc[:n]
    cap=float(f.changed_lines.quantile(.99))
    for name,g in [('raw',f),('drop_top_1pct',ordered.iloc[n:]),('winsorize_99pct',f)]:
        lines=g.changed_lines.clip(upper=cap) if name=='winsorize_99pct' else g.changed_lines
        tails.append(dict(cohort=cohort,mode=name,prs=len(g),signal_prs=int(g.agent_detected.sum()),pr_share_pct=100*g.agent_detected.mean(),total_lines=float(lines.sum()),signal_lines=float(lines[g.agent_detected].sum()),line_share_pct=100*lines[g.agent_detected].sum()/lines.sum(),cap=cap,top_n=n,top_line_pct=100*largest.changed_lines.sum()/f.changed_lines.sum(),top_signal_line_pct=100*largest.loc[largest.agent_detected,'changed_lines'].sum()/f.loc[f.agent_detected,'changed_lines'].sum()))
        for repo,idx in g.groupby('repository').groups.items():
            rr=g.loc[idx];ll=lines.loc[idx]
            project.append(dict(cohort=cohort,mode=name,repository=repo,prs=len(rr),signal_prs=int(rr.agent_detected.sum()),total_lines=float(ll.sum()),signal_lines=float(ll[rr.agent_detected].sum()),line_share_pct=100*ll[rr.agent_detected].sum()/ll.sum() if ll.sum() else None))
    top_records.append(largest.assign(cohort=cohort)[['id','repository','url','title','changed_lines','agent_detected','cohort']])
pd.DataFrame(tails).to_csv(P/'tail_sensitivity.csv',index=False)
pd.DataFrame(project).to_csv(P/'tail_project_contributions.csv',index=False)
pd.concat(top_records).to_csv(P/'top_pr_audit.csv',index=False)
result=dict(scope=scope,scope_sensitivity=scope_support,discussion=discussion,discussion_sensitivity=support,tail=tails,bootstrap_repeats=5000,seed=20260928,test_family=6,label_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
(P/'results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
