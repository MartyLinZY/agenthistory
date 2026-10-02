from context import ARTIFACT, OUTPUT, stage, raw_open
from pathlib import Path
import pandas as pd, numpy as np, json, collections
OUT=stage('labels')
M=stage('corpus');SRC=ARTIFACT/'github_all/raw'
a=pd.read_csv(M/'pr_metrics.csv');labels=pd.read_json(SRC/'pr_labels_v2.jsonl.gz',lines=True)
assert a.id.is_unique and labels.id.is_unique
l=labels[['id','task_type','phases','human_participation','model','input_hash']]
d=a.merge(l,on='id',how='left',validate='one_to_one');v=d[d.task_type.notna()].copy()
v['human_label']=v.human_participation.astype(bool)
v['phase_count']=v.phases.map(len)
def stats(g):
 return dict(n=len(g),human_n=int(g.human_label.sum()),human_pct=100*g.human_label.mean(),text_n=int(g.has_external_text_30d.sum()),text_pct=100*g.has_external_text_30d.mean(),files_median=g.changed_files.median(),lines_median=g.changed_lines.median(),lines_q25=g.changed_lines.quantile(.25),lines_q75=g.changed_lines.quantile(.75),cross_n=int(g.cross_file.sum()),cross_pct=100*g.cross_file.mean(),merged_n=int(g.merged.sum()),merged_pct=100*g.merged.mean())
def group(keys,frame=v,name=None):
 rows=[]
 for k,g in frame.groupby(keys):
  k=k if isinstance(k,tuple) else (k,); rows.append(dict(zip(keys,k),**stats(g)))
 r=pd.DataFrame(rows);r.to_csv(OUT/((name or '_'.join(keys))+'.csv'),index=False);return r
for keys in [['agent'],['quarter'],['task_type'],['task_type','agent'],['task_type','quarter'],['agent','quarter'],['task_type','agent','quarter']]:group(keys)
phases=['requirement_elicitation','requirement_analysis','design','implementation','integration','testing','deployment']
phase_rows=[]
for ph in phases:
 g=v[v.phases.map(lambda x:ph in x)]
 phase_rows.append(dict(phase=ph,coverage_pct=100*len(g)/len(v),**stats(g)))
 for key in ['agent','quarter']:
  z=group([key],g,name=f'phase_{ph}_{key}')
pd.DataFrame(phase_rows).to_csv(OUT/'phase_summary.csv',index=False)
phase_quarters=[]
for q,g in v.groupby('quarter'):
 for phase in phases:
  h=g[g.phases.map(lambda values:phase in values)]
  phase_quarters.append(dict(quarter=q,phase=phase,labeled_n=len(g),phase_n=len(h),phase_pct=100*len(h)/len(g),flag_n=int(h.human_label.sum()),flag_pct=100*h.human_label.mean(),text_n=int(h.has_external_text_30d.sum()),text_pct=100*h.has_external_text_30d.mean()))
pd.DataFrame(phase_quarters).to_csv(OUT/'phase_quarter.csv',index=False)

pd.crosstab(v.human_label,v.has_exported_text).to_csv(OUT/'human_exported_crosstab.csv')
assert (v.human_label==v.has_exported_text).all()
pd.crosstab(v.human_label,v.has_external_text_30d).to_csv(OUT/'human_text_crosstab.csv')
pd.crosstab(v.task_type,v.fix_title_candidate).to_csv(OUT/'task_title_crosstab.csv')
# Same definition and constant weights across recent quarters. No causal interpretation.
fixed=[];support=[]
for subset,frame in [('all',v),('bug_fix',v[v.task_type=='bug_fix'])]:
 for dims in [['agent'],['agent','task_type']] if subset=='all' else [['agent'],['repository_id']]:
  frame=frame[frame.quarter.isin(['2026Q1','2026Q2'])]
  t=frame.groupby(dims+['quarter']).human_label.agg(['size','sum']).unstack('quarter').dropna()
  te=frame.groupby(dims+['quarter']).has_external_text_30d.sum().unstack('quarter').loc[t.index]
  w=t['size'].sum(axis=1);w=w/w.sum()
  for q in ['2026Q1','2026Q2']:
   fixed.append(dict(subset=subset,weighting='+'.join(dims),quarter=q,strata=len(t),eligible_prs=int(t['size'][q].sum()),human_pct=100*((t['sum'][q]/t['size'][q])*w).sum(),text_pct=100*((te[q]/t['size'][q])*w).sum()))
pd.DataFrame(fixed).to_csv(OUT/'human_fixed_weights.csv',index=False)
# Missing retrieval analysis across selected PRs, without substituting retry-event counts.
selected=pd.read_json(OUT/'selected_without_details.csv') if False else pd.read_csv(OUT/'selected_without_details.csv')
selected['quarter']=pd.to_datetime(selected.created_at).dt.tz_localize(None).dt.to_period('Q').astype(str)
allsel=pd.concat([a[['id','agent','quarter']].assign(retrieved=True),selected[['id','agent','quarter']].assign(retrieved=False)],ignore_index=True)
for key in ['agent','quarter']:
 t=allsel.groupby(key).retrieved.agg(selected='size',retrieved='sum');t['missing']=t.selected-t.retrieved;t['retrieved_pct']=100*t.retrieved/t.selected;t.to_csv(OUT/f'retrieval_{key}.csv')
missing=d[d.task_type.isna()]; missing[['id','agent','quarter','url']].to_csv(OUT/'missing_label_distribution.csv',index=False)
raw=pd.read_csv(OUT/'raw_metrics_signal_audit.csv');assert set(raw.id)==set(a.id)
r=a.merge(raw,on='id',suffixes=('','_check'));assert (r.changed_files==r.changed_files_check).all() and (r.changed_lines==r.changed_lines_check).all()
b=pd.read_parquet(OUTPUT/'projects/prs_signal_v2.parquet')
o=a.merge(b,on='id',suffixes=('_a','_b'));o.to_csv(OUT/'cohort_overlap.csv',index=False)
support=a.groupby('repository_id').quarter.nunique()
summary=dict(n=len(a),labeled=len(v),human_exported_disagreements=int((v.human_label!=v.has_exported_text).sum()),label_coverage_pct=100*len(v)/len(a),overall=stats(v),phase_count=v.phase_count.value_counts().sort_index().to_dict(),human_label_bounds_pct=[100*v.human_label.sum()/len(a),100*(v.human_label.sum()+len(missing))/len(a)],retrieval_pct=100*len(a)/len(allsel),missing_labels_by_agent=missing.agent.value_counts().to_dict(),overlap=dict(n=len(o),repositories=o.repository_a.nunique(),b_detected=int(o.agent_detected.sum()),scope_changed=int(((o.changed_files_a!=o.changed_files_b)|(o.changed_lines_a!=o.changed_lines_b)).sum())),repository_one_quarter=int((support==1).sum()),repository_all_quarters=int((support==6).sum()),repository_q1q2=len(set(a[a.quarter=='2026Q1'].repository_id)&set(a[a.quarter=='2026Q2'].repository_id)),top1pct_n=int(np.ceil(len(a)*.01)),top1pct_line_pct=100*a.changed_lines.nlargest(int(np.ceil(len(a)*.01))).sum()/a.changed_lines.sum(),lowstar_pct=100*(a.stars<=10).mean(),nonbot_any_n=int((a.nonbot_text_count>0).sum()),external_events_30d=int(a.external_text_30d_count.sum()),interaction_incomplete=int((~a.interaction_complete).sum()),text30_complete_pct=100*a.loc[a.interaction_complete,'has_external_text_30d'].mean(),local_signal_cross_pct=100*r.loc[r.local_signal_v2,'cross_file'].mean(),no_multi_cross_pct=100*a.loc[~a.multi_agent_signal,'cross_file'].mean(),complete_files_median=a.loc[a.files_complete,'changed_files'].median(),complete_lines_median=a.loc[a.files_complete,'changed_lines'].median(),incomplete_files_median=a.loc[~a.files_complete,'changed_files'].median(),incomplete_lines_median=a.loc[~a.files_complete,'changed_lines'].median())
(OUT/'label_results.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item()))
v.drop(columns='phases').assign(phases=v.phases.map(lambda x:'|'.join(x))).to_parquet(OUT/'labeled_metrics.parquet',index=False)
print(json.dumps(summary,indent=2,default=lambda x:x.item()))
