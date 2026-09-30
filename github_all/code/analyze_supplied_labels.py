#!/usr/bin/env python3
"""Summarize provided annotations without calling a model or inferring labels."""
import argparse, collections, gzip, json
from pathlib import Path
import pandas as pd

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--metrics',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 rows=[]
 path=args.source/'pr_labels_v2.jsonl.gz'
 with gzip.open(path,'rt',encoding='utf-8') as f:
  for line in f:rows.append(json.loads(line))
 labels=pd.DataFrame(rows);metrics=pd.read_csv(args.metrics)
 if not labels.id.is_unique or not metrics.id.is_unique:raise ValueError('Duplicate IDs')
 if not set(labels.id)<=set(metrics.id):raise ValueError('Labels without raw metric records')
 joined=metrics.merge(labels,on='id',how='left',validate='one_to_one',suffixes=('','_label'))
 valid=joined[joined.status=='ok'].copy()
 for field in ['agent','url','number']:
  if field+'_label' in valid and not valid[field].eq(valid[field+'_label']).all():raise ValueError('Label metadata mismatch: '+field)
 args.output.mkdir(parents=True,exist_ok=True)
 tables=[]
 for fields in [['task_type'],['agent'],['quarter'],['task_type','agent'],['task_type','quarter']]:
  result=valid.groupby(fields,dropna=False).agg(labeled_prs=('id','size'),human_participation_positive=('human_participation','sum'),external_text_30d=('has_external_text_30d','sum')).reset_index()
  result.to_csv(args.output/('_'.join(fields)+'.csv'),index=False);tables.append('_'.join(fields))
 phase_counts=collections.Counter(ph for phases in labels.phases for ph in phases)
 pd.DataFrame([{'phase':k,'labeled_prs':v,'denominator':len(labels)} for k,v in sorted(phase_counts.items())]).to_csv(args.output/'phase_counts.csv',index=False)
 joined[joined.status.isna()][['id','agent','quarter','url']].to_csv(args.output/'missing_labels.csv',index=False)
 summary={'pr_rows':len(metrics),'label_rows':len(labels),'missing_labels':len(metrics)-len(labels),'models':labels.model.value_counts().to_dict(),'status_counts':labels.status.value_counts().to_dict(),'human_participation_positive':int(labels.human_participation.sum()),'task_counts':labels.task_type.value_counts().to_dict(),'phase_counts':dict(phase_counts),'human_flag_vs_exported_text_disagreements':int((valid.human_participation!=valid.has_exported_text).sum()),'interpretation':'Supplied model outputs, not manually validated phase participation or ground truth. Phase categories overlap; no inference for unlabeled PRs.'}
 (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary))
if __name__=='__main__':main()
