from context import ARTIFACT, OUTPUT, stage, raw_open
"""Independent raw-event reconstruction of the new workflow conclusions."""
from pathlib import Path
from datetime import datetime
import json,statistics,re,hashlib
import pandas as pd,numpy as np
P=stage('interaction')
def date(x):return datetime.fromisoformat(x.replace('Z','+00:00')) if x else None
same=[];other=[];same_pre=other_pre=0;textprs=reviewprs=union=0;merged=known=0;fast=0
with raw_open(ARTIFACT/'github_all/raw/prs.jsonl.gz', 'rt') as f:
    for line in f:
        r=json.loads(line)
        if r['state']!='MERGED':continue
        merged+=1;start=date(r['createdAt']);end=date(r['mergedAt']);minutes=(end-start).total_seconds()/60;fast+=minutes<=5
        aa=(r.get('author') or {}).get('login');bb=(r.get('mergedBy') or {}).get('login');a=(aa or '').lower();b=(bb or '').lower()
        events={}
        for typ,ls in [('review',r['reviews']),('comment',r['comments']),('inline',[c for th in r['reviewThreads'] for c in th['comments']])]:
            for i,e in enumerate(ls):events.setdefault(e.get('id') or e.get('url') or f'{typ}-{i}',(typ,e))
        has_text=has_review=False
        for typ,e in events.values():
            actor=e.get('author') or {};login=(actor.get('login') or '').lower()
            if not login or login==a or actor.get('__typename')=='Bot' or login.endswith('[bot]'):continue
            t=date((e.get('submittedAt') or e.get('createdAt')) if typ=='review' else e.get('createdAt'))
            if t is None or not start<=t<end:continue
            has_text|=bool(str(e.get('body') or '').strip())
            has_review|=typ=='review' and e.get('state')!='PENDING'
        evidence=has_text or has_review;textprs+=has_text;reviewprs+=has_review;union+=evidence
        if a and b:
            known+=1
            if a==b:same.append(minutes);same_pre+=evidence
            else:other.append(minutes);other_pre+=evidence
assert (merged,known,len(same),len(other))==(108549,108473,88470,20003)
assert (textprs,reviewprs,union,fast)==(5818,7277,9332,56696)
assert (same_pre,other_pre)==(1822,7504)
assert round(statistics.median(same),2)==2.48 and round(statistics.median(other),2)==24.13
a=pd.read_parquet(P/'enriched_metrics.parquet');saved=pd.read_csv(OUTPUT/'corpus/file_category_summary.csv').set_index('category')
assert int(a.loc[a.files_complete,'touches_test'].sum())==int(saved.loc['test_path','prs_with_category'])
source=a[a.files_complete&a.task_type.notna()&a.touches_source.eq(True)]
task=pd.read_csv(P/'task_test_cochange.csv').set_index('task')
for name,g in source.groupby('task_type'):
    assert len(g)==task.loc[name,'n'] and int(g.touches_test.sum())==task.loc[name,'test_cochange_n']
result=dict(status='passed',merged_prs=merged,known_actor_prs=known,same_actor_prs=len(same),other_actor_prs=len(other),premerge_external_evidence=union,raw_workflow_reconstructed=True,path_totals_reconciled=True)
(P/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
