from context import ARTIFACT, OUTPUT, stage, raw_open
from pathlib import Path
import json, re, sys, collections
import pandas as pd
OUT=stage('labels')
SRC=ARTIFACT/'github_all/raw'
sys.path.insert(0,str(ARTIFACT/'top10/code'))
from collect_repo_contributions import agents
pattern=re.compile(r'Generated\s+with\s+\[Claude\s+Code\]\(https?://[^\s)]+\)',re.I)
labels={}
for line in raw_open(SRC/'pr_labels_v2.jsonl', 'rt'):
 r=json.loads(line);assert r['id'] not in labels;labels[r['id']]=r
 assert r['status']=='ok' and type(r['human_participation']) is bool
 assert re.fullmatch('[0-9a-f]{64}',r['input_hash'])
 assert len(r['phases'])==len(set(r['phases']))
cohorts=collections.defaultdict(dict)
for line in raw_open(SRC/'selections.jsonl', 'rt'):
 r=json.loads(line);assert r['node_id'] not in cohorts[r['cohort']];cohorts[r['cohort']][r['node_id']]=r
assert cohorts['unified-1pct']==cohorts['unified-2025-2026'] or set(cohorts['unified-1pct'])==set(cohorts['unified-2025-2026'])
primary=cohorts['unified-1pct'];seen=set();rows=[];mismatch=[];unlabeled=[]
for line in raw_open(SRC/'prs.jsonl', 'rt'):
 r=json.loads(line);pid=r['id'];assert pid not in seen;seen.add(pid)
 a=primary[pid]['agent'];l=labels.get(pid)
 if l:
  for k, v in [('agent',a),('created_at',r['createdAt']),('url',r['url']),('number',r['number'])]:
   if l[k]!=v:mismatch.append({'id':pid,'field':k,'label':l[k],'raw':v})
 else:unlabeled.append({'id':pid,'agent':a,'created_at':r['createdAt'],'url':r['url']})
 sig=agents(r)
 if pattern.search(r.get('body') or ''):sig['claude_code']='markdown_claude_body_signature'
 rows.append(dict(id=pid,local_signal_v2=bool(sig),assigned_agent_in_local_signals=a in sig,local_agents='|'.join(sorted(sig)),changed_files=r['changedFiles'],changed_lines=r['additions']+r['deletions']))
assert not mismatch, mismatch[:5]
assert set(labels)<=seen
pd.DataFrame(rows).to_csv(OUT/'raw_metrics_signal_audit.csv',index=False)
pd.DataFrame(unlabeled).to_csv(OUT/'unlabeled_prs.csv',index=False)
missing=[dict(id=k,**v) for k,v in primary.items() if k not in seen]
pd.DataFrame(missing).to_csv(OUT/'selected_without_details.csv',index=False)
s=dict(status='passed', raw_prs=len(seen), label_prs=len(labels), missing_labels=len(unlabeled), label_metadata_mismatches=len(mismatch), labels_without_raw=len(set(labels)-seen), selected=len(primary), selected_missing_detail=len(missing), label_models=dict(collections.Counter(x['model'] for x in labels.values())))
(OUT/'audit.json').write_text(json.dumps(s,indent=2))
print(json.dumps(s,indent=2))
