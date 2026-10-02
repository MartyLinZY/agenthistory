from context import ARTIFACT, OUTPUT, stage, raw_open
"""Read all delivered PRs once; recover timing and path evidence without network calls."""
from pathlib import Path
from datetime import datetime
from collections import Counter
import sys, json, hashlib
import pandas as pd
P=stage('interaction')
sys.path.insert(0,str(ARTIFACT/'github_all/code'))
from analyze_github_delivery import file_kind
SOURCE=ARTIFACT/'github_all/raw/prs.jsonl.gz'
def stamp(v):return datetime.fromisoformat(v.replace('Z','+00:00')) if v else None
def nonbot(a):
    a=a or {};login=(a.get('login') or '').lower()
    return bool(login) and a.get('__typename')!='Bot' and not login.endswith('[bot]')
rows=[];counts=Counter();sha=hashlib.sha256()
with raw_open(SOURCE) as f:
    for line in f:
        sha.update(line);r=json.loads(line);counts['raw_prs']+=1
        created=stamp(r['createdAt']);merged=stamp(r.get('mergedAt'));author=(r.get('author') or {}).get('login','').lower();merger=(r.get('mergedBy') or {}).get('login','').lower()
        flags=r['collection']['truncated'];files=r['files'];complete=(not flags.get('files',False) and len(files)==r['changedFiles'] and len({x['path'] for x in files})==len(files))
        categories=Counter(file_kind(x['path']) for x in files)
        checks=r.get('checks') or {};check_contexts=checks.get('contexts',[]) if isinstance(checks,dict) else checks
        source_line=sum(x['additions']+x['deletions'] for x in files if file_kind(x['path'])=='source_candidate')
        test_line=sum(x['additions']+x['deletions'] for x in files if file_kind(x['path'])=='test_path')
        rec=dict(id=r['id'],merged_at=r.get('mergedAt'),closed_at=r.get('closedAt'),author_known=bool(author),merger_known=bool(merger),merger_nonbot=nonbot(r.get('mergedBy')),same_author_merger=bool(author and merger and author==merger),merge_minutes=(merged-created).total_seconds()/60 if merged else None,files_complete_check=complete,touches_source=bool(categories['source_candidate']) if complete else None,touches_test=bool(categories['test_path']) if complete else None,test_files=categories['test_path'] if complete else None,source_lines_check=source_line if complete else None,test_lines=test_line if complete else None,check_records=len(r.get('checks') or []),patch_fields=sum('patch' in x for x in files),commit_records=len(r.get('commits') or []),ext_text_30d_legacy=0,ext_text_premerge_30d=0,ext_text_postmerge_30d=0,ext_review_premerge=0,ext_review_premerge_30d=0,ext_approval_premerge=0,ext_approval_empty_premerge=0,ext_review_all=0,ext_text_premerge_all=0)
        seen=set()
        events=[('review',e) for e in r.get('reviews',[])]+[('comment',e) for e in r.get('comments',[])]+[('inline',e) for t in r.get('reviewThreads',[]) for e in t.get('comments',[])]
        for i,(kind,e) in enumerate(events):
            key=e.get('id') or e.get('url') or f'{kind}:missing:{i}'
            if key in seen:continue
            seen.add(key);actor=e.get('author') or {}
            if not nonbot(actor) or (actor.get('login') or '').lower()==author:continue
            counts['external_records']+=1
            text=bool(str(e.get('body') or '').strip());legacy=stamp(e.get('createdAt') or e.get('submittedAt'))
            if text and legacy and 0<=(legacy-created).total_seconds()<30*86400:rec['ext_text_30d_legacy']+=1
            when=stamp((e.get('submittedAt') or e.get('createdAt')) if kind=='review' else e.get('createdAt'))
            if kind=='review' and e.get('submittedAt') and e.get('createdAt')!=e['submittedAt']:counts['review_created_submitted_differ']+=1
            if not when:counts['external_record_without_timestamp']+=1;continue
            elapsed=(when-created).total_seconds();in30=0<=elapsed<30*86400
            pre=bool(merged and created<=when<merged);post=bool(merged and when>=merged)
            if text:
                rec['ext_text_premerge_all']+=int(pre)
                rec['ext_text_premerge_30d']+=int(pre and in30)
                rec['ext_text_postmerge_30d']+=int(post and in30)
            if kind=='review' and e.get('state')!='PENDING':
                rec['ext_review_all']+=1;rec['ext_review_premerge']+=int(pre);rec['ext_review_premerge_30d']+=int(pre and in30)
                rec['ext_approval_premerge']+=int(pre and e.get('state')=='APPROVED')
                rec['ext_approval_empty_premerge']+=int(pre and e.get('state')=='APPROVED' and not text)
        rec['premerge_external_evidence']=bool(rec['ext_text_premerge_all'] or rec['ext_review_premerge'])
        # A nonempty checks wrapper can contain no actual check contexts.
        rec['check_payload_present']=bool(checks)
        rec['check_records']=len(check_contexts)
        counts['merged_state_without_time']+=r['state']=='MERGED' and not merged
        counts['merged_time_without_state']+=bool(merged) and r['state']!='MERGED'
        counts['negative_merge_duration']+=bool(merged and merged<created)
        counts['merged_without_merger']+=bool(merged) and not merger
        counts['files_incomplete']+=not complete
        rows.append(rec)
        if len(rows)%30000==0:print(f'Extracted {len(rows):,} PRs',flush=True)
a=pd.DataFrame(rows);assert a.id.is_unique and len(a)==126740
base=pd.read_csv(OUTPUT/'corpus/pr_metrics.csv')
c=base.merge(a,on='id',validate='one_to_one')
assert (c.files_complete==c.files_complete_check).all()
assert (c.external_text_30d_count==c.ext_text_30d_legacy).all()
assert c.loc[c.files_complete,'source_lines'].equals(c.loc[c.files_complete,'source_lines_check'].astype(float))
labels=pd.read_parquet(OUTPUT/'labels/labeled_metrics.parquet')[['id','task_type','phases']]
c=c.merge(labels,on='id',how='left',validate='one_to_one')
c.to_parquet(P/'enriched_metrics.parquet',index=False)
counts['known_authors']=int(c.author_known.sum());counts['known_merge_actors']=int(c.loc[c.merged,'merger_known'].sum())
counts['prs_with_checks']=int((c.check_records>0).sum());counts['files_with_patch_field']=int(c.patch_fields.sum())
counts['prs_with_check_container']=int(c.check_payload_present.sum())
manifest=dict(counts=counts,source=str(SOURCE.relative_to(ARTIFACT)),source_sha256=sha.hexdigest(),rows=len(c),path_classifier='github_all/code/analyze_github_delivery.py:file_kind',event_time='submittedAt for reviews when present, otherwise createdAt; exact merge-time events belong to postmerge',legacy_event_counts_reconciled=True)
(P/'field_audit.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
