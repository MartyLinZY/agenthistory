from context import ARTIFACT, OUTPUT, stage, raw_open
"""Create a file-level table for complete PRs; raw export remains read-only."""
from pathlib import Path
from collections import Counter
import sys,json,hashlib
import pandas as pd
P=stage('modules')
sys.path.insert(0,str(ARTIFACT/'github_all/code'))
from analyze_github_delivery import file_kind
source=ARTIFACT/'github_all/raw/prs.jsonl.gz'
rows=[]; eligible=[]; sha=hashlib.sha256()
with raw_open(source) as f:
    for line in f:
        sha.update(line);r=json.loads(line);fs=r['files']
        if r['collection']['truncated'].get('files',False) or len(fs)!=r['changedFiles'] or len({x['path'] for x in fs})!=len(fs):continue
        assert sum(x['additions']+x['deletions'] for x in fs)==r['additions']+r['deletions']
        eligible.append(r['id'])
        rows.extend((r['id'],x['path'],file_kind(x['path']),x['additions'],x['deletions']) for x in fs)
a=pd.DataFrame(rows,columns=['id','path','file_kind','additions','deletions'])
a.to_parquet(P/'files.parquet',index=False)
metrics=pd.read_parquet(OUTPUT/'interaction/enriched_metrics.parquet')
pr=metrics[metrics.id.isin(eligible)].copy();assert len(pr)==125174 and pr.files_complete.all()
pr.to_parquet(P/'eligible_prs.parquet',index=False)
role=a.groupby('file_kind').agg(prs=('id','nunique'),files=('path','size'));role.to_csv(P/'role_check.csv')
old=pd.read_csv(OUTPUT/'corpus/file_category_summary.csv').set_index('category')
assert (role.prs==old.loc[role.index,'prs_with_category']).all() and (role.files==old.loc[role.index,'files']).all()
dirs=Counter(); names=Counter()
for path in a.loc[a.file_kind.eq('source_candidate'),'path']:
    parts=path.lower().split('/');dirs.update(set(parts[:-1]));names.update([parts[-1]])
pd.DataFrame(dirs.most_common(),columns=['directory_token','changed_file_occurrences']).to_csv(P/'directory_inventory.csv',index=False)
pd.DataFrame(names.most_common(1500),columns=['file_name','changed_file_occurrences']).to_csv(P/'filename_inventory.csv',index=False)
info=dict(source=str(source.relative_to(ARTIFACT)),source_sha256=sha.hexdigest(),eligible_prs=len(pr),file_rows=len(a),role_totals_reconciled=True,zero_file_prs=int(pr.changed_files.eq(0).sum()))
(P/'extraction_audit.json').write_text(json.dumps(info,indent=2));print(info)
print(pd.DataFrame(dirs.most_common(130)).to_string(index=False,header=False))
