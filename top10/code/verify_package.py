#!/usr/bin/env python3
"""Verify raw hashes, cohort, counts, IDs, dates and file foreign keys."""
from pathlib import Path
import argparse, collections, gzip, hashlib, json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def require(ok,message):
 if not ok:raise RuntimeError(message)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 return h.hexdigest()
def main():
 repos=pd.read_csv(ROOT/'metadata/repositories.csv');allowed=set(repos.repository)
 require(len(allowed)==10,'Expected ten repositories')
 prs=pd.read_parquet(ROOT/'features/prs.parquet');files=pd.read_parquet(ROOT/'features/files.parquet')
 require(len(prs)==190094 and prs.id.is_unique,'PR count / uniqueness')
 require(len(files)==1306724,'File count')
 require(set(prs.repository)<=allowed and set(files.repository)<=allowed,'Unexpected project')
 require(files.pr_id.isin(prs.id).all(),'Orphan file foreign key')
 owners=prs.set_index('id').repository
 require(files.repository.eq(files.pr_id.map(owners)).all(),'File repository mismatch')
 counts=files.groupby('pr_id').size()
 require(prs.id.map(counts).fillna(0).eq(prs.files_listed).all(),'Per-PR listed-file count mismatch')
 require((files[['additions','deletions']]>=0).all().all(),'Negative file line counts')
 require((prs[['additions','deletions','changed_files']]>=0).all().all(),'Negative PR counts')
 lo='2025-01-01T00:00:00Z';hi='2026-07-01T00:00:00Z'
 created=(prs.created_at>=lo)&(prs.created_at<hi)
 merged=prs.merged_at.notna()&(prs.merged_at>=lo)&(prs.merged_at<hi)
 require(created.eq(prs.created_in_window).all() and merged.eq(prs.merged_in_window).all(),'Cohort flags')
 require((created|((prs.created_at<lo)&merged)).all(),'Out-of-window PR')
 seen=set();raw_files=0
 index=ROOT/'metadata/raw_archives.json'
 if index.exists():
  archives=json.loads(index.read_text())
 else:
  archives=[{'repository':r.repository,'compressed_path':r.raw_relative_path,'uncompressed_sha256':r.raw_sha256} for r in repos.itertuples()]
 require({r['repository'] for r in archives}==allowed,'Archive cohort mismatch')
 for entry in archives:
  h=hashlib.sha256();n=0
  path=ROOT/entry['compressed_path']
  with (gzip.open(path,'rb') if path.suffix=='.gz' else path.open('rb')) as stream:
   for line in stream:
    h.update(line);p=json.loads(line);n+=1
    require(p['repository']['nameWithOwner']==entry['repository'],'Raw project mismatch')
    require(p['id'] not in seen,'Duplicate raw ID');seen.add(p['id'])
    raw_files+=len((p.get('files') or {}).get('nodes') or [])
  require(h.hexdigest()==entry['uncompressed_sha256'],'Uncompressed checksum')
  expected=int(repos.loc[repos.repository==entry['repository'],'included_prs'].iloc[0])
  require(n==expected,'Raw repository count')
 require(seen==set(prs.id),'Raw / feature ID mismatch')
 require(raw_files==len(files),'Raw / feature file-count mismatch')
 excluded=json.loads((ROOT/'metadata/exclusions.json').read_text())
 require(len(excluded)==2 and not ({r['id'] for r in excluded}&seen),'Exclusion ledger')
 require(int(repos.expected_prs.sum())==190096 and int(repos.excluded_prs.sum())==2,'Expected population')
 csv=pd.read_csv(ROOT/'features/prs.csv.gz',dtype=str,keep_default_na=False)
 require(set(csv.id)==seen and len(csv)==len(prs),'CSV PR identity mismatch')
 for col in prs.select_dtypes(include=['object','string']).columns:
  require(csv[col].tolist()==prs[col].fillna('').astype(str).tolist(),f'CSV string mismatch: {col}')
 print(json.dumps({'passed':True,'repositories':10,'prs':len(prs),'file_rows':len(files),'raw_sha256_verified':10,'csv_ids_verified':True}))
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--dataset',type=Path,required=True,help='Reconstructed dataset containing features and metadata')
 ROOT=parser.parse_args().dataset
 main()
