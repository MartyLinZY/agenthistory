#!/usr/bin/env python3
"""Check both raw snapshots using only the standard library."""
from pathlib import Path
import gzip,hashlib,json
ROOT=Path(__file__).resolve().parent

def sha_file(path,compressed=False):
 h=hashlib.sha256()
 opener=gzip.open if compressed else open
 with opener(path,'rb') as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 return h.hexdigest()
def main():
 for entry in json.loads((ROOT/'github_all/raw/metadata/raw_archives.json').read_text()):
  path=ROOT/'github_all'/entry['stored_path']
  if sha_file(path,path.suffix=='.gz')!=entry['uncompressed_sha256']:raise RuntimeError('GitHub raw checksum: '+entry['file'])
 for entry in json.loads((ROOT/'top10/raw/metadata/raw_archives.json').read_text()):
  path=ROOT/'top10'/entry['compressed_path']
  if sha_file(path,True)!=entry['uncompressed_sha256']:raise RuntimeError('Top10 raw checksum: '+entry['repository'])
 print(json.dumps({'passed':True,'github_source_files':6,'top10_raw_archives':10}))
if __name__=='__main__':main()
