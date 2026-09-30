#!/usr/bin/env python3
"""Restore losslessly archived JSONL; refuse conflicting existing files."""
from pathlib import Path
import gzip, hashlib, json, os
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 return h.hexdigest()
def main():
 for entry in json.loads((ROOT/'raw/metadata/raw_archives.json').read_text()):
  source=ROOT/entry['compressed_path'];target=ROOT/entry['restored_path']
  if target.exists():
   if sha(target)!=entry['uncompressed_sha256']:raise RuntimeError(f'Conflicting existing file: {target.relative_to(ROOT)}')
   continue
  temp=target.with_suffix('.jsonl.tmp');h=hashlib.sha256()
  try:
   with gzip.open(source,'rb') as inp,temp.open('xb') as out:
    for block in iter(lambda:inp.read(4*1024**2),b''):out.write(block);h.update(block)
   if h.hexdigest()!=entry['uncompressed_sha256']:raise RuntimeError('Archive checksum mismatch')
   # Atomic no-clobber publication: never overwrite a concurrently created file.
   os.link(temp,target)
  finally:
   if temp.exists():temp.unlink()
  print(entry['repository'])
 print('All ten raw JSONL files restored and verified.')
if __name__=='__main__':main()
