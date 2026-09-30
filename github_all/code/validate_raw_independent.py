#!/usr/bin/env python3
"""Independently validate full-delivery output against raw JSONL using stdlib."""
import argparse
import gzip
import collections
import csv
import hashlib
import json
import statistics
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    ids=set();total=collections.Counter();qcounts=collections.Counter();files=[];lines=[]
    raw = args.source/'prs.jsonl'
    if not raw.exists(): raw = raw.with_name(raw.name + '.gz')
    with (gzip.open(raw, 'rt', encoding='utf-8') if raw.suffix == '.gz' else raw.open(encoding='utf-8')) as f:
        for line in f:
            r=json.loads(line)
            if r['id'] in ids:raise ValueError('Duplicate raw PR ID')
            ids.add(r['id']);total['rows']+=1
            total['cross_file']+=r['changedFiles']>=2
            total['additions']+=r['additions'];total['deletions']+=r['deletions']
            files.append(r['changedFiles']);lines.append(r['additions']+r['deletions'])
            year,month=r['createdAt'][:4],int(r['createdAt'][5:7])
            qcounts[f'{year}Q{(month+2)//3}']+=1
    with (args.output/'pr_metrics.csv').open() as f:
        metrics=list(csv.DictReader(f))
    output_ids={r['id'] for r in metrics}
    assert len(metrics)==len(output_ids)==len(ids)==126740
    assert output_ids==ids,'Output dropped or added PR IDs'
    s=json.loads((args.output/'summary.json').read_text());whole=s['overall']
    assert s['analyzed_prs']==total['rows']==whole['n']
    assert whole['cross_file_n']==total['cross_file']
    assert whole['additions_sum']==total['additions']
    assert whole['deletions_sum']==total['deletions']
    assert whole['changed_lines_sum']==sum(lines)
    assert whole['changed_files_median']==statistics.median(files)
    assert whole['changed_lines_median']==statistics.median(lines)
    assert abs(whole['cross_file_pct']-100*total['cross_file']/total['rows'])<1e-10
    assert s['quarter_counts']==dict(qcounts)
    assert sum(s['agent_counts'].values())==len(ids)
    assert s['sampling_performed'] is False and s['bootstrap_performed'] is False
    with (args.output/'source_manifest.csv').open() as f:
        for r in csv.DictReader(f):
            path=args.source/r['file'];h=hashlib.sha256()
            with path.open('rb') as src:
                for b in iter(lambda:src.read(4*1024*1024),b''):h.update(b)
            assert h.hexdigest()==r['sha256'] and path.stat().st_size==int(r['bytes'])
    result={'status':'passed','method':'independent stdlib pass through every raw PR',
            'exact_raw_output_id_set_match':True,'raw_rows':total['rows'],
            'independently_checked_cross_file_count':total['cross_file'],
            'independently_checked_file_median':statistics.median(files),
            'independently_checked_changed_line_median':statistics.median(lines),
            'line_totals_and_quarter_counts_match':True,'source_hashes_match':True}
    (args.output/'independent_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
