#!/usr/bin/env python3
"""Analyze EVERY delivered PR; never use a selection manifest to filter rows.

All checks, measurements, plots, and reports are generated programmatically.
No sampling, resampling, network requests, model calls, or manuscript edits.
"""
import argparse
import gzip
import collections
import hashlib
import json
import platform
import re
import shutil
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_github_delivery import file_kind, quarter, AGENTS, NAMES, COLORS, QUARTERS


def jsonl(path):
    if not path.exists():
        path = path.with_name(path.name + '.gz')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                raise ValueError(f'{path.name}:{line_number}: empty record')
            try:
                value = json.loads(line)
            except Exception as error:
                raise ValueError(f'{path.name}:{line_number}: invalid JSON') from error
            if not isinstance(value, dict):
                raise ValueError(f'{path.name}:{line_number}: expected object')
            yield value


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')


def nonbot_candidate(actor):
    actor = actor or {}
    login = (actor.get('login') or '').lower()
    return bool(login) and actor.get('__typename') != 'Bot' and not login.endswith('[bot]')


def summarize(g):
    result = {'n': len(g), 'repositories': g.repository_id.nunique()}
    for col in ['changed_files', 'additions', 'deletions', 'changed_lines',
                'directories', 'source_files', 'source_lines']:
        v = g[col].dropna()
        result[col + '_eligible'] = len(v)
        for key, val in [('median', v.median()), ('q25', v.quantile(.25)),
                         ('q75', v.quantile(.75)), ('p90', v.quantile(.9)),
                         ('mean', v.mean()), ('std', v.std()), ('sum', v.sum())]:
            result[col + '_' + key] = val
    for col in ['cross_file', 'source_cross_file', 'has_exported_text', 'has_external_text_all',
                'has_external_text_30d', 'has_review', 'has_comment', 'files_complete',
                'interaction_complete', 'multi_agent_signal', 'fix_title_candidate',
                'bug_label', 'merged']:
        v = g[col].dropna()
        result[col + '_n'] = int(v.sum())
        result[col + '_eligible'] = len(v)
        result[col + '_pct'] = 100 * v.mean()
    return result


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])


def number(v):
    return f'{v:,.0f}' if float(v).is_integer() else f'{v:,.2f}'.rstrip('0').rstrip('.')


def build_reports(out, df, summary, groups, quarters, temporal, candidate_quarters):
    # Manuscript and interpretation are generated after joining supplied model labels.
    (out/'README.md').write_text('All delivered PRs are analyzed once. See ../README.md for supplied-label summaries and reproducibility boundaries. No additional sampling or model calls.\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-prs', type=int)
    args = parser.parse_args()
    src, out = args.source, args.output
    out.mkdir(parents=True, exist_ok=True)
    (out/'figures').mkdir(exist_ok=True)
    input_files = sorted(p for p in src.iterdir() if p.suffix in {'.json', '.jsonl', '.gz'})
    input_state = {p.name: (p.stat().st_size, p.stat().st_mtime_ns, sha256(p)) for p in input_files}
    metadata = json.loads((src/'run_metadata.json').read_text())
    # Manifest is a LABEL LOOKUP ONLY. It never decides whether a PR is included.
    assignment = {}
    for r in jsonl(src/'selections.jsonl'):
        if r['cohort'] == 'unified-2025-2026':
            if r['node_id'] in assignment:
                raise ValueError('Duplicate primary-manifest label')
            assignment[r['node_id']] = r
    counts = collections.Counter()
    truncation = collections.Counter()
    signal_counts = collections.Counter()
    file_summary = collections.defaultdict(collections.Counter)
    records = []
    candidate_records = []
    bad = []
    seen = set()
    for r in jsonl(src/'prs.jsonl'):
        counts['input_rows'] += 1
        pid = r.get('id')
        if not pid or pid in seen:
            raise ValueError('Missing or duplicate PR ID: reconcile before analysis')
        seen.add(pid)
        for key in ['changedFiles', 'additions', 'deletions']:
            if not isinstance(r.get(key), int) or r[key] < 0:
                raise ValueError(f'Invalid numeric field {key} in PR {pid}')
        signals = sorted(set(x['agent'] for x in r.get('searchMatches', [])))
        signal_counts.update(signals)
        label = assignment.get(pid)
        agent = label['agent'] if label else (signals[0] if len(signals) == 1 else 'multiple_or_unknown')
        counts['without_manifest_label'] += label is None
        counts['manifest_agent_not_in_signals'] += bool(label and agent not in signals)
        if label and pd.Timestamp(label['created_at']) != pd.Timestamp(r['createdAt']):
            raise ValueError('Creation-time mismatch between manifest and detail')
        repo = r['repository']
        flags = r['collection']['truncated']
        truncation.update(k for k, v in flags.items() if v)
        files = r['files']
        unique_paths = len({f['path'] for f in files}) == len(files)
        complete = not flags.get('files', False) and unique_paths and len(files) == r['changedFiles']
        file_lines_match = sum(f['additions'] + f['deletions'] for f in files) == r['additions'] + r['deletions']
        kinds, line_kinds = collections.Counter(), collections.Counter()
        for f in files:
            kind = file_kind(f['path'])
            kinds[kind] += 1
            line_kinds[kind] += f['additions'] + f['deletions']
        if complete:
            for kind in kinds:
                file_summary[kind]['prs_with_category'] += 1
                file_summary[kind]['files'] += kinds[kind]
                if file_lines_match:
                    file_summary[kind]['lines_in_reconciled_prs'] += line_kinds[kind]
        created = pd.Timestamp(r['createdAt'])
        collected = pd.Timestamp(r['collection']['collectedAt'])
        events = []
        for name, entries in [('review', r.get('reviews', [])), ('pr_comment', r.get('comments', [])),
                              ('inline_review_comment', [x for t in r.get('reviewThreads', []) for x in t.get('comments', [])])]:
            events.extend((name, e) for e in entries if str(e.get('body') or '').strip())
        nonbot_all = external_all = external_30 = 0
        event_ids = set()
        for j, (source, e) in enumerate(events):
            key = e.get('id') or e.get('url')
            if not key:
                key = f'{source}:missing-id:{j}'
                counts['event_without_id_or_url'] += 1
            if key in event_ids:
                continue
            event_ids.add(key)
            actor = e.get('author') or {}
            if not nonbot_candidate(actor):
                continue
            nonbot_all += 1
            if (actor.get('login') or '').lower() == ((r.get('author') or {}).get('login') or '').lower():
                continue
            external_all += 1
            stamp = e.get('createdAt') or e.get('submittedAt')
            if not stamp:
                counts['external_events_missing_timestamp'] += 1
                continue
            dt = (pd.Timestamp(stamp) - created).total_seconds() / 86400
            counts['external_events_before_creation'] += dt < 0
            external_30 += 0 <= dt < 30
        title_candidate = bool(re.search(r'(?i)\b(fix(?:es|ed|ing)?|bug(?:fix)?|regression|hotfix)\b', r['title']))
        labels = [x.get('name', '') for x in (r.get('labels') or {}).get('nodes', [])]
        bug_labels = [x for x in labels if re.sub('[^a-z0-9]', '', x.lower()) in {'bug', 'bugfix', 'bugfixes', 'typebug', 'kindbug', 'typebugfix', 'hotfix', 'typehotfix'}]
        row = dict(id=pid, repository_id=repo['id'], repository=repo['nameWithOwner'], url=r['url'],
                   agent=agent, signal_agents='|'.join(signals), multi_agent_signal=len(signals)>1,
                   quarter=quarter(r['createdAt']), created_at=r['createdAt'], collected_at=r['collection']['collectedAt'],
                   age_days=(collected-created).total_seconds()/86400, state=r['state'], merged=r['state']=='MERGED',
                   changed_files=r['changedFiles'], additions=r['additions'], deletions=r['deletions'],
                   changed_lines=r['additions']+r['deletions'], cross_file=r['changedFiles']>=2,
                   files_listed=len(files), files_complete=complete, file_lines_match=file_lines_match,
                   directories=len({str(PurePosixPath(f['path']).parent) for f in files}) if complete else None,
                   source_files=kinds['source_candidate'] if complete else None,
                   source_lines=line_kinds['source_candidate'] if complete and file_lines_match else None,
                   source_cross_file=kinds['source_candidate']>=2 if complete else None,
                   has_exported_text=bool(r.get('humanReviewTexts')), exported_text_count=len(r.get('humanReviewTexts') or []),
                   nonbot_text_count=nonbot_all, external_text_count=external_all, external_text_30d_count=external_30,
                   has_external_text_all=external_all>0, has_external_text_30d=external_30>0,
                   has_review=bool(r.get('reviews')), has_comment=bool(r.get('comments')),
                   interaction_complete=not any(flags.get(k, False) for k in ['reviews','comments','reviewThreads']),
                   commits_listed=len(r.get('commits') or []), stars=repo['stargazerCount'],
                   language=(repo.get('primaryLanguage') or {}).get('name'), fix_title_candidate=title_candidate,
                   bug_label=bool(bug_labels), base_oid=r.get('baseRefOid'), head_oid=r.get('headRefOid'))
        records.append(row)
        if title_candidate or bug_labels:
            candidate_records.append(dict(id=pid, url=r['url'], agent=agent, quarter=row['quarter'], title=r['title'],
                                          title_keyword=title_candidate, matched_bug_labels='|'.join(bug_labels),
                                          validated_task='', evidence_pointer=''))
        if not complete or not file_lines_match or row['multi_agent_signal']:
            bad.append(dict(id=pid, files_complete=complete, file_lines_match=file_lines_match,
                            multi_agent_signal=row['multi_agent_signal'], url=r['url']))
    df = pd.DataFrame(records)
    if args.expected_prs is not None:
        assert len(df) == args.expected_prs, f'Expected {args.expected_prs}, got {len(df)}'
    assert counts['input_rows'] == len(df) == len(seen)
    assert (df.changed_lines == df.additions + df.deletions).all()
    assert df.source_files.notna().equals(df.files_complete)
    assert (df.source_lines.notna() == (df.files_complete & df.file_lines_match)).all()
    assert set(df.quarter) <= set(QUARTERS)
    assert (df.age_days >= 0).all()
    groups = pd.DataFrame([{'agent':'all', **summarize(df)}] +
                          [{'agent':a, **summarize(df[df.agent == a])} for a in AGENTS + sorted(set(df.agent)-set(AGENTS))])
    quarters = pd.DataFrame([{'quarter':q, **summarize(g)} for q,g in df.groupby('quarter')])
    temporal = pd.DataFrame([{'quarter':q,'agent':a,**summarize(g)} for (q,a),g in df.groupby(['quarter','agent'])])
    assert groups[groups.agent!='all'].n.sum() == quarters.n.sum() == temporal.n.sum() == len(df)
    df['star_bucket'] = pd.cut(df.stars, [-1,0,10,100,1000,10000,np.inf], labels=['0','1-10','11-100','101-1000','1001-10000','>10000'])
    star_summary = pd.DataFrame([{'star_bucket':str(k),**summarize(g)} for k,g in df.groupby('star_bucket',observed=True)])
    df.to_csv(out/'pr_metrics.csv', index=False)
    groups.to_csv(out/'summary_agent.csv', index=False)
    quarters.to_csv(out/'summary_quarter.csv', index=False)
    temporal.to_csv(out/'summary_agent_quarter.csv', index=False)
    star_summary.to_csv(out/'summary_star_bucket.csv', index=False)
    pd.DataFrame(bad).to_csv(out/'quality_flags.csv', index=False)
    pd.DataFrame(candidate_records).to_csv(out/'task_rule_candidates.csv', index=False)
    candidate_quarters = df.groupby('quarter').agg(all_prs=('id','size'),title_candidates=('fix_title_candidate','sum'),bug_label_prs=('bug_label','sum')).reset_index()
    candidate_quarters['title_candidate_pct'] = 100*candidate_quarters.title_candidates/candidate_quarters.all_prs
    candidate_quarters.to_csv(out/'task_candidate_quarters.csv', index=False)
    candidate_scope = pd.DataFrame([{'quarter':q,'agent':a,**summarize(g)} for (q,a),g in df[df.fix_title_candidate].groupby(['quarter','agent'])])
    candidate_scope.to_csv(out/'title_candidate_scope_agent_quarter.csv', index=False)
    pd.DataFrame([{'category':k,**v,'eligible_complete_prs':int(df.files_complete.sum()),'eligible_reconciled_prs':int((df.files_complete & df.file_lines_match).sum())} for k,v in file_summary.items()]).to_csv(out/'file_category_summary.csv',index=False)
    df.groupby('repository_id').agg(repository=('repository','last'),prs=('id','size'),quarters=('quarter','nunique'),stars_min=('stars','min'),stars_max=('stars','max')).reset_index().to_csv(out/'repository_period_support.csv',index=False)
    pd.DataFrame([{'group':'all_delivered',**summarize(df)},{'group':'excluding_multiple_agent_signals',**summarize(df[~df.multi_agent_signal])}]).to_csv(out/'attribution_sensitivity.csv',index=False)
    summary = {'analysis_scope':'every record in prs.jsonl; no additional sampling', 'raw_rows':int(counts['input_rows']),
               'analyzed_prs':len(df),'unique_pr_ids':df.id.nunique(),'repositories':df.repository_id.nunique(),
               'sampling_performed':False,'bootstrap_performed':False,'agent_label_source':'largest manifest lookup only; no row filtering',
               'agent_counts':df.agent.value_counts().to_dict(),'state_counts':df.state.value_counts().to_dict(),
               'quarter_counts':df.quarter.value_counts().sort_index().to_dict(),'multi_agent_signals':int(df.multi_agent_signal.sum()),
               'search_signal_counts_nonexclusive':dict(signal_counts),'files_complete':int(df.files_complete.sum()),
               'files_incomplete':int((~df.files_complete).sum()),'file_lines_match':int(df.file_lines_match.sum()),
               'truncated_fields':dict(truncation),'diagnostic_counts':dict(counts),'zero_file_prs':int((df.changed_files==0).sum()),
               'zero_line_prs':int((df.changed_lines==0).sum()),'collection_min':df.collected_at.min(),'collection_max':df.collected_at.max(),
               'min_age_days':df.age_days.min(),'patch_collection_enabled':metadata['includePatch'],'checks_collection_enabled':metadata['includeChecks'],
               'overall':groups.iloc[0].to_dict()}
    write_json(out/'summary.json', summary)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig, axes = plt.subplots(1,2,figsize=(8,3.1))
    for a,color in zip(AGENTS,COLORS):
        g=df[df.agent==a]
        for ax,col in zip(axes,['changed_files','changed_lines']):
            x=np.sort(g[col].to_numpy());ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',color=color,label=f'{NAMES[a]} (n={len(g):,})')
    for ax,label in zip(axes,['Changed files per PR','Added + deleted lines per PR']):
        ax.set_xscale('symlog',linthresh=1);ax.set_xlabel(label);ax.set_ylabel('Empirical cumulative fraction');ax.grid(alpha=.15)
    axes[1].legend(fontsize=6.5,loc='lower right');fig.tight_layout()
    for suffix in ['pdf','png']:fig.savefig(out/'figures'/f'01_scope_ecdf.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3.1))
    for a,color in zip(AGENTS,COLORS):
        g=temporal[temporal.agent==a].set_index('quarter').reindex(QUARTERS)
        axes[0].plot(range(6),g.n,marker='o',color=color,label=NAMES[a])
        axes[1].plot(range(6),g.changed_lines_median.where(g.n>=20),marker='o',color=color)
    for ax in axes:ax.set_xticks(range(6),QUARTERS,rotation=35);ax.set_yscale('log');ax.grid(alpha=.15)
    axes[0].set_ylabel('PRs in all delivered records');axes[0].legend(fontsize=7)
    axes[1].set_ylabel('Median added + deleted lines');fig.tight_layout()
    for suffix in ['pdf','png']:fig.savefig(out/'figures'/f'02_quarter_scope.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,3.1));g=quarters.set_index('quarter').reindex(QUARTERS)
    ax.plot(range(6),g.has_exported_text_pct,marker='o',label='Exported candidate-account text')
    ax.plot(range(6),g.has_external_text_30d_pct,marker='s',label='Non-author candidate text, first 30 days')
    ax.set_xticks(range(6),QUARTERS);ax.set_ylabel('PRs with detected text (%)');ax.legend(fontsize=7);ax.grid(alpha=.15);fig.tight_layout()
    for suffix in ['pdf','png']:fig.savefig(out/'figures'/f'03_feedback_coverage.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    build_reports(out,df,summary,groups,quarters,temporal,candidate_quarters)
    manifest=[]
    for f in input_files:
        after=(f.stat().st_size,f.stat().st_mtime_ns,sha256(f))
        assert after==input_state[f.name],f'Input changed: {f.name}'
        manifest.append({'file':f.name,'bytes':after[0],'sha256':after[2]})
    pd.DataFrame(manifest).to_csv(out/'source_manifest.csv',index=False)
    before_file=out/'paper_unchanged_before.json'
    paper_ok=None
    if before_file.exists():
        paper_ok=all(Path(p).exists() and sha256(Path(p))==v for p,v in json.loads(before_file.read_text()).items())
        assert paper_ok,'A fingerprinted paper file changed during this task'
    from PIL import Image
    figures_ok=True
    for f in (out/'figures').glob('*.png'):
        with Image.open(f) as im:im.verify()
    for f in (out/'figures').glob('*.pdf'):
        assert f.stat().st_size>1000
    checks={'status':'passed','all_raw_rows_analyzed':counts['input_rows']==len(df),
            'unique_prs':int(df.id.nunique()),'analyzed_prs':len(df),'agent_total':int(groups[groups.agent!='all'].n.sum()),
            'quarter_total':int(quarters.n.sum()),'agent_quarter_total':int(temporal.n.sum()),'no_sampling':True,
            'no_bootstrap':True,'nonnegative_API_counts':True,'path_eligibility_checked':True,'input_files_unchanged':True,
            'paper_files_unchanged':paper_ok,'figure_files_programmatically_verified':figures_ok,'manual_visual_review_performed':False}
    write_json(out/'checks.json',checks)
    write_json(out/'runtime.json',{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                                  'matplotlib':matplotlib.__version__,'analysis_script_sha256':sha256(Path(__file__)),
                                  'helper_script_sha256':sha256(Path(__file__).with_name('analyze_github_delivery.py'))})
    print(json.dumps(checks,ensure_ascii=False,indent=2))
    print(groups[['agent','n','changed_files_median','changed_lines_median','cross_file_pct','has_exported_text_pct','has_external_text_30d_pct','fix_title_candidate_n']].to_string(index=False))
    print(quarters[['quarter','n','changed_files_median','changed_lines_median','cross_file_pct']].to_string(index=False))


if __name__=='__main__':
    main()
