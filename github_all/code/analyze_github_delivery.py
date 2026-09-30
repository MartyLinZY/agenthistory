#!/usr/bin/env python3
"""Analyze one manifest-defined PR cohort without modifying source exports.

Descriptive snapshot analysis only: no semantic human/task labels are inferred.
Run: python3 scripts/analyze_github_delivery.py --source /path/to/github --output /path/to/output
"""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AGENTS = ['codex', 'copilot', 'claude_code', 'jules', 'devin']
NAMES = dict(zip(AGENTS, ['Codex', 'Copilot', 'Claude Code', 'Jules', 'Devin']))
QUARTERS = ['2025Q1','2025Q2','2025Q3','2025Q4','2026Q1','2026Q2']
COLORS = ['#0072B2','#D55E00','#009E73','#CC79A7','#7A6500']
SOURCE_EXT = set('.c .cc .cpp .cxx .h .hpp .cs .go .java .js .jsx .ts .tsx .py .rb .rs .php .swift .kt .kts .scala .vue .svelte .dart .ex .exs .erl .lua .m .mm .r .jl .sh .sql .html .css .scss .sass .less .f .f90 .pl .pm .clj .cljs .hs .ml .fs .sol .zig'.split())

def records(path):
    with path.open() as f:
        for n,line in enumerate(f,1):
            try: yield json.loads(line)
            except Exception as e: raise ValueError(f'{path.name}:{n}: invalid JSON') from e

def quarter(value):
    return f'{value[:4]}Q{(int(value[5:7])-1)//3+1}'

def file_kind(path):
    """Conservative path-only proxy, not validated production-code semantics."""
    p=PurePosixPath(path.lower()); parts=set(p.parts); name=p.name
    if name in {'package-lock.json','yarn.lock','pnpm-lock.yaml','poetry.lock','uv.lock','cargo.lock','composer.lock','gemfile.lock','go.sum'} or name.endswith('.lock'):
        return 'lockfile'
    if parts & {'node_modules','vendor','dist','build','generated','__generated__'} or re.search(r'(\.min\.(js|css)$|\.g\.cs$|\.pb\.(go|cc|h)$)',name):
        return 'generated_or_vendor_path'
    if parts & {'test','tests','__tests__','spec','specs','testing'} or re.search(r'(^test_|_test\.|\.(test|spec)\.|test\.java$)',name):
        return 'test_path'
    if p.suffix in {'.md','.rst','.adoc'} or parts & {'doc','docs','documentation'} or name.startswith(('readme','license','changelog')):
        return 'documentation_path'
    if p.suffix in SOURCE_EXT: return 'source_candidate'
    if p.suffix in {'.yaml','.yml','.json','.toml','.ini','.cfg','.xml'} or '.github' in parts or name in {'dockerfile','makefile'}:
        return 'configuration_path'
    return 'other_or_unknown'

def summarize(g):
    out={'n':len(g),'repositories':g.repository_id.nunique()}
    for col in ['changed_files','additions','deletions','changed_lines','listed_directories','source_files','source_lines']:
        v=g[col].dropna()
        out.update({f'{col}_n':len(v), f'{col}_median':v.median(), f'{col}_q25':v.quantile(.25),f'{col}_q75':v.quantile(.75),f'{col}_p90':v.quantile(.9),f'{col}_sum':v.sum()})
    for col in ['cross_file','source_cross_file','has_exported_text','has_nonbot_nonself_text_30d','has_formal_review','merged','files_complete','file_lines_reconcile','multi_agent_signal','fix_title_candidate']:
        v=g[col].dropna();out[col+'_n']=int(v.sum());out[col+'_eligible']=len(v);out[col+'_pct']=100*v.mean()
    return out

def cluster_interval(g,seed=20260923,reps=2000):
    # Equal-probability repository resampling: sensitivity to sample composition,
    # NOT a design-based CI for the discovery population.
    a=g.groupby('repository_id').cross_file.agg(['sum','count']).to_numpy()
    if len(a)<20: return [None,None]
    rng=np.random.default_rng(seed); estimates=[]
    for _ in range(reps):
        ix=rng.integers(0,len(a),len(a)); v=a[ix].sum(axis=0);estimates.append(100*v[0]/v[1])
    return np.quantile(estimates,[.025,.975]).tolist()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    src=args.source;out=args.output;out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    metadata=json.loads((src/'run_metadata.json').read_text());cohort='unified-0.1pct'
    selections={};cohort_counts=collections.Counter();cohort_sets=collections.defaultdict(set)
    for r in records(src/'selections.jsonl'):
        cohort_counts[r['cohort']]+=1;cohort_sets[r['cohort']].add(r['node_id'])
        if r['cohort']==cohort:
            if r['node_id'] in selections:raise ValueError('Duplicate target selection ID')
            selections[r['node_id']]=r
    sel=pd.DataFrame(selections.values());sel['quarter']=sel.created_at.map(quarter)
    rows=[];seen=set();duplicates=0;all_detail_rows=0;source_times=[];missing_text_times=0;unexpected_dates=0
    paths=collections.Counter();truncs=collections.Counter();review_sources=collections.Counter();quality=[];candidates=[]
    for r in records(src/'prs.jsonl'):
        all_detail_rows+=1
        if r['id'] not in selections:continue
        if r['id'] in seen:duplicates+=1;continue
        seen.add(r['id']);s=selections[r['id']]
        if pd.Timestamp(s['created_at'])!=pd.Timestamp(r['createdAt']):
            raise ValueError('Selection/detail metadata mismatch')
        created=pd.Timestamp(r['createdAt']);collected=pd.Timestamp(r['collection']['collectedAt']);age=(collected-created).total_seconds()/86400
        source_times.append(r['collection']['collectedAt']);flags=r['collection']['truncated'];truncs.update(k for k,v in flags.items() if v)
        files=r['files'];complete=not flags.get('files',False) and len(files)==r['changedFiles'] and len({x['path'] for x in files})==len(files)
        linematch=sum(x['additions']+x['deletions'] for x in files)==r['additions']+r['deletions']
        kinds=collections.Counter();flines=collections.Counter()
        for f in files:
            k=file_kind(f['path']);kinds[k]+=1;flines[k]+=f['additions']+f['deletions'];paths[k]+=1
        source_files=kinds['source_candidate'] if complete else None
        source_lines=flines['source_candidate'] if complete and linematch else None
        text=r.get('humanReviewTexts') or [];review_sources.update(t['source'] for t in text)
        raw_events=[]
        for source,items in [('review',r['reviews']),('pr_comment',r['comments']),('inline_review_comment',[c for t in r['reviewThreads'] for c in t['comments']])]:
            raw_events.extend((source,e) for e in items if str(e.get('body') or '').strip())
        nonbot30=0;nonbotall=0;nonbot_nonself_all=0;eventseen=set()
        for source,e in raw_events:
            key=e.get('id') or e.get('url')
            if key in eventseen:continue
            eventseen.add(key)
            actor=e.get('author') or {};login=(actor.get('login') or '').lower();author=(r.get('author') or {}).get('login','').lower()
            t=e.get('createdAt') or e.get('submittedAt')
            if not t:missing_text_times+=1;continue
            delta=(pd.Timestamp(t)-created).total_seconds()/86400
            if delta<0:unexpected_dates+=1
            if login and actor.get('__typename')!='Bot' and not login.endswith('[bot]'):
                nonbotall+=1
                if login!=author:
                    nonbot_nonself_all+=1
                    if 0<=delta<30:nonbot30+=1
        signal_agents=sorted(set(x['agent'] for x in r['searchMatches']))
        title_candidate=bool(re.search(r'(?i)\b(fix(?:es|ed|ing)?|bug(?:fix)?|regression|hotfix)\b',r['title']))
        interaction_complete=not any(flags.get(k,False) for k in ['comments','reviews','reviewThreads'])
        row={'id':r['id'],'repository':r['repository']['nameWithOwner'],'repository_id':r['repository']['id'],'selection_repository':s['repository'],'repository_name_changed':s['repository']!=r['repository']['nameWithOwner'],'url':r['url'],'agent':s['agent'],'quarter':quarter(r['createdAt']),'created_at':r['createdAt'],'collected_at':r['collection']['collectedAt'],'age_at_collection_days':age,'state':r['state'],'merged':r['state']=='MERGED','changed_files':r['changedFiles'],'additions':r['additions'],'deletions':r['deletions'],'changed_lines':r['additions']+r['deletions'],'cross_file':r['changedFiles']>=2,'files_listed':len(files),'files_complete':complete,'file_lines_reconcile':linematch,'listed_directories':len({str(PurePosixPath(f['path']).parent) for f in files}) if complete else None,'source_files':source_files,'source_lines':source_lines,'source_cross_file':source_files>=2 if source_files is not None else None,'exported_text_count':len(text),'has_exported_text':bool(text),'nonbot_text_all_count':nonbotall,'nonbot_nonself_text_all_count':nonbot_nonself_all,'nonbot_nonself_text_30d_count':nonbot30,'has_nonbot_nonself_text_30d':bool(nonbot30),'interaction_collections_complete':interaction_complete,'has_formal_review':bool(r['reviews']),'commits_listed':len(r['commits']),'stars':r['repository']['stargazerCount'],'language':(r['repository'].get('primaryLanguage') or {}).get('name'),'multi_agent_signal':len(signal_agents)>1,'signal_agents':'|'.join(signal_agents),'base_oid':r['baseRefOid'],'head_oid':r['headRefOid'],'fix_title_candidate':title_candidate}
        rows.append(row)
        if not complete or not linematch or row['multi_agent_signal']:quality.append({'id':r['id'],'url':r['url'],'files_complete':complete,'file_lines_reconcile':linematch,'multi_agent_signal':row['multi_agent_signal']})
        if title_candidate:candidates.append({'id':r['id'],'url':r['url'],'agent':s['agent'],'quarter':row['quarter'],'title':r['title'],'screen':'title_keyword_only','adjudicated_task':'','evidence_pointer':''})
    assert duplicates==0,'Duplicate detail IDs must be reconciled'
    df=pd.DataFrame(rows);assert len(df)==len(seen)
    for k in ['changed_files','additions','deletions']:
        assert df[k].notna().all() and (df[k]>=0).all()
    df.to_csv(out/'pr_metrics.csv',index=False)
    pd.DataFrame(quality).to_csv(out/'quality_flags.csv',index=False)
    pd.DataFrame(candidates).to_csv(out/'bugfix_title_candidates.csv',index=False)
    sel[~sel.node_id.isin(seen)].to_csv(out/'missing_selected_prs.csv',index=False)
    coverage=[]
    for q in QUARTERS:
        for a in AGENTS:
            n=int(((sel.agent==a)&(sel.quarter==q)).sum());g=df[(df.agent==a)&(df.quarter==q)]
            coverage.append({'quarter':q,'agent':a,'selected':n,'retrieved':len(g),'missing':n-len(g),'retrieved_pct':100*len(g)/n if n else None,'multi_agent_signals':int(g.multi_agent_signal.sum()),'files_incomplete':int((~g.files_complete).sum())})
    cov=pd.DataFrame(coverage);cov.to_csv(out/'coverage_agent_quarter.csv',index=False)
    summaries=[]
    for a in ['all']+AGENTS:
        g=df if a=='all' else df[df.agent==a];v={'agent':a,**summarize(g)};lo,hi=cluster_interval(g);v.update(cross_file_cluster_lo=lo,cross_file_cluster_hi=hi);summaries.append(v)
    summ=pd.DataFrame(summaries);summ.to_csv(out/'summary_agent.csv',index=False)
    temporal=pd.DataFrame([{'quarter':q,'agent':a,**summarize(g)} for (q,a),g in df.groupby(['quarter','agent'])]);temporal.to_csv(out/'summary_agent_quarter.csv',index=False)
    qsummary=pd.DataFrame([{'quarter':q,**summarize(g)} for q,g in df.groupby('quarter')]);qsummary.to_csv(out/'summary_quarter.csv',index=False)
    df['star_bucket']=pd.cut(df.stars,bins=[-1,0,10,100,1000,10000,np.inf],labels=['0','1-10','11-100','101-1000','1001-10000','>10000'])
    stars=df.groupby('star_bucket',observed=False).agg(prs=('id','size'),repositories=('repository_id','nunique')).reset_index();stars['pr_pct']=stars.prs/len(df)*100;stars.to_csv(out/'sample_star_distribution.csv',index=False)
    repos=df.groupby('repository_id').agg(repository=('repository','last'),sample_prs=('id','size'),stars_min=('stars','min'),stars_max=('stars','max'),first_observed=('collected_at','min'),last_observed=('collected_at','max')).reset_index().sort_values('stars_max',ascending=False)
    repos.to_csv(out/'sample_repositories.csv',index=False)
    windows=list(records(src/'search_windows.jsonl'));wd=pd.DataFrame(windows);wd['quarter']=wd.start_at.map(quarter)
    leaf=wd[wd.status=='complete'].copy();leaf['count_gap']=(leaf.issue_count-leaf.fetched_count).clip(lower=0)
    leaf.groupby(['agent','quarter']).agg(leaf_windows=('id','count'),truncated_windows=('truncated','sum'),issue_counts_nonunique=('issue_count','sum'),fetched_counts_nonunique=('fetched_count','sum'),count_gap_nonunique=('count_gap','sum')).reset_index().to_csv(out/'search_window_audit.csv',index=False)
    fail=collections.Counter(r['stage'] for r in records(src/'failures.jsonl'))
    # Do not publish error bodies, which may contain irrelevant payloads.
    missing_rate=(len(sel)-len(df))/len(sel)
    overall=summaries[0]
    frame_reported=metadata['counts']['unique_prs']
    summary={'cohort':cohort,'selection_counts':dict(cohort_counts),'selection_unique_counts':{k:len(v) for k,v in cohort_sets.items()},'selected':len(sel),'retrieved':len(df),'missing':len(sel)-len(df),'retrieval_pct':100*(1-missing_rate),'all_detail_rows':all_detail_rows,'duplicate_target_details':duplicates,'repository_name_changed':int(df.repository_name_changed.sum()),'discovery_unique_prs_metadata_only':frame_reported,'overall_selected_to_metadata_frame_pct':100*len(sel)/frame_reported,'repositories':df.repository_id.nunique(),'creation_min':df.created_at.min(),'creation_max':df.created_at.max(),'collection_min':min(source_times),'collection_max':max(source_times),'min_age_at_collection_days':df.age_at_collection_days.min(),'multi_agent_signals':int(df.multi_agent_signal.sum()),'truncation_flags':dict(truncs),'files_complete':int(df.files_complete.sum()),'file_lines_reconcile':int(df.file_lines_reconcile.sum()),'feedback_sources':dict(review_sources),'missing_text_timestamps':missing_text_times,'negative_text_time_deltas':unexpected_dates,'search_windows':len(wd),'complete_leaf_windows':len(leaf),'truncated_leaf_windows':int(leaf.truncated.sum()),'leaf_issue_minus_fetched_nonunique':int(leaf.count_gap.sum()),'failure_events_by_stage':dict(fail),'source_patch_enabled':metadata['includePatch'],'source_checks_enabled':metadata['includeChecks'],'binary_diff_policy':'Not identifiable from this export; API line counts are text-diff statistics, not binary volume.','cross_file_missingness_bounds_pct':[100*int(df.cross_file.sum())/len(sel),100*(int(df.cross_file.sum())+len(sel)-len(df))/len(sel)],'same_repository_across_all_six_quarters':int((df.groupby('repository_id').quarter.nunique()==6).sum()),'overall':overall}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda v:v.item() if hasattr(v,'item') else v))
    manifest=[]
    for path in sorted(src.iterdir()):
        if path.suffix not in {'.json','.jsonl'}:continue
        before=path.stat();h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
        after=path.stat();assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns,'Source changed during hashing'
        manifest.append({'file':path.name,'bytes':after.st_size,'sha256':h.hexdigest()})
    pd.DataFrame(manifest).to_csv(out/'source_manifest.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(8.0,3.0))
    for a,c in zip(AGENTS,COLORS):
        g=df[df.agent==a]
        for ax,col in zip(axs,['changed_files','changed_lines']):
            x=np.sort(g[col].to_numpy());ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',label=f'{NAMES[a]} (n={len(g):,})',color=c)
    for ax,label in zip(axs,['Changed files per PR','Added + deleted lines per PR']):
        ax.set_xscale('symlog',linthresh=1);ax.set_xlabel(label);ax.set_ylabel('Empirical cumulative fraction');ax.grid(alpha=.15)
    axs[1].legend(fontsize=7,loc='lower right');fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(out/'figures'/f'01_scope_ecdf.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(8.0,3.1))
    for a,c in zip(AGENTS,COLORS):
        g=temporal[temporal.agent==a].set_index('quarter').reindex(QUARTERS)
        axs[0].plot(range(6),g.n,marker='o',label=NAMES[a],color=c)
        # Suppress median trend points based on fewer than 20 PRs.
        axs[1].plot(range(6),g.changed_lines_median.where(g.n>=20),marker='o',color=c)
    for ax in axs:ax.set_xticks(range(6),QUARTERS,rotation=35);ax.grid(alpha=.15)
    axs[0].set_ylabel('Retrieved PRs in the 0.1% cohort');axs[0].set_yscale('log');axs[0].legend(fontsize=7)
    axs[1].set_ylabel('Median added + deleted lines');axs[1].set_yscale('log');fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(out/'figures'/f'02_quarter_scope.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='overall'},ensure_ascii=False,indent=2))
    print(summ[['agent','n','repositories','changed_files_median','changed_lines_median','cross_file_pct','has_exported_text_pct','has_nonbot_nonself_text_30d_pct','fix_title_candidate_pct']].to_string(index=False))

if __name__=='__main__':main()
