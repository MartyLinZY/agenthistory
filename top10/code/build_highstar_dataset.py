#!/usr/bin/env python3
"""Package the selected ten-repository census and run descriptive analyses.

No resampling. Two explicitly excluded, inaccessible yt-dlp PRs remain in an
exclusion ledger. Original acquisition artifacts are never modified.
"""
import argparse
import gzip
import collections
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collect_repo_contributions import metrics, read_jsonl, START, END, QUARTERS, AGENTS, now
from analyze_github_delivery import file_kind


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def percentage(a, b):
    return float(100*a/b) if b else None


def stage_flags(path: str, kind: str) -> dict:
    p = PurePosixPath(path.lower())
    parts = set(p.parts)
    return {
        'code_path': kind == 'source_candidate',
        'test_path': kind == 'test_path',
        'ci_path': '.github/workflows' in str(p) or '.circleci' in parts or p.name in {'jenkinsfile', '.gitlab-ci.yml', '.travis.yml'},
        'deployment_path': bool(parts & {'deploy','deployment','deployments','k8s','kubernetes','helm','terraform'}) or p.name.startswith(('dockerfile','docker-compose','compose.y')) or p.suffix == '.tf',
        'documentation_path': kind == 'documentation_path',
    }


def contribution(g: pd.DataFrame) -> dict:
    a = g[g.agent_detected]
    known = g[g.source_lines_eligible]
    known_agent = known[known.agent_detected]
    missing = g[~g.source_lines_eligible]
    source_total = float(known.source_churn.sum())
    source_agent = float(known_agent.source_churn.sum())
    unknown_agent = float(missing.loc[missing.agent_detected, 'changed_lines'].sum())
    unknown_other = float(missing.loc[~missing.agent_detected, 'changed_lines'].sum())
    complete = len(missing) == 0
    return {
        'pr_count': len(g), 'agent_pr_count': len(a), 'agent_pr_pct': percentage(len(a), len(g)),
        'all_additions': int(g.additions.sum()), 'agent_additions': int(a.additions.sum()),
        'agent_additions_pct': percentage(a.additions.sum(), g.additions.sum()),
        'all_deletions': int(g.deletions.sum()), 'agent_deletions': int(a.deletions.sum()),
        'all_text_churn': int(g.changed_lines.sum()), 'agent_text_churn': int(a.changed_lines.sum()),
        'agent_text_churn_pct': percentage(a.changed_lines.sum(), g.changed_lines.sum()),
        'source_eligible_prs': len(known), 'source_coverage_pct': percentage(len(known), len(g)),
        'source_coverage_complete': complete, 'known_source_churn': int(source_total),
        'known_agent_source_churn': int(source_agent),
        'agent_source_churn_pct': percentage(source_agent, source_total) if complete else None,
        'agent_source_additions_pct': percentage(known_agent.source_additions.sum(), known.source_additions.sum()) if complete else None,
        'source_share_lower_bound_pct': percentage(source_agent, source_total+unknown_other),
        'source_share_upper_bound_pct': percentage(source_agent+unknown_agent, source_total+unknown_agent),
        'source_unknown_agent_text_churn': int(unknown_agent),
        'source_unknown_other_text_churn': int(unknown_other),
    }


def complexity(g: pd.DataFrame) -> dict:
    r = {'pr_count': len(g), 'cross_file_prs': int(g.cross_file.sum()),
         'cross_file_pct': percentage(g.cross_file.sum(), len(g))}
    for col in ['changed_files','changed_lines','source_churn','listed_directories']:
        v = g[col].dropna()
        r.update({col+'_eligible_n': len(v), col+'_median': float(v.median()) if len(v) else None,
                  col+'_p90': float(v.quantile(.9)) if len(v) else None,
                  col+'_mean': float(v.mean()) if len(v) else None})
    return r


def md_table(headers: list, rows: list) -> str:
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def fmt(x, digits=2) -> str:
    if x is None or pd.isna(x):
        return 'NA'
    if 0 < x < .0005:
        return '<0.001%'
    if 0 < x < .01 and digits == 2:
        digits = 3
    return f'{x:.{digits}f}%'


def build(source: Path, target: Path) -> None:
    if source.resolve()==target.resolve():
        raise ValueError('Source and output must be different directories')
    raw_only = (source/'raw/metadata/repositories.json').exists()
    packaged = raw_only or (source/'metadata/manifest.json').exists()
    metadata_source = source/'raw/metadata' if raw_only else (source/'metadata' if packaged else source)
    raw_source = source/'raw' if packaged else source/'repositories'
    target.mkdir(parents=True, exist_ok=True)
    for name in ['raw','metadata','features','analysis','figures','code']:
        (target/name).mkdir(exist_ok=True)
    repos = json.loads((metadata_source/'repositories.json').read_text())
    assert len(repos) == 10 and not any(r['repository']=='torvalds/linux' for r in repos)
    audit = json.loads((raw_source/'yt-dlp__yt-dlp/rest_recovery_audit.json').read_text())
    excluded = audit['unresolved']
    assert {e['number'] for e in excluded} == {15386,15481}
    for e in excluded:
        e.update(repository='yt-dlp/yt-dlp', reason='search_identity_present_but_detail_unresolvable',
                 policy='Excluded from analyzable population by explicit user instruction; not imputed.')
    write_json(target/'metadata/exclusions.json', excluded)
    for name in ['repositories.json','selection_protocol.json','engineering_evidence.json','initial_counts.json','high_star_search_raw.json']:
        shutil.copy2(metadata_source/name, target/'metadata'/name)
    shutil.copy2(Path(__file__), target/'code'/Path(__file__).name)
    for name in ['collect_repo_contributions.py','analyze_github_delivery.py','add_highstar_monthly_analysis.py']:
        shutil.copy2(Path(__file__).parent/name, target/'code'/name)
    rows = []; coverage = []; validation = []; seen = set(); file_buffer = []; file_count = 0
    file_schema = pa.schema([('pr_id',pa.string()),('repository',pa.string()),('path',pa.string()),
        ('additions',pa.int64()),('deletions',pa.int64()),('file_kind',pa.string()),
        ('pr_file_list_complete',pa.bool_()),('pr_file_lines_reconcile',pa.bool_())]+
        [(k,pa.bool_()) for k in stage_flags('src/a.py','source_candidate')])
    file_writer = pq.ParquetWriter(target/'features/files.parquet', file_schema, compression='zstd')
    archive_index = {r['repository']: r for r in json.loads((metadata_source/'raw_archives.json').read_text())} if (metadata_source/'raw_archives.json').exists() else {}
    for repo in repos:
        name = repo['repository']; folder = raw_source/name.replace('/','__')
        raw = folder/'prs.jsonl'; dest = target/'raw'/name.replace('/','__')/'prs.jsonl';dest.parent.mkdir(exist_ok=True)
        if raw.exists():
            shutil.copy2(raw,dest)
        else:
            raw = raw.with_name(raw.name+'.gz')
            with gzip.open(raw,'rb') as inp, dest.open('wb') as out:
                shutil.copyfileobj(inp,out)
        status = json.loads((folder/('acquisition_status.json' if packaged else 'status.json')).read_text());write_json(dest.parent/'acquisition_status.json',status)
        if (folder/'rest_recovery_audit.json').exists():
            shutil.copy2(folder/'rest_recovery_audit.json',dest.parent/'rest_recovery_audit.json')
        assert status['enumeration_complete'] or name == 'yt-dlp/yt-dlp'
        raw_stats = {c:collections.Counter() for c in ['created','merged']}
        start_index = len(rows)
        for p in read_jsonl(dest):
            assert p['id'] not in seen;seen.add(p['id'])
            assert not any(p['id']==e['id'] for e in excluded)
            assert p['repository']['nameWithOwner'] == name
            assert START <= p['createdAt'] < END or (p['createdAt']<START and p['mergedAt'] and START<=p['mergedAt']<END)
            for field in ['additions','deletions','changedFiles']:
                assert isinstance(p[field],int) and p[field]>=0
            m = metrics(p)
            fs = (p.get('files') or {}).get('nodes') or []
            phases = collections.Counter()
            for f in fs:
                kind = file_kind(f['path']);flags = stage_flags(f['path'],kind);phases.update({k:int(v) for k,v in flags.items()})
                file_buffer.append({'pr_id':p['id'],'repository':name,'path':f['path'],
                    'additions':f['additions'],'deletions':f['deletions'],'file_kind':kind,
                    'pr_file_list_complete':m['files_complete'],'pr_file_lines_reconcile':m['file_line_sums_match'],**flags})
                file_count += 1
                if len(file_buffer)>=50000:
                    file_writer.write_table(pa.Table.from_pylist(file_buffer,schema=file_schema));file_buffer=[]
            m.update(stars_snapshot=repo['stars'], cohort_rank=repo['rank'], language=repo['language'],
                repository_created_at=repo['created_at'], author_login=(p.get('author') or {}).get('login'),
                author_type=(p.get('author') or {}).get('__typename'), title=p['title'],
                changed_lines=p['additions']+p['deletions'], cross_file=p['changedFiles']>=2,
                files_listed=len(fs), listed_directories=len({str(PurePosixPath(f['path']).parent) for f in fs}) if m['files_complete'] else None,
                source_lines_eligible=m['files_complete'] and m['file_line_sums_match'],
                source_churn=m['source_additions']+m['source_deletions'] if m['source_additions'] is not None else None,
                human_stage_evidence_available=False, acquisition_enumeration_complete=status['enumeration_complete'],
                repository_excluded_prs=2 if name=='yt-dlp/yt-dlp' else 0)
            for stage in stage_flags('',''):
                m['touches_'+stage] = bool(phases[stage]) if m['files_complete'] else None
            rows.append(m)
            # Independently aggregate original scalar fields and re-evaluate signal
            # predicates without using metrics() or summary_rows().
            branch=(p.get('headRefName') or '').casefold();actor=((p.get('author') or {}).get('login') or '').casefold()
            signal=branch.startswith(('codex/','copilot/')) or bool(re.search(r'Co-Authored-By:\s*Claude\b|Generated with Claude Code',p.get('body') or '',re.I)) or actor in {'google-labs-jules[bot]','google-labs-jules','devin-ai-integration[bot]','devin-ai-integration'}
            assert bool(signal)==m['agent_detected']
            for c,stamp in [('created',p['createdAt']),('merged',p['mergedAt'])]:
                if stamp and START<=stamp<END:
                    raw_stats[c].update(n=1,agent_n=int(signal),churn=p['additions']+p['deletions'],agent_churn=(p['additions']+p['deletions'])*int(signal))
        sub=pd.DataFrame(rows[start_index:]);count=len(sub)
        assert count==status['unique_prs']
        assert count+(2 if name=='yt-dlp/yt-dlp' else 0)==status['expected_prs']
        for c in ['created','merged']:
            if count:
                cohort=sub[sub[c+'_in_window']]
                observed=[len(cohort),int(cohort.agent_detected.sum()),int(cohort.changed_lines.sum()),int(cohort.loc[cohort.agent_detected,'changed_lines'].sum())]
            else:
                observed=[0,0,0,0]
            s=raw_stats[c]
            assert [s['n'],s['agent_n'],s['churn'],s['agent_churn']]==observed
        digest=sha256(dest)
        if raw.suffix=='.gz':
            assert digest==archive_index[name]['uncompressed_sha256']
        else:
            assert digest==sha256(raw)
        coverage.append({'repository':name,'rank':repo['rank'],'stars_snapshot':repo['stars'],'language':repo['language'],
            'repository_created_at':repo['created_at'],'expected_prs':status['expected_prs'],'included_prs':count,
            'excluded_prs':2 if name=='yt-dlp/yt-dlp' else 0,'acquisition_enumeration_complete':status['enumeration_complete'],
            'included_coverage_pct':percentage(count,status['expected_prs']),
            'created_in_window':status['created_in_window'],'merged_in_window':status['merged_in_window'],
            'merged_source_eligible':status['merged_source_lines_eligible'],
            'raw_relative_path':str(dest.relative_to(target)),'raw_sha256':digest})
        validation.append({'repository':name,'raw_count':count,'independent_scalar_reconciliation':True,'raw_sha256':digest,'raw_stats':raw_stats})
        print(json.dumps({'repository':name,'included':count,'file_rows_so_far':file_count}),flush=True)
    if file_buffer:file_writer.write_table(pa.Table.from_pylist(file_buffer,schema=file_schema))
    file_writer.close()
    df=pd.DataFrame(rows)
    for col in ['source_additions','source_deletions','source_churn','listed_directories']:
        df[col]=pd.to_numeric(df[col]).astype('Int64')
    for col in ['touches_'+s for s in stage_flags('','')]:df[col]=df[col].astype('boolean')
    df.to_parquet(target/'features/prs.parquet',index=False,compression='zstd')
    df.to_csv(target/'features/prs.csv.gz',index=False,compression='gzip',quoting=csv.QUOTE_ALL)
    cov=pd.DataFrame(coverage);cov.to_csv(target/'metadata/repositories.csv',index=False)
    assert len(df)==190094 and len(seen)==len(df) and cov.excluded_prs.sum()==2
    assert pq.ParquetFile(target/'features/files.parquet').metadata.num_rows==file_count
    write_json(target/'metadata/validation.json',{'at':now(),'passed':True,'unique_prs':len(df),'file_rows':file_count,
        'scope':'All included raw PRs; identity/date/scalar checks, independent raw numerator/denominator reconciliation, copy hashes. Signal validity not manually adjudicated.',
        'repositories':validation})
    manifest={'dataset_id':target.name,'version':'1.0.0','built_at':now(),'repositories':10,'expected_prs':190096,
        'included_prs':len(df),'excluded_prs':2,'listed_file_rows':file_count,'observation_start':START,'observation_end_exclusive':END,
        'sampling':'No further sampling; census of selected repositories within the observation window, minus two user-authorized detail exclusions.',
        'source_directory':'raw/','signal_rule_version':'visible_five_agent_signals_v1','path_rule_version':'analyze_github_delivery.file_kind',
        'cohort_note':'10 selected high-star technical repositories from a frozen snapshot; not an exhaustively verified global technical top-ten ranking.',
        'privacy_note':'Raw public GitHub author names and PR text retained from the source export; source collector redacts recognizable GitHub token patterns; no public redistribution performed.'}
    write_json(target/'metadata/manifest.json',manifest)
    analyze(df,cov,target,manifest)


def analyze(df: pd.DataFrame, cov: pd.DataFrame, target: Path, manifest: dict) -> None:
    tables=target/'analysis'
    contrib=[];comp=[];phase=[];agent_rows=[];snapshots=[]
    cohorts={'created':df[df.created_in_window],'merged':df[df.merged_in_window]}
    for cohort,base in cohorts.items():
        datecol=cohort+'_quarter'
        for name in list(cov.repository)+['ALL_SELECTED']:
            g=base if name=='ALL_SELECTED' else base[base.repository==name]
            for q in QUARTERS+['ALL']:
                period=g if q=='ALL' else g[g[datecol]==q]
                for task in ['all','fix_title_candidate']:
                    z=period if task=='all' else period[period.fix_title_candidate]
                    contrib.append({'repository':name,'cohort':cohort,'quarter':q,'task':task,**contribution(z)})
            for signal,z in [('all',g),('agent_signal',g[g.agent_detected]),('no_detected_signal',g[~g.agent_detected])]:
                comp.append({'repository':name,'cohort':cohort,'signal_group':signal,**complexity(z)})
                eligible=z[z.files_complete]
                for stage in stage_flags('',''):
                    n=int(eligible['touches_'+stage].fillna(False).sum())
                    phase.append({'repository':name,'cohort':cohort,'signal_group':signal,'path_signal':stage,
                        'total_prs':len(z),'file_complete_prs':len(eligible),'eligible_pct':percentage(len(eligible),len(z)),
                        'touching_prs':n,'touching_pct_of_eligible':percentage(n,len(eligible)),
                        'human_participation_pct':None,'interpretation':'File-path change evidence only; not execution or human-stage participation.'})
            for agent in AGENTS:
                for q in QUARTERS+['ALL']:
                    period=g if q=='ALL' else g[g[datecol]==q]
                    z=period.loc[period.agents.str.split('|').map(lambda x:agent in x).astype(bool)]
                    agent_rows.append({'repository':name,'cohort':cohort,'quarter':q,'agent':agent,'agent_pr_count':len(z),
                        'all_pr_count':len(period),'agent_pr_pct':percentage(len(z),len(period)),
                        'text_churn':int(z.changed_lines.sum()),'overlap_allowed':True})
        if cohort=='created':
            for name,g in base.groupby('repository'):
                for signal,z in [('agent_signal',g[g.agent_detected]),('no_detected_signal',g[~g.agent_detected])]:
                    snapshots.append({'repository':name,'signal_group':signal,'created_prs':len(z),
                        'merged_by_capture':int((z.state=='MERGED').sum()),'open_by_capture':int((z.state=='OPEN').sum()),
                        'closed_unmerged_by_capture':int((z.state=='CLOSED').sum()),
                        'merged_by_capture_pct':percentage((z.state=='MERGED').sum(),len(z))})
    c=pd.DataFrame(contrib);cx=pd.DataFrame(comp);ph=pd.DataFrame(phase);ag=pd.DataFrame(agent_rows)
    c.to_csv(tables/'contributions_repo_quarter_task.csv',index=False)
    cx.to_csv(tables/'cross_file_and_lines.csv',index=False);ph.to_csv(tables/'engineering_path_footprints.csv',index=False)
    ag.to_csv(tables/'agent_signal_breakdown.csv',index=False)
    pd.DataFrame(snapshots).to_csv(tables/'created_cohort_status_at_capture.csv',index=False)
    fixed_names=set(cov.loc[cov.repository_created_at<START,'repository'])
    sensitivity=[]
    for scope,names in [('all_selected',set(cov.repository)),('exclude_openclaw_vscode',set(cov.repository)-{'openclaw/openclaw','microsoft/vscode'}),('existed_before_2025',fixed_names)]:
        for cohort,base in cohorts.items():
            for q in QUARTERS+['ALL']:
                z=base[base.repository.isin(names)]
                if q!='ALL':z=z[z[cohort+'_quarter']==q]
                sensitivity.append({'scope':scope,'cohort':cohort,'quarter':q,**contribution(z)})
    sens=pd.DataFrame(sensitivity);sens.to_csv(tables/'composition_sensitivity.csv',index=False)
    # Validate additive quarters and numerator containment for every view.
    for key,g in c.groupby(['repository','cohort','task']):
        allrow=g[g.quarter=='ALL'].iloc[0];qs=g[g.quarter!='ALL']
        for col in ['pr_count','agent_pr_count','all_text_churn','agent_text_churn']:
            assert int(qs[col].sum())==int(allrow[col]),(key,col)
    assert (c.agent_pr_count<=c.pr_count).all()
    ratio_cols=[col for col in c if col.endswith('_pct')]
    for col in ratio_cols:assert c[col].dropna().between(0,100).all(),col
    bounds=c.dropna(subset=['source_share_lower_bound_pct','source_share_upper_bound_pct'])
    assert (bounds.source_share_lower_bound_pct<=bounds.source_share_upper_bound_pct+1e-9).all()
    make_figures(c,cov,sens,target)
    write_docs(df,c,cx,ph,ag,cov,sens,target,manifest)
    from add_highstar_monthly_analysis import add_monthly
    add_monthly(target,df,cov)


def make_figures(c,cov,sens,target):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
    names=list(cov.repository);y=np.arange(len(names));fig,ax=plt.subplots(figsize=(10,6))
    for cohort,offset,color,label in [('created',-.18,'#24689B','Created in window: all states'),('merged',.18,'#CA6836','Merged in window')]:
        vals=[]
        for n in names:
            r=c[(c.repository==n)&(c.cohort==cohort)&(c.quarter=='ALL')&(c.task=='all')].iloc[0];vals.append(r.agent_pr_pct)
        ax.barh(y+offset,np.nan_to_num(vals),height=.34,color=color,label=label)
        for i,v in enumerate(vals):ax.text((v if pd.notna(v) else 0)+.1,y[i]+offset,'NA' if pd.isna(v) else (f'{v:.3f}%' if 0<v<.1 else f'{v:.2f}%'),va='center',fontsize=8)
    ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,16);ax.set_xlabel('PRs with detectable Agent signal / included PRs (%)')
    ax.set_title('Selected high-star projects: two contribution cohorts');ax.legend(loc='lower right',fontsize=9)
    fig.text(.01,.01,'2025-01 to 2026-06 UTC. yt-dlp excludes 2 inaccessible PRs. NA = no denominator. No signal does not mean human-only.',fontsize=8)
    fig.tight_layout(rect=(0,.03,1,1));fig.savefig(target/'figures/pr_share_by_repository.png');fig.savefig(target/'figures/pr_share_by_repository.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
    for ax,cohort in zip(axs,['created','merged']):
        for task,color,style in [('all','#24689B','-'),('fix_title_candidate','#CA6836','--')]:
            z=c[(c.repository=='ALL_SELECTED')&(c.cohort==cohort)&(c.task==task)&(c.quarter!='ALL')].set_index('quarter').reindex(QUARTERS)
            ax.plot(QUARTERS,z.agent_pr_pct,style+'o',color=color,label=task.replace('_',' '))
        fixed=sens[(sens.scope=='existed_before_2025')&(sens.cohort==cohort)&(sens.quarter!='ALL')].set_index('quarter').reindex(QUARTERS)
        ax.plot(QUARTERS,fixed.agent_pr_pct,':s',color='#48815E',label='All tasks, pre-2025 repositories')
        ax.set_title(cohort.capitalize()+' cohort');ax.set_ylim(bottom=0);ax.tick_params(axis='x',rotation=35);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    axs[0].set_ylabel('Detectable Agent PR share (%)');fig.suptitle('Quarterly participation: composition and repair-title candidates')
    fig.text(.01,.01,'Pooled ratios use summed counts, not averages of percentages. Repair titles are unvalidated candidates. No causal inference.',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.96));fig.savefig(target/'figures/quarterly_participation.png');fig.savefig(target/'figures/quarterly_participation.pdf');plt.close(fig)


def write_docs(df,c,cx,ph,ag,cov,sens,target,manifest):
    def get(repo='ALL_SELECTED',cohort='created',task='all',q='ALL'):
        return c[(c.repository==repo)&(c.cohort==cohort)&(c.task==task)&(c.quarter==q)].iloc[0]
    created=get();merged=get(cohort='merged');fixc=get(task='fix_title_candidate');fixm=get(cohort='merged',task='fix_title_candidate')
    share_two=100*sum(get(n).agent_pr_count for n in ['openclaw/openclaw','microsoft/vscode'])/created.agent_pr_count
    overview=[]
    for n in cov.repository:
        a=get(n);b=get(n,'merged')
        overview.append([n,f'{int(a.pr_count):,}',f'{int(a.agent_pr_count):,}',fmt(a.agent_pr_pct),f'{int(b.pr_count):,}',f'{int(b.agent_pr_count):,}',fmt(b.agent_pr_pct),fmt(b.agent_text_churn_pct)])
    report=[
        '# 高星技术项目 Agent 贡献分析', '',
        f'生成时间：{manifest["built_at"]}。观察期：2025-01-01 至 2026-07-01（右端不含），UTC。数据集：`{target.name}`。', '',
        f'10个选定仓库，接口预期190,096条，纳入 **{len(df):,}条唯一PR**；按用户要求排除yt-dlp的2条不可访问详情。未再次抽样。9个仓库完整枚举，yt-dlp为排除2条后的可分析集合；不能把10个都称为无缺失全量。DeepSeek Harness创建晚于观察期，记录为0条；Vue的92条PR均无期内合并记录。', '',
        f'期内创建的全部状态PR为 **{int(created.pr_count):,}条**，其中可识别Agent参与 **{int(created.agent_pr_count):,}条（{fmt(created.agent_pr_pct)}）**；期内合并PR为 **{int(merged.pr_count):,}条**，Agent参与 **{int(merged.agent_pr_count):,}条（{fmt(merged.agent_pr_pct)}）**。两组按不同日期入组，不能用第二组/第一组直接计算合并率。', '',
        '## 1. Agent贡献的PR数量与行数', '',
        md_table(['仓库','创建PR','Agent创建PR','创建占比','期内合并PR','Agent合并PR','合并占比','合并全文件增删行占比'],overview), '',
        f'全部创建PR中，Agent参与PR的全文件增删行占比为 **{fmt(created.agent_text_churn_pct)}**；期内合并口径为 **{fmt(merged.agent_text_churn_pct)}**。这是PR变更量之和，重复修改会重复计数，包含测试、文档、配置等；不是仓库代码存量比例，也不是Agent独立创作行数。', '',
        f'OpenClaw和VS Code合计占可识别Agent创建PR的 **{share_two:.2f}%**，说明总体结果明显受项目构成影响。未检测到信号不等于纯人工。项目自身是Agent产品，也不表示该项目每个PR都由Agent生成。', '',
        '源码候选按路径规则排除测试、文档、配置、锁文件及生成/第三方目录；完整文件列表且行数对账一致才给点估计。缺失时给确定性上下界，未知PR的源码增删行可在0至该PR总增删行之间变化。该区间不是置信区间。', '',
        md_table(['仓库','合并PR源码明细覆盖','源码增删行占比','缺失敏感性下界—上界'],[
            [n,f'{int(get(n,"merged").source_eligible_prs):,}/{int(get(n,"merged").pr_count):,}',fmt(get(n,'merged').agent_source_churn_pct),
             fmt(get(n,'merged').source_share_lower_bound_pct)+' — '+fmt(get(n,'merged').source_share_upper_bound_pct)] for n in cov.repository]), '',
        'OpenClaw和VS Code各1条合并PR明细不完整；yt-dlp有34条已纳入合并PR的文件明细尚不完整。yt-dlp的区间只覆盖排除2条后的已纳入集合，不对被排除记录作假设。全部创建PR的文件明细未全量补齐，因此源码占比不能直接以缺失当零。', '',
        '## 2. 跨文件与代码变更规模', '',
    ]
    comp_rows=[]
    for cohort in ['created','merged']:
        for signal in ['agent_signal','no_detected_signal']:
            r=cx[(cx.repository=='ALL_SELECTED')&(cx.cohort==cohort)&(cx.signal_group==signal)].iloc[0]
            comp_rows.append([cohort,signal,f'{int(r.pr_count):,}',fmt(r.cross_file_pct),f'{r.changed_files_median:.1f}',f'{r.changed_files_p90:.1f}',f'{r.changed_lines_median:.1f}',f'{r.changed_lines_p90:.1f}'])
    report += [md_table(['入组口径','信号组','PR数','跨文件率（≥2）','文件数中位数','文件数P90','增删行中位数','增删行P90'],comp_rows), '',
        '文件数和增删行来自PR级完整统计，不依赖文件列表是否截断；目录数、源码文件与阶段路径指标依赖文件列表完整性。上述对比未控制项目、任务、维护者及PR规模，不能解释为Agent的因果影响。逐仓库数值见 `analysis/cross_file_and_lines.csv`。', '',
        '## 3. 按季度下钻代码修复候选（并入仓库贡献问题）', '',
        '以标题中的fix/fixes/fixed/fixing、bug/bugs、regression/regressions、hotfix等词形识别修复候选，不等同于人工验证的修复任务。分母是在同一仓库、同一季度、同一入组口径下的全部修复候选；分子为其中可识别Agent参与的PR。', '',
        md_table(['季度','全部创建PR','Agent创建占比','修复候选创建PR','修复候选Agent占比','全部合并PR','Agent合并占比','修复候选Agent合并占比'],[
            [q,f'{int(get(q=q).pr_count):,}',fmt(get(q=q).agent_pr_pct),f'{int(get(task="fix_title_candidate",q=q).pr_count):,}',fmt(get(task='fix_title_candidate',q=q).agent_pr_pct),
             f'{int(get(cohort="merged",q=q).pr_count):,}',fmt(get(cohort='merged',q=q).agent_pr_pct),fmt(get(cohort='merged',task='fix_title_candidate',q=q).agent_pr_pct)] for q in QUARTERS]), '',
        f'全期修复候选：创建口径{int(fixc.pr_count):,}条，Agent参与占比{fmt(fixc.agent_pr_pct)}；合并口径{int(fixm.pr_count):,}条，Agent参与占比{fmt(fixm.agent_pr_pct)}。', '',
        f'项目趋势不一致：OpenClaw创建口径从2026Q1的{fmt(get("openclaw/openclaw",q="2026Q1").agent_pr_pct)}升至Q2的{fmt(get("openclaw/openclaw",q="2026Q2").agent_pr_pct)}；VS Code同期从{fmt(get("microsoft/vscode",q="2026Q1").agent_pr_pct)}降至{fmt(get("microsoft/vscode",q="2026Q2").agent_pr_pct)}。因此不能用总体曲线宣称各项目Agent参与持续上升。', '',
        '对照表 `analysis/composition_sensitivity.csv` 同时给出剔除OpenClaw/VS Code、仅保留2025年前已创建仓库的结果。后者用于检查新项目进入造成的构成变化，不是随机对照或因果估计。', '',
        '![各项目Agent PR占比](figures/pr_share_by_repository.png)', '',
        '图1：并列呈现创建和合并口径，显示项目间差异；0表示未检测到信号，NA表示没有分母。', '',
        '![季度参与趋势](figures/quarterly_participation.png)', '',
        '图2：全任务、修复候选与2025年前已创建仓库的汇总趋势。各曲线均由分子分母分别求和计算，不能当作同一批PR的转化曲线。', '',
        '## 4. 软件工程阶段与人工参与：可测量范围', '',
        '现有数据没有需求访谈、设计决策、人类/Agent逐步操作、完整评审事件或CI执行主体，因此**不能给出需求获取、分析、设计、生成、集成、测试、部署各阶段的人工参与比例或工时占比**。User类型账号也可能运行Agent，不能直接等同于人类执行。', '',
        md_table(['阶段','当前可观察字段','能否计算人工参与比例'],[
            ['需求获取、分析、设计','标题和正文可作后续标注材料，当前无已验证阶段标签','不能'],
            ['代码生成/修改','源码路径变更、文件数和增删行','不能；只测交付物变更'],
            ['集成','合并状态、CI配置路径变更候选','不能；合并和配置变更不等于人类集成操作'],
            ['测试','测试路径变更候选','不能；不表示测试实际执行或执行主体'],
            ['部署','Docker/Kubernetes/Helm/Terraform等路径变更候选','不能；不表示发生部署'],
            ['文档','文档路径变更','不能；不等同于需求获取或设计活动']]), '',
        '路径变化表 `analysis/engineering_path_footprints.csv` 仅在文件列表完整的PR中计算，展示有效分母和覆盖率，多种路径可重叠；不作阶段人工比例的替代。完整评审和事件数据未在本批次采集，不能将其缺失解释为无人参与。', '',
        '## 5. 与灵犀平台对照和后续工作', '',
        '平台侧应对齐：仓库/项目、PR或交付单元ID、创建与合并时间（UTC）、季度、Agent类别、任务标签、变更文件数、新增/删除行，以及逐阶段参与者类型和事件。先对照可见Agent参与PR比例、修复候选的季度参与度、跨文件率和增删行分布；阶段人工比例需要平台提供真实事件分母。', '',
        '低星对照尚未采集，因此本报告不支持“高星比低星更高/更低”结论。10个项目来自已确认技术候选及星数快照；当前星数不代表历史星数，不声称已穷尽验证全球所有技术项目排名。Vue对应vuejs/vue而非其他Vue仓库。', '',
        '本报告为选定仓库可分析集合的描述性统计，不进行二次抽样、bootstrap、显著性检验或全GitHub总体外推。优先下一步是用独立标签验证Agent识别和修复候选，再按一致观察窗补齐低星对照及阶段事件。', '',
        '复现入口、所有文件说明和字段字典见 [README](README.md)、[数据字典](DATA_DICTIONARY.md)；校验记录见 `metadata/validation.json`。原始采集目录、Desktop/github原始数据和论文未修改。']
    phase_rows=[]
    for stage in stage_flags('',''):
        values=[]
        for signal in ['agent_signal','no_detected_signal']:
            r=ph[(ph.repository=='ALL_SELECTED')&(ph.cohort=='created')&(ph.signal_group==signal)&(ph.path_signal==stage)].iloc[0]
            values.extend([f'{int(r.touching_prs):,}/{int(r.file_complete_prs):,}',fmt(r.touching_pct_of_eligible)])
        phase_rows.append([stage]+values)
    at=report.index('## 5. 与灵犀平台对照和后续工作')
    report[at:at]=['创建队列的工程路径变更分布如下；分母仅为文件列表完整的PR，不能当作全部PR或人类参与分母。','',
        md_table(['路径信号','Agent参与PR：命中/可核验','占比','未识别信号PR：命中/可核验','占比'],phase_rows),'']
    (target/'高星项目分析报告.md').write_text('\n'.join(report)+'\n')
    weekly=(f'# AgentBench 高星项目周报\n\n已完成10个选定高星技术项目的数据整理，固定2025年1月至2026年6月观察期。接口预期190,096条PR，按约定排除yt-dlp两条不可访问详情，纳入190,094条唯一PR，不再次抽样；9个仓库完整枚举，yt-dlp保留排除记录。\n\n'
        f'期内创建PR {int(created.pr_count):,}条，可识别Agent参与{int(created.agent_pr_count):,}条，占{fmt(created.agent_pr_pct)}；期内合并PR {int(merged.pr_count):,}条，Agent参与{int(merged.agent_pr_count):,}条，占{fmt(merged.agent_pr_pct)}。合并PR口径下，Agent参与PR的全文件增删行占比为{fmt(merged.agent_text_churn_pct)}，不等同于Agent独立生成源码占比。\n\n'
        f'完成仓库×季度×任务候选分析：修复候选创建PR {int(fixc.pr_count):,}条，Agent参与占比{fmt(fixc.agent_pr_pct)}；项目变化方向不同，OpenClaw与VS Code在2026Q1至Q2分别呈上升和下降。总体可识别Agent创建PR的{share_two:.2f}%来自这两个项目，需关注项目构成影响。\n\n'
        '新增独立命名数据集、原始PR副本、PR级与文件级Parquet、字段字典、完整性与排除台账、跨文件/行数统计、季度修复候选表和两张图。源码比例仅在文件明细完整时给点估计，否则保留缺失并给确定性上下界。\n\n'
        '目前尚不能回答各软件工程阶段的人工参与比例：缺少阶段级参与者和操作记录；路径变更只作为交付物信号。低星对照尚未完成。与灵犀平台对齐时，先统一PR/交付单元、日期、Agent与任务标签、文件数和增删行定义，再补人机阶段事件。\n')
    (target/'周报_高星项目_20260924.md').write_text(weekly)
    (target/'README.md').write_text(f'''# {target.name}

独立的高星技术项目PR数据集，版本1.0.0。范围为10个用户确认项目、2025-01-01至2026-07-01（右端不含，UTC）。包含期内创建的全部状态PR，以及更早创建但期内合并的PR。190,096条预期，190,094条详情，2条经用户明确要求排除；不再抽样。PR按ID唯一，文件表是一条PR中的一个已取回文件，跨PR的同名路径不是重复错误。

## 文件与用途

| 路径 | 粒度/格式 | 内容与使用注意 |
|---|---|---|
| raw/<owner>__<repo>/prs.jsonl | 每行一个PR，UTF-8 | 原始采集副本；含正文、标题、作者、分支、标签、时间、统计量、文件连接及分页状态。用文件逐行解析，不用str.splitlines，正文可含Unicode分隔符。 |
| raw/*/acquisition_status.json | 每仓库一个JSON | 保留采集时状态；yt-dlp仍为partial，独立数据集的排除政策不篡改采集事实。 |
| features/prs.parquet | {len(df):,}条PR | 建议分析入口；空值、布尔和数值类型保留。 |
| features/prs.csv.gz | 同上，gzip CSV | 方便其他工具导入；空单元格表示未知，不是0。 |
| features/files.parquet | {manifest['listed_file_rows']:,}条已取回文件 | 包含PR外键、路径、增删行、路径分类与完整性标记；不是全体PR的完整文件全集。 |
| metadata/repositories.csv | 10个仓库 | 排名/星数快照、语言、创建时间、预期/纳入/排除数、明细覆盖及原始文件SHA256。 |
| metadata/exclusions.json | 2条排除记录 | yt-dlp #15386、#15481的ID、URL和错误证据，不填造详情。 |
| metadata/manifest.json | 数据集级 | 版本、观察窗、规则版本、来源、样本政策。 |
| metadata/validation.json | 仓库级校验 | 遍历全部已纳入PR验证唯一性、日期、非负统计及独立分子分母对账；不是标签准确率验证。 |
| DATA_PROFILE.md、DATA_DICTIONARY.md | 中文数据说明 | 数据特征、状态分布、缺失情况及逐字段定义。 |
| metadata/selection_protocol.json 等 | 来源快照 | 保留原选择、采集、结构验证和排名证据；若与新排除政策不同，以manifest和exclusions说明分析口径。 |
| analysis/contributions_repo_quarter_task.csv | 仓库×口径×季度×任务 | PR比例、全文件新增/删除/增删行、源码覆盖及上下界；ALL_SELECTED为汇总，不能再与各仓库行相加。 |
| analysis/cross_file_and_lines.csv | 仓库×口径×信号组 | 跨文件率、文件数/行数/目录数的中位数、P90和均值。 |
| analysis/engineering_path_footprints.csv | 仓库×口径×信号组×路径信号 | 完整文件列表子集内的源码、测试、CI、部署、文档变更候选；人工比例为空。 |
| analysis/agent_signal_breakdown.csv | 仓库×口径×季度×Agent | 五类Agent可见信号，多个Agent可重叠，不能相加当总体。 |
| analysis/created_cohort_status_at_capture.csv | 仓库×信号组 | 对同一创建队列统计采集时合并/开放/关闭未合并；不是期内合并率，不是最终成功率。 |
| analysis/composition_sensitivity.csv | 项目集合×口径×季度 | 总体、剔除OpenClaw/VS Code、2025年前已存在项目的对照。 |
| figures/*.png、*.pdf | 静态图 | 项目PR占比、季度任务趋势；图注解释分母。 |
| 高星项目分析报告.md、周报_高星项目_20260924.md | 中文报告 | 四个问题的可答部分、结果与证据缺口。 |
| code/*.py | 生成代码 | 支持原采集目录或本独立数据集作为输入；保留特征与分类规则。 |

## 使用示例

```python
import pandas as pd
prs = pd.read_parquet('features/prs.parquet')
submitted = prs[prs.created_in_window]  # 全部状态，按created_quarter分季度
merged = prs[prs.merged_in_window]      # 按merged_quarter分季度
files = pd.read_parquet('features/files.parquet')
```

## 复现

在原工作目录运行：
```bash
python3 scripts/build_highstar_dataset.py --source outputs/github_technical_prs_20260924 --output datasets/{target.name}
```
也可仅使用本数据集重建到另一个目录（不能把输入和输出设成同一路径）：
```bash
python3 code/build_highstar_dataset.py --source . --output ../{target.name}_rebuilt
```
依赖Python、pandas、numpy、pyarrow、matplotlib、requests。脚本离线运行，不访问网络，不改源数据；重复运行覆盖输出目录内的同名产物。原始副本包含公开GitHub作者与正文，不代表已经审查再发布许可；本轮只在本地保存。

没有信号不等于人工；User作者类型不等于人类操作；修复标题标签未验证；缺少逐阶段交互/评审/CI运行事件；低星对照未采集。DeepSeek Harness晚于观察期创建，0条；Vue 92条创建PR、0条期内合并，合并占比为NA。
''')
    descriptions={
        'id':'GitHub PR节点ID，主键','repository':'owner/repository；文件表通过pr_id关联PR','number':'仓库内PR编号','url':'PR公开链接','created_at':'PR创建时间UTC','merged_at':'合并时间UTC，未合并为空',
        'created_quarter':'创建季度','merged_quarter':'合并季度，未合并为空','created_in_window':'创建日期是否在观察窗内','merged_in_window':'合并日期是否在观察窗内',
        'state':'采集时OPEN/CLOSED/MERGED；CLOSED为未合并关闭','agent_detected':'是否命中五类可见Agent规则；False不代表人工','agents':'命中类别，以|连接，可重叠','evidence':'JSON字符串形式的规则命中依据',
        'fix_title_candidate':'标题修复词形候选，不是人工标签','bug_label_candidate':'标签标准化后的bug候选；与标题候选分开',
        'changed_files':'API提供的PR完整变更文件数','additions':'全部文件新增行','deletions':'全部文件删除行','files_complete':'文件分页结束且路径唯一/数量与API一致','file_line_sums_match':'已取文件新增删除之和与PR统计分别一致',
        'source_additions':'仅完整且行数对账通过时的源码候选新增行，否则空','source_deletions':'同上，删除行','stars_snapshot':'选择时星数快照；不是历史季度星数','cohort_rank':'当前选定名单顺序，不是GitHub全局排名',
        'language':'选择时仓库主语言','repository_created_at':'仓库创建时间UTC','author_login':'PR作者账号，可能为空','author_type':'GitHub账号类型User/Bot等，不推断人类操作','title':'原始PR标题',
        'changed_lines':'additions+deletions，全文件修改量，不是净增量','cross_file':'changed_files≥2','files_listed':'当前已取回的文件条目数','listed_directories':'文件列表完整时不同父目录数；根目录为.；否则空，不等同模块数',
        'source_lines_eligible':'files_complete且file_line_sums_match','source_churn':'source_additions+source_deletions；不合格为空',
        'human_stage_evidence_available':'本数据集均False，未采逐阶段人机事件','acquisition_enumeration_complete':'原始采集完整枚举标志；yt-dlp为False','repository_excluded_prs':'该仓库明确排除的PR数量，yt-dlp为2，不能逐行求和',
    }
    for stage in stage_flags('',''):descriptions['touches_'+stage]='完整文件列表内是否命中'+stage+'路径规则；不完整为空，不代表对应活动已执行'
    assert set(df.columns)==set(descriptions),(set(df.columns)-set(descriptions))
    dictionary=['# 数据字段与指标字典','','## PR级表：features/prs.parquet','',md_table(['字段','类型','含义'],[[k,str(df[k].dtype),descriptions[k]] for k in df.columns]),'',
        '## 文件级表：features/files.parquet','',
        md_table(['字段','含义'],[['pr_id','PR节点ID外键'],['repository','仓库名'],['path','仓库内相对路径，不指向本机文件'],['additions / deletions','该文件新增/删除行'],['file_kind','lockfile / generated_or_vendor_path / test_path / documentation_path / source_candidate / configuration_path / other_or_unknown'],['pr_file_list_complete / pr_file_lines_reconcile','所属PR的列表完整性与行数对账'],['code_path / test_path / ci_path / deployment_path / documentation_path','规则派生布尔值，可重叠；部署与CI为路径候选']]),'',
        '## 分析口径与公式','',
        '- created：期内创建且有可分析详情的PR，任意状态；merged：期内合并且有可分析详情的PR。两个队列重叠但不相同，不能相加。',
        '- agent_pr_pct = 100 × agent_pr_count / pr_count；任意Agent汇总按PR去重。',
        '- agent_text_churn_pct = 100 × Agent参与PR的(additions+deletions)之和 / 全部纳入PR的对应之和。',
        '- source_coverage_pct = 100 × 源码行数合格PR数 / 全部纳入PR数；不完整时agent_source_churn_pct为空。',
        '- 缺失源码上下界：已知源码分母D、Agent源码A，未知Agent PR总增删行Ua、未知其他PR总增删行Uo；下界A/(D+Uo)，上界(A+Ua)/(D+Ua)，乘100。仅为路径缺失的确定性敏感性区间，不覆盖Agent误识别或被排除PR；零分母为空。',
        '- fix_title_candidate使用英文标题词形规则，中文标题、隐式修复、标签和语义未完整覆盖；不是功能正确性标签。',
        '- ALL为观察期汇总；ALL_SELECTED为10仓库纳入集合汇总。不要把汇总行和明细行一起再聚合。',
        '- no_detected_signal表示未识别可见Agent信号，不标成human。所有时间趋势均是快照回看历史日期，PR文本/分支可能在之后修改，不能声称标签在当时已经可见。',
        '- 比例0表示有分母但分子0；NA/空表示无分母或证据不足。均值/中位数/P90只对非空字段计算，并提供eligible_n。', '',
        '## Agent信号规则','',
        'Codex：headRefName以codex/开头；Copilot：copilot/开头；Claude Code：正文含Generated with Claude Code或Co-Authored-By: Claude；Jules：google-labs-jules或其[bot]账号；Devin：devin-ai-integration或其[bot]账号。规则命中是参与信号，不是角色/工时/独立代码作者的真值。未估计精确率或召回率。']
    (target/'DATA_DICTIONARY.md').write_text('\n'.join(dictionary)+'\n')
    state_rows=[]
    for name in cov.repository:
        g=df[df.repository==name];r=cov[cov.repository==name].iloc[0]
        state_rows.append([name,f'{len(g):,}',int((g.state=='OPEN').sum()),int((g.state=='CLOSED').sum()),int((g.state=='MERGED').sum()),
            f'{int(g.files_complete.sum()):,}',f'{int(r.merged_source_eligible):,}/{int(r.merged_in_window):,}'])
    agent_table=ag[(ag.repository=='ALL_SELECTED')&(ag.cohort=='created')&(ag.quarter=='ALL')]
    profile=['# 数据特征与覆盖情况','',
        f'独立数据集含10个选定仓库、{len(df):,}条唯一PR、{manifest["listed_file_rows"]:,}条已获取文件记录，PR特征表有{len(df.columns)}列。原始PR详情为JSONL，分析表为Parquet及gzip CSV。2条PR详情在显式排除台账中。', '',
        '状态是2026-09-24采集快照；下表“已合并”可包含观察期后才合并的PR，因此不同于按mergedAt筛选的“期内合并”。完整文件列表覆盖与完整PR枚举是两件事。','',
        md_table(['仓库','纳入PR','开放','关闭未合并','快照已合并','文件列表完整PR','期内合并源码合格/总数'],state_rows),'',
        '## Agent信号分布（期内创建队列）','',
        md_table(['类别','命中PR','占创建PR比例'],[[r.agent,int(r.agent_pr_count),fmt(r.agent_pr_pct)] for r in agent_table.itertuples()]),'',
        '类别可重叠，类别计数之和不保证等于去重Agent总数。规则只覆盖五类显式信号，缺乏手工金标准；不能据此宣称其他Agent不存在。','',
        '## 关键字段缺失','',
        md_table(['字段','空值数','解释'],[[k,int(df[k].isna().sum()),descriptions[k]] for k in ['author_login','author_type','merged_at','source_additions','source_deletions','listed_directories']]),'',
        '原始PR JSON字段含id/number/url、title/body、state、createdAt/updatedAt/closedAt/mergedAt、additions/deletions/changedFiles、headRefName/baseRefName、author、repository、labels、files。files带totalCount、pageInfo.hasNextPage/endCursor及nodes；未结束分页不能当作完整文件集合。labels仅请求前100项，完整原始totalCount保留。','',
        '本批次没有comments、reviews、reviewThreads、完整commit作者链、check-runs或deployment事件。需求/分析/设计的参与主体、人工工时、提示交互次数及阶段转移无法从这些字段恢复。标题与正文可供后续有验证的任务标注，当前仅使用修复标题规则候选。','',
        '划分粒度：PR为主键；文件以(pr_id,path)为通常关联键，原始路径不完整或异常重复时保留并通过质量标记识别；仓库表以repository关联。按创建/合并时间形成两个重叠队列，不是互斥分组。', '',
        '文件类型为保守路径代理：源码候选、测试、文档、配置、锁文件、生成/第三方路径、其他。大文件改动、依赖更新和批量生成文件会显著拉大均值，因此报告中位数及P90，并将全文件行数与源码候选行数分开。']
    (target/'DATA_PROFILE.md').write_text('\n'.join(profile)+'\n')
    summary={'created_prs':int(created.pr_count),'created_agent_prs':int(created.agent_pr_count),'created_agent_pr_pct':float(created.agent_pr_pct),
             'merged_prs':int(merged.pr_count),'merged_agent_prs':int(merged.agent_pr_count),'merged_agent_pr_pct':float(merged.agent_pr_pct),
             'merged_agent_text_churn_pct':float(merged.agent_text_churn_pct),'openclaw_vscode_share_of_agent_created_pct':share_two,
             'fix_created_prs':int(fixc.pr_count),'fix_created_agent_pct':float(fixc.agent_pr_pct)}
    write_json(target/'analysis/key_findings.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.source,args.output)
