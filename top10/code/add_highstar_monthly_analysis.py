#!/usr/bin/env python3
"""Add monthly, pooled Agent PR line-contribution trends to the local dataset."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from build_highstar_dataset import contribution, fmt, md_table, write_json
from collect_repo_contributions import AGENTS, now


def monthly_tables(df: pd.DataFrame, repositories: list[str], start: str, end: str):
    months=pd.period_range(start[:7],pd.Timestamp(end)-pd.Timedelta(days=1),freq='M').astype(str).tolist()
    rows=[];agent_rows=[]
    for cohort in ['created','merged']:
        base=df.loc[df[cohort+'_in_window']].copy()
        base['month']=base[cohort+'_at'].str[:7]
        for repository in repositories+['ALL_SELECTED']:
            repo_data=base if repository=='ALL_SELECTED' else base[base.repository==repository]
            for month in months:
                g=repo_data[repo_data.month==month]
                rows.append({'repository':repository,'cohort':cohort,'month':month,**contribution(g)})
                if repository=='ALL_SELECTED':
                    for agent in ['any']+AGENTS:
                        z=g.copy()
                        if agent!='any':
                            z['agent_detected']=z.agents.str.split('|').map(lambda x:agent in x).astype(bool)
                        agent_rows.append({'cohort':cohort,'month':month,'agent':agent,**contribution(z)})
    return pd.DataFrame(rows),pd.DataFrame(agent_rows)


def replace_section(path: Path, body: str, name: str) -> None:
    start=f'<!-- {name}:start -->';end=f'<!-- {name}:end -->'
    original=path.read_text() if path.exists() else ''
    if start in original:
        before,tail=original.split(start,1)
        _,after=tail.split(end,1)
        original=before+after
    path.write_text(original.rstrip()+'\n\n'+start+'\n'+body+'\n'+end+'\n')


def plot_monthly(monthly: pd.DataFrame, target: Path, suffix: str, since: str,
                 repository: str = 'ALL_SELECTED') -> None:
    m=monthly[(monthly.repository==repository)&(monthly.month>=since)]
    created=m[m.cohort=='created'].set_index('month').sort_index()
    merged=m[m.cohort=='merged'].set_index('month').sort_index()
    months=merged.index.to_list();x=np.arange(len(months))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    axes[0].plot(x,merged.agent_text_churn_pct,'o-',color='#24689B',label='Merged PRs: all-file additions + deletions')
    axes[0].plot(x,created.agent_text_churn_pct,'s--',color='#888888',label='Created PRs (all states): all-file additions + deletions')
    axes[0].set_ylabel('Agent-involved PR change share (%)');axes[0].legend(fontsize=9)
    axes[0].set_title('Monthly line contributions: '+
        ('selected high-star repositories' if repository=='ALL_SELECTED' else repository))
    exact=merged.agent_source_churn_pct.to_numpy(dtype=float)
    lower=merged.source_share_lower_bound_pct.to_numpy(dtype=float)
    upper=merged.source_share_upper_bound_pct.to_numpy(dtype=float)
    incomplete=~merged.source_coverage_complete.to_numpy(dtype=bool)
    axes[1].plot(x,exact,'o-',color='#39816C',label='Merged PRs: source candidates, complete coverage')
    axes[1].vlines(x[incomplete],lower[incomplete],upper[incomplete],color='#CB7038',linewidth=4,label='Incomplete source coverage: deterministic bounds')
    axes[1].scatter(x[incomplete],lower[incomplete],color='#CB7038',marker='_',s=90)
    axes[1].scatter(x[incomplete],upper[incomplete],color='#CB7038',marker='_',s=90)
    axes[1].set_ylabel('Agent-involved source change share (%)');axes[1].legend(fontsize=9)
    for ax in axes:
        ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
    axes[1].set_xticks(x,months,rotation=45,ha='right')
    fig.text(.02,.015,'Month uses createdAt or mergedAt in UTC. PR-level attribution is not independent Agent authorship. Two yt-dlp PRs excluded.\nSource intervals address missing file details only; they are not confidence intervals. No averaging of repository percentages.',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,1))
    for ext in ['png','pdf']:
        fig.savefig(target/'figures'/f'agent_monthly_code_share{suffix}.{ext}')
    plt.close(fig)


def add_repository_monthly(monthly: pd.DataFrame, target: Path, repositories: list[str]) -> dict:
    """Export within-repository ratios, preserving zero versus undefined months."""
    detail_dir=target/'analysis/monthly_by_repository'
    detail_dir.mkdir(exist_ok=True)
    matrix_files=[]
    for cohort in ['created','merged']:
        group=monthly[(monthly.repository!='ALL_SELECTED')&(monthly.cohort==cohort)]
        for metric in ['agent_text_churn_pct','agent_additions_pct']:
            wide=group.pivot(index='month',columns='repository',values=metric).reindex(columns=repositories)
            name=f'monthly_{cohort}_{metric}_by_repository.csv'
            wide.to_csv(target/'analysis'/name)
            matrix_files.append(name)
    months=sorted(monthly.loc[monthly.month>='2025-07','month'].unique())
    fig,axes=plt.subplots(5,2,figsize=(15,17),sharex=True,sharey=True)
    focus=monthly[(monthly.repository!='ALL_SELECTED')&(monthly.month>='2025-07')]
    ymax=max(5,float(focus.agent_text_churn_pct.max())*1.12)
    for ax,repo in zip(axes.flat,repositories):
        for cohort,color,style,label in [('merged','#24689B','o-','Merged PRs'),
                                          ('created','#888888','s--','Created PRs (all states)')]:
            values=focus[(focus.repository==repo)&(focus.cohort==cohort)].set_index('month').reindex(months)
            ax.plot(range(len(months)),values.agent_text_churn_pct,style,color=color,label=label,markersize=3)
        ax.set_title(repo,fontsize=11)
        ax.set_ylim(0,ymax);ax.grid(axis='y',alpha=.2)
        ax.set_xticks(range(len(months)),months,rotation=60,ha='right',fontsize=8)
        ax.tick_params(labelbottom=True)
        if focus[(focus.repository==repo)&(focus.cohort=='merged')].all_text_churn.sum()==0:
            ax.text(.04,.85,'No merged-PR denominator',transform=ax.transAxes,fontsize=9)
    fig.supylabel('Agent-involved PR additions + deletions / repository monthly total (%)')
    fig.suptitle('Monthly Agent contribution by repository | July 2025 - June 2026',fontsize=16)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.965),ncol=2,fontsize=10)
    fig.text(.03,.012,'Shared vertical scale. Gaps = undefined ratio (zero denominator); 0% = no detected Agent-attributed changes.\nPR-level visible attribution, not independently authored Agent code. Months use createdAt / mergedAt in UTC.',fontsize=10)
    fig.tight_layout(rect=(.02,.05,1,.94))
    for ext in ['png','pdf']:
        fig.savefig(target/'figures'/f'agent_monthly_code_share_by_repository_202507_202606.{ext}')
    plt.close(fig)
    lines=['# 各项目月度Agent代码贡献','',
        '按10个项目分别统计，完整月份为2025-01至2026-06；总览图聚焦2025-07至2026-06。每个项目均使用**该项目自身当月的PR变更量作为分母**，不是该项目占十仓库总量的份额。','',
        '主口径：当月合并的Agent参与PR新增行+删除行 ÷ 该项目当月全部合并PR新增行+删除行。另列仅新增行占比、全部状态PR按创建月份统计的对照，以及源码候选占比。计数均为全量纳入数据，不再抽样。','',
        '月份按UTC的合并/创建时间划分，不是代码实际编写时间。Agent参与依据既有五类可见信号识别，不能解释为Agent独立编写的行数；没有信号也不能判为纯人工。全文件行数含文档、配置和测试。','',
        '**NA表示分母为零、比例无定义；0%表示分母存在但未识别到Agent对应变更。** 源码明细不全时显示确定性上下界，非置信区间。yt-dlp仍按授权排除2条不可访问PR。','',
        '![各项目月度趋势](figures/agent_monthly_code_share_by_repository_202507_202606.png)','',
        '总览图使用统一纵轴。各项目独立图采用各自纵轴范围，阅读时请看刻度；下方月表保留较小的非零占比。CSV保留未四舍五入的百分数及全部分子分母。','']
    for repo in repositories:
        slug=repo.replace('/','__')
        data=monthly[monthly.repository==repo].copy()
        data.to_csv(detail_dir/f'{slug}.csv',index=False)
        m=data[data.cohort=='merged'].set_index('month').sort_index()
        c=data[data.cohort=='created'].set_index('month').sort_index()
        plot_monthly(monthly,target,f'_{slug}','2025-01',repo)
        valid=m.agent_text_churn_pct.dropna()
        lines.extend([f'## {repo}','',f'[完整指标CSV](analysis/monthly_by_repository/{slug}.csv)',''])
        if valid.empty:
            lines.extend(['观察期没有合并PR，合并口径占比全部为NA；创建口径单独保留。',''])
        else:
            peak=valid.idxmax()
            lines.extend([f'可计算月份为{valid.index[0]}至{valid.index[-1]}，'
                f'合并增删行占比最大值为{fmt(valid.loc[peak])}（{peak}；若并列仅列首月）。',''])
        lines.append(md_table(['月份','合并PR','Agent合并PR','全部增删行','Agent参与PR增删行',
            '合并增删行占比','合并新增行占比','创建增删行占比','合并源码占比/范围'],[
            [month,f'{int(r.pr_count):,}',f'{int(r.agent_pr_count):,}',f'{int(r.all_text_churn):,}',
             f'{int(r.agent_text_churn):,}',fmt(r.agent_text_churn_pct),fmt(r.agent_additions_pct),
             fmt(c.loc[month,'agent_text_churn_pct']),fmt(r.agent_source_churn_pct) if r.source_coverage_complete
             else fmt(r.source_share_lower_bound_pct)+'—'+fmt(r.source_share_upper_bound_pct)+'（明细不全）']
            for month,r in m.iterrows()]))
        lines.extend(['',f'![{repo}月度趋势](figures/agent_monthly_code_share_{slug}.png)',''])
    (target/'各项目月度Agent代码贡献.md').write_text('\n'.join(lines)+'\n')
    reference='## 按项目分别统计\n\n[各项目月度Agent代码贡献](各项目月度Agent代码贡献.md)：'
    reference+='10个项目各自18个月的月表、独立趋势图、仅新增行指标及源码覆盖范围。分母为各项目自身当月的全部纳入PR行数。\n\n'
    reference+='每项目完整数据见 `analysis/monthly_by_repository/`；月份×项目占比矩阵见 `analysis/monthly_*_by_repository.csv`。'
    for name in ['月度Agent代码贡献趋势.md','高星项目分析报告.md','周报_高星项目_20260924.md','README.md']:
        replace_section(target/name,reference,'repository_monthly_code_share')
    return {'repository_count':len(repositories),'repository_monthly_rows':int((monthly.repository!='ALL_SELECTED').sum()),
            'per_repository_csv_files':len(repositories),'ratio_matrix_files':matrix_files}


def add_monthly(target: Path, df=None, cov=None) -> None:
    if df is None:df=pd.read_parquet(target/'features/prs.parquet')
    if cov is None:cov=pd.read_csv(target/'metadata/repositories.csv')
    manifest=json.loads((target/'metadata/manifest.json').read_text())
    monthly,by_agent=monthly_tables(df,list(cov.repository),manifest['observation_start'],manifest['observation_end_exclusive'])
    monthly.to_csv(target/'analysis/contributions_repo_month.csv',index=False)
    by_agent.to_csv(target/'analysis/agent_monthly_line_contributions.csv',index=False)
    focus=monthly[(monthly.repository=='ALL_SELECTED')&(monthly.month>='2025-07')]
    focus.to_csv(target/'analysis/overall_monthly_202507_202606.csv',index=False)
    for cohort in ['created','merged']:
        base=df[df[cohort+'_in_window']]
        actual=base.assign(month=base[cohort+'_at'].str[:7],
            total_churn=base.additions+base.deletions,
            detected_churn=(base.additions+base.deletions).where(base.agent_detected,0)
        ).groupby(['repository','month']).agg(pr_count=('repository','size'),
            agent_pr_count=('agent_detected','sum'),all_text_churn=('total_churn','sum'),
            agent_text_churn=('detected_churn','sum'))
        exported=monthly[(monthly.repository!='ALL_SELECTED')&(monthly.cohort==cohort)].set_index(['repository','month'])
        expected=actual.reindex(exported.index,fill_value=0)
        for key in expected:
            assert np.array_equal(expected[key].to_numpy(),exported[key].to_numpy()),(cohort,key)
        expected_ratio=100*expected.agent_text_churn/expected.all_text_churn.replace(0,np.nan)
        assert np.allclose(expected_ratio,exported.agent_text_churn_pct,equal_nan=True)
        # Independent scalar aggregation from packaged PRs; check every month and
        # reconcile monthly counts/sums to the existing quarterly table.
        q=pd.read_csv(target/'analysis/contributions_repo_quarter_task.csv')
        for month, g in base.groupby(base[cohort+'_at'].str[:7]):
            r=monthly[(monthly.repository=='ALL_SELECTED')&(monthly.cohort==cohort)&(monthly.month==month)].iloc[0]
            assert r.pr_count==len(g)
            assert r.agent_pr_count==int(g.agent_detected.sum())
            assert r.all_text_churn==int((g.additions+g.deletions).sum())
            agent=g[g.agent_detected]
            assert r.agent_text_churn==int((agent.additions+agent.deletions).sum())
        m=monthly[monthly.cohort==cohort].copy();m['quarter']=pd.PeriodIndex(m.month,freq='M').asfreq('Q').astype(str)
        for (repository,quarter),g in m.groupby(['repository','quarter']):
            old=q[(q.repository==repository)&(q.cohort==cohort)&(q.quarter==quarter)&(q.task=='all')].iloc[0]
            for key in ['pr_count','agent_pr_count','all_text_churn','agent_text_churn']:
                assert int(g[key].sum())==int(old[key]),(repository,quarter,key)
    for col in [k for k in monthly if k.endswith('_pct')]:
        assert monthly[col].dropna().between(0,100).all(),col
    # Contribution of each repository to the pooled monthly numerator and
    # denominator. Helps distinguish changed adoption from changed project mix.
    composition=monthly[monthly.repository!='ALL_SELECTED'].copy()
    pooled=monthly[monthly.repository=='ALL_SELECTED'][['cohort','month','all_text_churn','agent_text_churn']]
    composition=composition.merge(pooled,on=['cohort','month'],suffixes=('','_pooled'),validate='many_to_one')
    composition['share_of_pooled_agent_churn_pct']=100*composition.agent_text_churn/composition.agent_text_churn_pooled.replace(0,np.nan)
    composition['share_of_pooled_total_churn_pct']=100*composition.all_text_churn/composition.all_text_churn_pooled.replace(0,np.nan)
    composition.to_csv(target/'analysis/monthly_repository_composition.csv',index=False)
    plot_monthly(monthly,target,'','2025-01')
    plot_monthly(monthly,target,'_202507_202606','2025-07')
    merged=focus[focus.cohort=='merged'].sort_values('month')
    created=focus[focus.cohort=='created'].set_index('month')
    peak=merged.loc[merged.agent_text_churn_pct.idxmax()]
    first,last=merged.iloc[0],merged.iloc[-1]
    may=merged[merged.month=='2026-05'].iloc[0]
    june=merged[merged.month=='2026-06'].iloc[0]
    tf=composition[(composition.repository=='tensorflow/tensorflow')&(composition.cohort=='merged')].set_index('month')
    claw=composition[(composition.repository=='openclaw/openclaw')&(composition.cohort=='merged')].set_index('month')
    lines=['# Agent参与PR的月度代码变更趋势','',
        '主口径为按合并月份统计的Agent参与PR增删行占比；对照口径为按创建月份统计的全部状态PR。两种口径都按月汇总分子、分母后相除，不对仓库百分比求平均。月度是PR创建/合并月份，无法定位PR内部每行代码的实际编写月份。','',
        '**定义：Agent参与PR的全部新增行+删除行 / 当月纳入PR的全部新增行+删除行。** 这包含测试、文档、配置等，并非Agent独立创作代码占比；源码候选另算，缺失时仅报告上下界。','',
        f'完整观察期为2025-01至2026-06。下面单列2025-07至2026-06：已合并PR口径从{first.month}的 **{fmt(first.agent_text_churn_pct)}** 到{last.month}的 **{fmt(last.agent_text_churn_pct)}**，期间最高为 **{peak.month}的{fmt(peak.agent_text_churn_pct)}**。这是描述性变化，不能解释为各项目均持续上升。','',
        md_table(['月份','期内合并PR','Agent合并PR','全部增删行','Agent参与PR增删行','合并增删行占比','创建口径增删行占比','合并源码占比/范围'],[
            [r.month,f'{int(r.pr_count):,}',f'{int(r.agent_pr_count):,}',f'{int(r.all_text_churn):,}',f'{int(r.agent_text_churn):,}',fmt(r.agent_text_churn_pct),fmt(created.loc[r.month,'agent_text_churn_pct']),
             fmt(r.agent_source_churn_pct) if r.source_coverage_complete else fmt(r.source_share_lower_bound_pct)+'—'+fmt(r.source_share_upper_bound_pct)+'（明细不全）'] for r in merged.itertuples()]),'',
        '## 2026年5月至6月的变化拆解','',
        f'合并口径下，Agent参与PR的全文件增删行从{int(may.agent_text_churn):,}降至{int(june.agent_text_churn):,}，同时全部纳入PR增删行从{int(may.all_text_churn):,}增至{int(june.all_text_churn):,}。分子减少与分母增大共同造成占比回落。', '',
        f'TensorFlow的全部增删行从{int(tf.loc["2026-05","all_text_churn"]):,}增至{int(tf.loc["2026-06","all_text_churn"]):,}；OpenClaw的Agent参与PR增删行从{int(claw.loc["2026-05","agent_text_churn"]):,}降至{int(claw.loc["2026-06","agent_text_churn"]):,}。这些是可核验的项目构成与变更量差异，不能据此判断Agent实际使用次数、能力或人工工时下降。', '',
        '![月度代码变更占比](figures/agent_monthly_code_share_202507_202606.png)','',
        '图中源码点估计仅在当月全部纳入PR源码明细合格时显示。橙色区间是缺失文件数据的确定性上下界，不是置信区间；上下界相同也仍保留明细不完整标记。Agent识别误差、两个排除PR的贡献不包含在区间中。','',
        '各Agent月度比例见`analysis/agent_monthly_line_contributions.csv`：any为PR去重总体；各类别分子为该类Agent参与PR的全部变更，类别可重叠，不能相加。新增行占比另有agent_additions_pct字段，避免将删除行当新增代码。','',
        '项目构成拆解见`analysis/monthly_repository_composition.csv`，分别给出每个项目占总体Agent增删行和总体全部增删行的份额。新项目进入、大PR或重构会改变总体曲线，不能仅凭曲线推断Agent能力、使用率或人类工时变化。','',
        '全18个月的原始统计及源码覆盖在`analysis/contributions_repo_month.csv`；配套全期图为`figures/agent_monthly_code_share.png`。所有统计仍是10个选定仓库的纳入集合，yt-dlp排除2条、DeepSeek Harness观察期无PR，原始数据不变。']
    (target/'月度Agent代码贡献趋势.md').write_text('\n'.join(lines)+'\n')
    summary=(f'## 月度代码贡献趋势补充\n\n已新增2025-01至2026-06的月度统计，并单列2025-07至2026-06。'
        f'主口径为当月合并PR中，Agent参与PR的全文件增删行占比：{first.month}为{fmt(first.agent_text_churn_pct)}，'
        f'{last.month}为{fmt(last.agent_text_churn_pct)}；期间峰值在{peak.month}，为{fmt(peak.agent_text_churn_pct)}。'
        '该值是PR层级归属，不等于Agent独立编写行数。源码覆盖不足的月份给出上下界。\n\n'
        '[完整月度表与口径说明](月度Agent代码贡献趋势.md)\n\n![月度变化](figures/agent_monthly_code_share_202507_202606.png)')
    replace_section(target/'高星项目分析报告.md',summary,'monthly_code_share')
    replace_section(target/'周报_高星项目_20260924.md',summary.split('\n\n![月度变化]')[0],'monthly_code_share')
    replace_section(target/'README.md','## 月度分析文件\n\n'
        '- `月度Agent代码贡献趋势.md`：月度结果、全文件与源码口径、2025-07至2026-06汇总。\n'
        '- `analysis/contributions_repo_month.csv`：仓库×创建/合并口径×月，含完整分子分母和源码覆盖。\n'
        '- `analysis/agent_monthly_line_contributions.csv`：Agent类别×创建/合并口径×月，类别允许重叠。\n'
        '- `analysis/overall_monthly_202507_202606.csv`：指定12个月的总体统计。\n'
        '- `analysis/monthly_repository_composition.csv`：各项目对总体月度分子、分母的贡献。\n'
        '- `figures/agent_monthly_code_share*.png/pdf`：18个月、12个月两套图。\n'
        '- `metadata/monthly_validation.json`：逐月对原始PR特征、逐季度对旧统计表的对账。\n\n'
        '离线重算：`python3 code/add_highstar_monthly_analysis.py --dataset .`。','monthly_code_share')
    replace_section(target/'DATA_DICTIONARY.md','## 月度口径补充\n\n'
        '`month`为UTC的YYYY-MM，created口径使用created_at，merged口径使用merged_at；不是行级编写时间。'
        '`agent=any`为去重总参与，其他值为各Agent规则；各Agent比例共用该月全部纳入PR的行数分母。'
        '`share_of_pooled_agent_churn_pct`与`share_of_pooled_total_churn_pct`分别表示该仓库占总体Agent增删行和总体全部增删行的份额。'
        '分母为0时为空；月度百分比不能直接求均值作为全年百分比。','monthly_code_share')
    repo_check=add_repository_monthly(monthly,target,list(cov.repository))
    manifest['analysis_updated_at']=now()
    manifest['analysis_extensions']=list(dict.fromkeys(manifest.get('analysis_extensions',[])+[
        'monthly_line_contributions_202501_202606','monthly_focus_202507_202606','monthly_by_repository_202501_202606']))
    write_json(target/'metadata/manifest.json',manifest)
    check={'at':now(),'passed':True,'months':18,'monthly_rows':len(monthly),'agent_rows':len(by_agent),**repo_check,
        'checks':['Every repository/month/cohort count, churn numerator, denominator and ratio independently checked against PR features','Every pooled monthly PR/Agent count and total/Agent churn checked against PR features','Every repository/cohort/month sum reconciled to prior quarterly counts and churn','All nonmissing percentages are within 0..100'],
        'last_merged_month':last.month,'last_merged_agent_text_churn_pct':float(last.agent_text_churn_pct),
        'peak_merged_month':peak.month,'peak_merged_agent_text_churn_pct':float(peak.agent_text_churn_pct)}
    write_json(target/'metadata/monthly_validation.json',check)
    print(json.dumps(check,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);args=p.parse_args();add_monthly(args.dataset)
