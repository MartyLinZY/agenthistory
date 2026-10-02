from context import stage
"""Compact empirical-study figures; existing estimates only. Run from any cwd.
Requires matplotlib, pandas, numpy. Exports vector PDFs and PNG previews.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import numpy as np
import figure_io as pf
ROOT=stage('figure_checks')
D=stage('figure_data'); OUT=stage('figures')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8.5,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
QA={}
def save(fig,name,h):
    # Physical size is shared across panels; labels are included in the exported box.
    for ext in ['pdf','png']:
        pf.save_figure(fig,OUT/f'{name}.{ext}',width=160,height_mm=h,raster_dpi=220)
    fig.canvas.draw()
    QA[name]={'width_mm':160,'height_mm':h,'data':'existing manuscript and analysis CSVs; no refitting'}
    plt.close(fig)

# Fig. 1: analytical connections, with acquisition counts and explicit populations.
fig,ax=plt.subplots(figsize=(160/25.4,78/25.4))
ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
def box(x,y,w,h,title,body,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.012',
                 facecolor=color,edgecolor='#7c8a96',linewidth=.6))
    ax.text(x+.015,y+h-.028,title,ha='left',va='top',fontsize=8.2,fontweight='bold')
    ax.text(x+.015,y+h-.105,body,ha='left',va='top',fontsize=7.3,linespacing=1.4)
def arrow(x1,y1,x2,y2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops={'arrowstyle':'->','lw':.8,'color':'#52616d'})
box(.015,.60,.46,.38,'A  Agent-associated PRs',
    'Discovery frame: 12,863,809 IDs\nSelected: 128,637 → retrieved: 126,740\nAnnotated: 126,685 (tasks + lifecycle)\n82,023 repositories; five agent groups','#edf5fa')
box(.525,.60,.46,.38,'B  Selected-project enumeration',
    'Ten repositories; 190,094 retrieved PRs\nCreated cohort: 189,757 PRs\nMerged cohort: 76,617 PRs\nAgent signals + no-signal PRs','#f4f2eb')
for x,title,body in [(.015,'RQ1  Activities','Lifecycle coverage\nDiscussion + merge\nworkflow decomposition'),
                     (.265,'RQ2  Change scope','Tasks + module paths\nMatched contexts\nFixed composition'),
                     (.515,'RQ3  Repairs','Quarterly scope\nDiscussion trends\nRecurring-project panels'),
                     (.765,'RQ4  Contribution','PR + line shares\nProject denominators\nAttribution + size tails')]:
    box(x,.22,.22,.28,title,body,'#f7f8f9')
for x in [.125,.375,.625]:
    arrow(.245,.60,x,.51)
arrow(.755,.60,.875,.51)
for x in [.245,.495,.745]:arrow(x,.36,x+.015,.36)
ax.text(.5,.115,'Recorded activity → change characteristics → temporal patterns → project context',ha='center',fontsize=7.4)
ax.text(.5,.025,'Results → empirical findings → engineering implications and validity limits',ha='center',fontsize=8.0,fontweight='bold')
save(fig,'study_overview',78)

# Fig. 2: lifecycle with non-redundant workflow context in the adjacent panel.
p=pd.read_csv(D/'phase_summary.csv')
fig=plt.figure(figsize=(160/25.4,67/25.4))
gs=fig.add_gridspec(1,3,width_ratios=[1.45,1,1.55],wspace=.46)
a,b,c=[fig.add_subplot(gs[0,i]) for i in range(3)]
y=np.arange(7);names=['Req. elicitation','Req. analysis','Design','Implementation','Integration','Testing','Deployment']
a.barh(y,p.coverage_pct,color='#0072B2',height=.55)
a.set_yticks(y,names);a.invert_yaxis();a.set_xlim(0,103);a.set_xticks([0,50,100]);a.set_xlabel('Phase coverage (%)')
a.set_title('(a) Lifecycle activity',loc='left',pad=12)
for i,v in enumerate(p.coverage_pct):a.text(v+1,i,f'{v:.1f}',va='center',fontsize=7)
b.scatter(p.human_pct,y-.1,label='Overall',color='#009E73',s=18,marker='o')
b.scatter(p.text_pct,y+.1,label='External, 30 d',color='#D55E00',s=19,marker='x')
a.set_ylim(6.6,-.6);b.set_yticks(y,[]);b.set_ylim(6.6,-.6);b.set_xlim(0,26);b.set_xticks([0,10,20]);b.set_xlabel('Discussion (%)')
b.legend(loc='upper center',bbox_to_anchor=(.5,1.2),fontsize=6.5,frameon=False)
c.set_title('(b) Merge workflows',loc='left',pad=12)
c.barh([0,1],[2.06,37.51],height=.35,color=['#0072B2','#D55E00'])
c.set_yticks([0,1],['Author\naccount','Other\naccount']);c.invert_yaxis();c.set_xlim(0,48)
c.set_xlabel('Pre-merge external\nevidence (%)');c.set_xticks([0,20,40])
for i,(v,n) in enumerate([(2.06,88470),(37.51,20003)]):
    c.text(v+1,i,f'{v:.2f}%',va='center',fontsize=7)
    c.text(0,i+.31,f'n = {n:,}',fontsize=7,color='#555')
c.set_ylim(1.65,-.65)
for ax in [a,b,c]:ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
save(fig,'lifecycle_1pct',67)

# Fig. 3: unadjusted task medians alongside within-context size distribution.
task=pd.DataFrame({'task':['Feature','Refactoring','Testing','Documentation','Bug fix','Config./maint.','Other'],
 'files':[5,5,3,2,2,2,1],'lines':[325,212,182,96,57,45,34]})
task.to_csv(D/'task_scope_manuscript.csv',index=False)
g=pd.read_csv(D/'scope_repositories.csv');s=pd.read_csv(D/'scope_comparisons.csv').query("comparison == 'repository_agent_quarter'").iloc[0]
fig,axes=plt.subplots(1,2,figsize=(160/25.4,67/25.4),gridspec_kw={'width_ratios':[1,1.25],'wspace':.6})
a,b=axes;y=np.arange(7)
a.scatter(task.lines,y,c='#0072B2',s=22)
a.set_yticks(y,task.task);a.invert_yaxis();a.set_xlim(0,420);a.set_xticks([0,100,200,300]);a.set_xlabel('Median changed lines')
a.set_title('(a) Task scope',loc='left')
for i,r in task.iterrows():a.text(r.lines+15,i,f'{r.lines} / {r.files}f',va='center',fontsize=7)
a.text(.02,.03,'Labels: lines / files (f)',transform=a.transAxes,fontsize=7)
a.set_ylim(7,-.7)
x=np.sort(np.exp(g.delta));b.step(x,np.arange(1,len(x)+1)/len(x),lw=1.4,color='#0072B2')
b.axvline(1,color='#777',ls='--',lw=.8);b.axvline(s.median_log1p_ratio,color='#D55E00',ls=':',lw=1)
b.set_xscale('log');b.set_ylim(0,1.02);b.set_xlabel('Matched feature / repair ratio\n(+1 line offset)');b.set_ylabel('Fraction of repositories')
b.set_title('(b) Matched scope',loc='left')
b.text(.03,.96,f'{len(g):,} repositories\nFeatures larger: {s.positive_pct:.2f}%\nMedian: {s.median_log1p_ratio:.2f}\n95% CI: {s.ratio_ci_low:.2f}–{s.ratio_ci_high:.2f}',
       transform=b.transAxes,va='top',fontsize=7,bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
for ax in axes:ax.grid(alpha=.15);ax.set_axisbelow(True)
save(fig,'matched_task_scope',67)

# Fig. 4: align module profiles and composition-sensitive changes horizontally.
mods=['interface','service_api','storage','security','automation'];agents=['codex','copilot','claude_code','jules','devin']
names=['Interface','API/service','Storage','Auth/security','CI/build/deploy']
d=pd.read_csv(D/'agent_coverage.csv');m=d[d.category.isin(mods)].pivot(index='category',columns='group',values='pct').loc[mods,agents]
t=pd.read_csv(D/'module_temporal_standardized.csv')
fig,axes=plt.subplots(1,2,figsize=(160/25.4,66/25.4),gridspec_kw={'width_ratios':[1.5,1],'wspace':.12})
a,b=axes;a.imshow(m,cmap='Blues',vmin=0,vmax=40,aspect='auto')
for i in range(5):
 for j in range(5):a.text(j,i,f'{m.iloc[i,j]:.1f}',ha='center',va='center',fontsize=7.5,color='white' if m.iloc[i,j]>25 else '#222')
a.set_yticks(range(5),names);a.set_xticks(range(5),['Codex','Copilot','Claude\nCode','Jules','Devin'],fontsize=7)
a.tick_params(length=0);a.set_title('(a) PRs touching module paths (%)',loc='left',pad=12)
for sp in a.spines.values():sp.set_visible(False)
for offset,mode,label,color,marker in [(-.2,'pooled','Observed','#666','o'),(0,'fixed_agent','Fixed agent','#D55E00','s'),(.2,'fixed_agent_task','Fixed agent + task','#0072B2','^')]:
 z=t[t['mode']==mode].set_index('category').loc[mods]
 b.scatter(z.delta_pp,np.arange(5)+offset,label=label,color=color,marker=marker,s=23)
b.axvline(0,color='#666',lw=.8);b.set_ylim(4.5,-.5);b.set_yticks(range(5),[]);b.set_xlim(-1.95,1.05)
b.set_title('(b) Q2 − Q1 change',loc='left',pad=12);b.set_xlabel('Percentage points');b.grid(axis='x',alpha=.15)
b.legend(loc='lower left',bbox_to_anchor=(-.02,-.42),ncol=1,frameon=False,fontsize=6.7,labelspacing=.25)
save(fig,'module_profiles_changes',66)
(ROOT/'empirical_layout_checks.json').write_text(json.dumps(QA,indent=2)+'\n')
print('Exported study overview and three redesigned result figures.')
