from context import stage
"""Compact empirical-study figures; existing estimates only. Run from any cwd.
Requires matplotlib, pandas, numpy. Exports vector PDFs and PNG previews.
"""
from pathlib import Path
import json
import argparse
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
parser=argparse.ArgumentParser(description=__doc__)
selection=parser.add_mutually_exclusive_group()
selection.add_argument('--modules-only', action='store_true', help='Rebuild only the module heatmap and temporal comparison.')
selection.add_argument('--overview-only', action='store_true', help='Rebuild only the RQ overview, preserving other figures.')
args=parser.parse_args()
def save(fig,name,h):
    # Physical size is shared across panels; labels are included in the exported box.
    for ext in ['pdf','png']:
        pf.save_figure(fig,OUT/f'{name}.{ext}',width=160,height_mm=h,raster_dpi=220)
    fig.canvas.draw()
    QA[name]={'width_mm':160,'height_mm':h,'data':'existing manuscript and analysis CSVs; no refitting'}
    plt.close(fig)

if not args.modules_only:
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
    for x,title,body in [(.015,'RQ1  Tasks + scope','Task-related scope\nModule co-change\nMatched contexts'),
                         (.265,'RQ2  Interaction','Lifecycle activities\nDiscussion + reviews\nIntegration workflows'),
                         (.515,'RQ3  Evolution','Activities + scope\nInteraction trends\nParticipation trends'),
                         (.765,'RQ4  Contributions','Project distribution\nPR vs. changed lines\nConcentration + tails')]:
        box(x,.22,.22,.28,title,body,'#f7f8f9')
    for x in [.125,.375,.625]:
        arrow(.245,.60,x,.51)
    arrow(.755,.60,.875,.51)
    arrow(.755,.60,.625,.51)
    for x in [.245,.495,.745]:arrow(x,.36,x+.015,.36)
    ax.text(.5,.115,'Tasks + scope → workflow interaction → evolution → project contributions',ha='center',fontsize=7.4)
    ax.text(.5,.025,'Results → empirical findings → engineering implications and validity limits',ha='center',fontsize=8.0,fontweight='bold')
    save(fig,'study_overview',68)

if args.overview_only:
    checks=ROOT/'rq_layout_checks.json'
    prior=json.loads(checks.read_text()) if checks.exists() else {}
    prior.update(QA)
    checks.write_text(json.dumps(prior,indent=2)+'\n')
    print('Rebuilt RQ overview only; empirical figures unchanged.')
    raise SystemExit(0)


# Compact late figures built from unchanged CSVs and recorded manuscript values.
mods=['interface','service_api','storage','security','automation']
names=['Interface','API/service','Storage','Auth/security','CI/build/deploy']
t=pd.read_csv(D/'module_temporal_standardized.csv')
agents=['codex','copilot','claude_code','jules','devin']
coverage=pd.read_csv(D/'agent_coverage.csv')
profile=coverage[coverage.category.isin(mods)].pivot(index='category',columns='group',values='pct').loc[mods,agents]
assert profile.shape == (5,5) and np.isfinite(profile.to_numpy()).all()
fig,(heat,dots)=plt.subplots(1,2,figsize=(160/25.4,54/25.4),sharey=True,
    gridspec_kw={'width_ratios':[1,1.45],'wspace':.18})
fig.subplots_adjust(left=.16,right=.985,bottom=.24,top=.73)
heat.imshow(profile.to_numpy(),cmap='Blues',vmin=0,vmax=40,aspect='auto')
for row in range(5):
 for col in range(5):
  value=profile.iloc[row,col]
  heat.text(col,row,f'{value:.1f}',ha='center',va='center',fontsize=7,
            color='white' if value>25 else '#222222')
heat.set_yticks(range(5),names,fontsize=7.5)
heat.set_xticks(range(5),['Codex','Copilot','Claude\nCode','Jules','Devin'],fontsize=6.5)
heat.tick_params(length=0)
heat.set_title('(a) Full-period coverage (%)',loc='left',fontsize=8,pad=8)
for spine in heat.spines.values():spine.set_visible(False)
for offset,mode,label,color,marker in [(-.2,'pooled','Observed','#666666','o'),(0,'fixed_agent','Fixed agent','#D55E00','s'),(.2,'fixed_agent_task','Fixed agent + task','#0072B2','^')]:
 d=t[t['mode']==mode].set_index('category').loc[mods]
 assert np.allclose(d.q2_pct-d.q1_pct,d.delta_pp)
 dots.scatter(d.delta_pp,np.arange(5)+offset,label=label,color=color,marker=marker,s=22,zorder=3)
dots.axvline(0,color='#666666',lw=.8)
dots.set_ylim(4.5,-.5);dots.set_xlim(-1.95,1.05)
dots.set_xticks([-1.5,-1,-.5,0,.5,1]);dots.tick_params(axis='x',labelsize=7)
dots.tick_params(axis='y',length=0,labelleft=False)
dots.set_title('(b) 2026Q2 − Q1 change',loc='left',fontsize=8,pad=8)
dots.set_xlabel('Percentage points',fontsize=7.5);dots.grid(axis='x',alpha=.15)
dots.spines['left'].set_visible(False)
fig.legend(*dots.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.58,1.01),
    ncol=3,frameon=False,fontsize=7,columnspacing=1.2,handletextpad=.4)
# Preserve physical dimensions for each chart rather than upscaling to 160 mm.
def narrow_save(fig,name,width,height):
 for ext in ['pdf','png']:pf.save_figure(fig,OUT/f'{name}.{ext}',width=width,height_mm=height,raster_dpi=240)
 QA[name]={'width_mm':width,'height_mm':height,'data':'unchanged CSVs or explicitly transcribed manuscript values'}
 plt.close(fig)
narrow_save(fig,'module_temporal_compact',160,54)
QA['module_temporal_compact'].update({
 'panels':['full-period module coverage by agent (%)','2026Q2 minus Q1 module coverage (percentage points)'],
 'sources':['agent_coverage.csv','module_temporal_standardized.csv'],
 'heatmap_cells':25,'temporal_points':15})
if args.modules_only:
 checks=ROOT/'rq_layout_checks.json'
 prior=json.loads(checks.read_text()) if checks.exists() else {}
 prior.update(QA)
 checks.write_text(json.dumps(prior,indent=2)+'\n')
 print('Rebuilt module heatmap and temporal comparison only; other figures unchanged.')
 raise SystemExit(0)

r=pd.read_csv(D/'repair_quarter.csv').query("task_type == 'bug_fix'")
z=pd.read_csv(D/'repair_agent_quarter.csv').query("task_type == 'bug_fix'")
agents=['codex','copilot','claude_code','jules','devin']
fig,axes=plt.subplots(2,1,figsize=(74/25.4,79/25.4),sharex=True,gridspec_kw={'hspace':.3})
x=np.arange(len(r))
for col,label,color,mark in [('human_pct','Overall','#009E73','o'),('text_pct','External, 30 d','#D55E00','s')]:
 axes[0].plot(x,r[col],color=color,marker=mark,ms=2.7,lw=1,label=label)
axes[0].set_ylim(0,21);axes[0].set_yticks([0,10,20]);axes[0].set_ylabel('Discussion (%)',fontsize=7.5)
axes[0].legend(loc='upper left',ncol=1,frameon=False,fontsize=6.2,handlelength=1.1,labelspacing=.2)
for agent,name,color,mark in zip(agents,['Codex','Copilot','Claude','Jules','Devin'],['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00'],['o','s','^','D','x']):
 d=z[z.agent==agent].set_index('quarter').reindex(r.quarter)
 axes[1].plot(x,d.lines_median.where(d.n>=20),color=color,marker=mark,ms=2.7,lw=1,label=name)
axes[1].set_yscale('log');axes[1].set_yticks([10,50,100],['10','50','100']);axes[1].minorticks_off();axes[1].set_ylabel('Median lines',fontsize=7.5)
axes[1].legend(loc='upper center',bbox_to_anchor=(.45,-.27),ncol=3,frameon=False,fontsize=6,handlelength=1.1,columnspacing=.75,labelspacing=.2)
axes[1].set_xticks(x,['25\nQ1','25\nQ2','25\nQ3','25\nQ4','26\nQ1','26\nQ2'],fontsize=6.5)
for ax in axes:ax.grid(alpha=.15);ax.tick_params(axis='y',labelsize=7)
narrow_save(fig,'repair_temporal_narrow',74,79)

# Values already reported in the manuscript; no refitting or interpolated estimates.
df=pd.DataFrame([['Overall','Pooled',13.92,14.84],['Overall','Fixed agent',15.25,15.18],['External, 30 d','Pooled',8.79,5.66],['External, 30 d','Fixed agent',8.01,6.61]],columns=['measure','comparison','q1_pct','q2_pct'])
df.to_csv(D/'repair_composition_manuscript.csv',index=False)
fig,axes=plt.subplots(2,1,figsize=(74/25.4,79/25.4),sharex=True,gridspec_kw={'hspace':.48})
for ax,metric in zip(axes,['Overall','External, 30 d']):
 for mode,color,mark in [('Pooled','#0072B2','o'),('Fixed agent','#D55E00','s')]:
  row=df[(df.measure==metric)&(df.comparison==mode)].iloc[0];y=[row.q1_pct,row.q2_pct]
  ax.plot([0,1],y,color=color,marker=mark,ms=3,lw=1,label=mode)
  for i,val in enumerate(y):
   dy=5 if (metric=='Overall' and mode=='Fixed agent') or (metric!='Overall' and ((mode=='Pooled' and i==0) or (mode=='Fixed agent' and i==1))) else -10
   ax.annotate(f'{val:.2f}',(i,val),xytext=(0,dy),textcoords='offset points',ha='center',fontsize=6.5,color=color)
 ax.set_title(metric,loc='left',fontsize=7.5,pad=3);ax.set_ylim(0,20);ax.set_yticks([0,10,20]);ax.set_xlim(-.22,1.22);ax.grid(axis='y',alpha=.15);ax.set_ylabel('Discussion (%)',fontsize=7.5);ax.tick_params(labelsize=7)
axes[1].set_xticks([0,1],['2026Q1','2026Q2'],fontsize=7)
axes[1].legend(loc='upper center',bbox_to_anchor=(.48,-.23),ncol=2,frameon=False,fontsize=6.3,handlelength=1.2,columnspacing=1)
narrow_save(fig,'repair_composition_narrow',74,79)
(ROOT/'rq_layout_checks.json').write_text(json.dumps(QA,indent=2)+'\n')
print('Rebuilt overview, compact module trends, and two narrow vertical-panel repair figures.')
