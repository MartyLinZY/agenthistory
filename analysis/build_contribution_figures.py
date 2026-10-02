from context import stage
"""Build single-column contribution figures from existing aggregate results."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import figure_io as pf

ROOT = stage('figure_checks')
DATA = stage('figure_data')
OUT = stage('figures')
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42,
})
QA = {}


def save(fig, name, height):
    for suffix in ['pdf', 'png']:
        pf.save_figure(fig, OUT / f'{name}.{suffix}', width=80,
                       height_mm=height, raster_dpi=240)
    QA[name] = {'width_mm': 80, 'height_mm': height,
                'data': 'existing aggregate CSVs; no refitting'}
    plt.close(fig)


# Separate panels emphasize sensitivity within each cohort on the same scale.
tail = pd.read_csv(DATA / 'tail_sensitivity.csv')
modes = ['raw', 'drop_top_1pct', 'winsorize_99pct']
fig, axes = plt.subplots(2, 1, sharey=True,
                         figsize=(80 / 25.4, 83 / 25.4))
fig.subplots_adjust(left=.19, right=.98, bottom=.10, top=.91, hspace=.65)
x = np.arange(3)
for ax, cohort, title in zip(axes, ['created', 'merged'],
                            ['(a) Created PR cohort', '(b) Merged PR cohort']):
    rows = tail[tail.cohort == cohort].set_index('mode').loc[modes]
    assert np.allclose(rows.signal_lines / rows.total_lines * 100,
                       rows.line_share_pct)
    bars = ax.bar(x, rows.line_share_pct, width=.60,
                  color=['#777777', '#0072B2', '#D55E00'],
                  edgecolor='white', linewidth=.4, zorder=3)
    ax.bar_label(bars, labels=[f'{v:.2f}' for v in rows.line_share_pct],
                 padding=3, fontsize=7)
    ax.set_xticks(x, ['Raw', 'Trimmed', 'Capped'], fontsize=8)
    ax.set_ylim(0, 12); ax.set_yticks([0, 3, 6, 9, 12])
    ax.set_ylabel('Changed-line share (%)', fontsize=8)
    ax.set_title(title, loc='left', fontsize=8, pad=6)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(axis='y', alpha=.15)
save(fig, 'line_share_tail_sensitivity', 83)

# Stack the monthly series and share the time axis to avoid two sparse panels.
monthly = pd.read_csv(DATA / 'monthly_shares_all_selected.csv')
months = sorted(monthly.month.unique())
assert len(months) == 18
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(80 / 25.4, 83 / 25.4))
fig.subplots_adjust(left=.17, right=.98, bottom=.13, top=.86, hspace=.43)
for cohort, label, color, marker in [
    ('created', 'Created', '#0072B2', 'o'),
    ('merged', 'Merged', '#D55E00', 's'),
]:
    rows = monthly[monthly.cohort == cohort].set_index('month').loc[months]
    assert np.allclose(rows.agent_n / rows.n * 100, rows.pr_pct)
    assert np.allclose(rows.agent_lines / rows.lines * 100, rows.lines_pct)
    for ax, column in zip(axes, ['pr_pct', 'lines_pct']):
        ax.plot(range(18), rows[column], label=label, color=color,
                marker=marker, markersize=2.5, linewidth=1)
for ax, title, ticks in zip(axes, ['(a) PR share', '(b) Changed-line share'],
                            [[0, 5, 10, 15], [0, 5, 10, 15]]):
    ax.set_title(title, loc='left', fontsize=8, pad=4)
    ax.set_ylim(0, 16); ax.set_yticks(ticks)
    ax.set_ylabel('Share (%)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=.15)
axes[1].set_xticks([0, 6, 12, 17], ['Jan 25', 'Jul 25', 'Jan 26', 'Jun 26'])
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
           bbox_to_anchor=(.57, 1.01), ncol=2, frameon=False,
           fontsize=8, handlelength=1.4, columnspacing=1.1)
save(fig, 'highstar_monthly_v2', 83)

(ROOT / 'contribution_layout_checks.json').write_text(
    json.dumps(QA, indent=2) + '\n')
print('Rebuilt single-column monthly and tail-sensitivity figures.')
