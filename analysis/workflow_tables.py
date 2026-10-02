from context import ARTIFACT, OUTPUT, stage, raw_open
"""Reproduce RQ2 workflow tables from aggregate counts validated against PR records."""
from pathlib import Path
import itertools
import csv
import json
import math

ROOT = stage('interaction')
OUT = stage('tables')
OUT.mkdir(exist_ok=True)
NAMES = {'codex': 'Codex', 'copilot': 'Copilot', 'claude_code': 'Claude Code',
         'jules': 'Jules', 'devin': 'Devin'}
cells = {}
with (ROOT / 'workflow_agent_actor.csv').open() as f:
    for row in csv.DictReader(f):
        n, positive = int(row['n']), int(row['positive'])
        assert n > 0 and 0 <= positive <= n
        rate = positive / n
        assert math.isclose(rate, float(row['rate']), rel_tol=1e-12, abs_tol=1e-12)
        key = (row['agent'], row['same_author_merger'] == 'True')
        assert key not in cells
        cells[key] = (n, positive, rate)
assert sum(v[0] for v in cells.values()) == 108473

def row_text(values):
    return ' & '.join(values) + r' \\'

lines = [r'\begin{tabular}{lrrrrr}', r'\toprule',
         r' & \multicolumn{2}{c}{Author-account merge} & \multicolumn{2}{c}{Other-account merge} & Overall \\',
         r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
         r'Agent & PRs & Coverage (\%) & PRs & Coverage (\%) & Coverage (\%) \\', r'\midrule']
summary = []
for agent, name in NAMES.items():
    values = [name]
    total = positive = 0
    for same in [True, False]:
        cell = cells.get((agent, same))
        if cell is None:
            values.extend(['0', 'NA'])
        else:
            n, yes, rate = cell
            values.extend([f'{n:,}', f'{100 * rate:.2f}'])
            total += n
            positive += yes
    overall = positive / total
    values.append(f'{100 * overall:.2f}')
    lines.append(row_text(values))
    summary.append({'agent': agent, 'n': total, 'positive': positive,
                    'same_author_share': cells.get((agent, True), (0, 0, 0))[0] / total,
                    'overall_evidence_pct': 100 * overall})
lines.append(r'\midrule')
values = ['All agents']
for same in [True, False]:
    group = [v for (a, w), v in cells.items() if w == same]
    n = sum(v[0] for v in group)
    values.extend([f'{n:,}', f'{100 * sum(v[1] for v in group) / n:.2f}'])
values.append(f'{100 * sum(v[1] for v in cells.values()) / sum(v[0] for v in cells.values()):.2f}')
lines.extend([row_text(values), r'\bottomrule', r'\end{tabular}'])
(OUT / 'workflow_by_agent.tex').write_text('\n'.join(lines) + '\n')

pairs, missing = [], []
for a, b in itertools.combinations(sorted(NAMES), 2):
    if any((agent, same) not in cells for agent in [a, b] for same in [False, True]):
        missing.append([a, b])
        continue
    na = sum(cells[a, same][0] for same in [False, True])
    nb = sum(cells[b, same][0] for same in [False, True])
    gap = composition = within = 0
    for same in [False, True]:
        wa, wb = cells[a, same][0] / na, cells[b, same][0] / nb
        ra, rb = cells[a, same][2], cells[b, same][2]
        gap += wb * rb - wa * ra
        composition += (wb - wa) * (rb + ra) / 2
        within += (rb - ra) * (wb + wa) / 2
    assert math.isclose(gap, composition + within, abs_tol=1e-12)
    pairs.append({'a': a, 'b': b, 'gap_pp': 100 * gap,
                  'composition_pp': 100 * composition, 'within_pp': 100 * within})
assert len(pairs) == 6 and len(missing) == 4
lines = [r'\begin{tabular}{llrrr}', r'\toprule',
         r'Agent $A$ & Agent $B$ & Gap ($B-A$) & Composition & Within workflow \\', r'\midrule']
for pair in pairs:
    lines.append(row_text([NAMES[pair['a']], NAMES[pair['b']]] +
                         [f'{pair[k]:+.2f}' for k in ['gap_pp', 'composition_pp', 'within_pp']]))
lines.extend([r'\bottomrule', r'\end{tabular}'])
(OUT / 'workflow_decomposition.tex').write_text('\n'.join(lines) + '\n')
(ROOT / 'workflow_table_checks.json').write_text(json.dumps({
    'source': 'workflow_agent_actor.csv', 'known_actor_merges': 108473,
    'summary': summary, 'decompositions': pairs, 'unsupported_pairs': missing,
    'note': 'Missing workflow cells are not assigned a zero event rate.'}, indent=2) + '\n')
print('Generated five-agent workflow table and all six supported pairwise decompositions.')
