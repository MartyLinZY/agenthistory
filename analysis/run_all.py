#!/usr/bin/env python3
"""Reproduce manuscript analyses offline from the archived raw records."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=ROOT/'build/analysis')
parser.add_argument('--from-step', choices=['raw', 'projects', 'labels', 'matched', 'interaction', 'modules', 'tables', 'figures', 'validate'], default='raw',
                    help='Resume after correcting a failed step; prerequisite outputs must already exist.')
args = parser.parse_args()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, AGENTHISTORY_OUTPUT=str(output), MPLBACKEND='Agg')
steps = [
    ('raw', [('merge_data.py', []), ('verify_submission.py', []),
             ('github_all/code/analyze_corpus.py', ['--source','github_all/raw','--output',str(output/'corpus'),'--expected-prs','126740']),
             ('top10/code/build_highstar_dataset.py', ['--source','top10','--output',str(output/'projects_base')])]),
    ('projects', [('analysis/projects.py', [])]),
    ('labels', [('analysis/audit_labels.py', []), ('analysis/analyze_labels.py', [])]),
    ('matched', [('analysis/matched.py', [])]),
    ('interaction', [('analysis/extract_interaction.py', []), ('analysis/interaction.py', []), ('analysis/interaction_sensitivity.py', [])]),
    ('modules', [('analysis/extract_modules.py', []), ('analysis/modules.py', []), ('analysis/module_sensitivity.py', [])]),
    ('tables', [('analysis/workflow_tables.py', []), ('analysis/developer_evidence.py', [])]),
    ('figures', [('analysis/prepare_figures.py', []), ('analysis/build_empirical_figures.py', []), ('analysis/build_rq_layout_figures.py', []), ('analysis/build_contribution_figures.py', [])]),
    ('validate', [('analysis/validate_matched.py', []), ('analysis/validate_interaction.py', []), ('analysis/validate_modules.py', []), ('analysis/validate.py', [])]),
]
start = next(i for i, (name, _) in enumerate(steps) if name == args.from_step)
for name, commands in steps[start:]:
    for script, options in commands:
        print(f'Running {script}', flush=True)
        began = time.monotonic()
        with (output/(Path(script).stem+'.log')).open('w') as log:
            result = subprocess.run([sys.executable, str(ROOT/script), *options], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            print(f'Failed: {script}; see {output/(Path(script).stem+".log")}', file=sys.stderr)
            sys.exit(result.returncode)
        print(f'Completed in {time.monotonic()-began:.1f}s', flush=True)
print('Replication completed; see validation.json and the stage-specific output directories.')
