"""Map regenerated analysis tables to the manuscript figure inputs."""
import shutil
import pandas as pd
from context import OUTPUT, stage
out = stage('figure_data')
for target, source in {
    'phase_summary.csv': 'labels/phase_summary.csv',
    'scope_repositories.csv': 'matched/scope_repository_agent_quarter_repositories.csv',
    'scope_comparisons.csv': 'matched/scope_comparisons.csv',
    'agent_coverage.csv': 'modules/agent_coverage.csv',
    'module_temporal_standardized.csv': 'modules/temporal_standardized.csv',
    'repair_quarter.csv': 'labels/task_type_quarter.csv',
    'repair_agent_quarter.csv': 'labels/task_type_agent_quarter.csv',
    'tail_sensitivity.csv': 'matched/tail_sensitivity.csv',
}.items():
    shutil.copy2(OUTPUT/source, out/target)
m = pd.read_csv(OUTPUT/'projects/monthly_shares.csv')
m[m.repository == 'ALL_SELECTED'].to_csv(out/'monthly_shares_all_selected.csv', index=False)
m[m.repository == 'ALL_SELECTED'].to_csv(OUTPUT/'projects/monthly_shares_all_selected.csv', index=False)
