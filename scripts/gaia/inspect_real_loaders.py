import sys
from pathlib import Path

# Repository root, so the script runs from any checkout location.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from src.data_loaders.business_loader import load_business
from src.data_loaders.metric_loader import load_metric
from src.data_loaders.trace_loader import load_trace

roots = {
    'business': str(REPO_ROOT / "data/raw/gaia/MicroSS/business/extracted/business/business_table_2021-08.csv"),
    'metric': str(REPO_ROOT / "data/raw/gaia/MicroSS/metric/extracted/metric/dbservice1_0.0.0.4_docker_cpu_core_0_norm_pct_2021-07-01_2021-07-15.csv"),
    'trace': str(REPO_ROOT / "data/raw/gaia/MicroSS/trace/extracted/trace/trace_table_dbservice1_2021-07.csv"),
}

for kind, path in roots.items():
    if kind == 'business':
        df = load_business(path)
    elif kind == 'metric':
        df = load_metric(path)
    else:
        df = load_trace(path)
    print(f'=== {kind} ===')
    print(df.head(10).to_string(index=False))
    print('\nDtypes:')
    print(df.dtypes.to_string())
    print('---\n')
