import sys
sys.path.append(r'c:\AI-SRE')

from src.data_loaders.business_loader import load_business
from src.data_loaders.metric_loader import load_metric
from src.data_loaders.trace_loader import load_trace

roots = {
    'business': r'c:\AI-SRE\data\raw\gaia\MicroSS\business\extracted\business\business_table_2021-08.csv',
    'metric': r'c:\AI-SRE\data\raw\gaia\MicroSS\metric\extracted\metric\dbservice1_0.0.0.4_docker_cpu_core_0_norm_pct_2021-07-01_2021-07-15.csv',
    'trace': r'c:\AI-SRE\data\raw\gaia\MicroSS\trace\extracted\trace\trace_table_dbservice1_2021-07.csv',
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
