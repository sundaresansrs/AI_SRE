import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

metric_dir = Path(r'C:\AI-SRE\data\raw\gaia\MicroSS\metric\extracted\metric')
all_files = sorted(metric_dir.glob('*.csv'))

print(f"Scanning {len(all_files)} metric files...")
print("=" * 80)

issues = {
    'empty_files': [],
    'non_numeric_values': [],
    'missing_columns': [],
    'load_failures': [],
    'other_issues': []
}

RAW_TARGET_COLUMNS = {'timestamp', 'value'}

for idx, filepath in enumerate(all_files):
    try:
        df = pd.read_csv(filepath, low_memory=False)
        
        if len(df) == 0:
            issues['empty_files'].append(filepath.name)
            continue
        
        if set(df.columns) != RAW_TARGET_COLUMNS:
            issues['missing_columns'].append((filepath.name, list(df.columns)))
            continue
        
        try:
            non_numeric = df[pd.to_numeric(df['value'], errors='coerce').isna() & df['value'].notna()]
            if len(non_numeric) > 0:
                issues['non_numeric_values'].append((filepath.name, len(non_numeric), str(non_numeric['value'].iloc[0])))
        except Exception as e:
            issues['other_issues'].append((filepath.name, f"Non-numeric check failed: {str(e)}"))
    
    except Exception as e:
        issues['load_failures'].append((filepath.name, str(e)))
    
    if (idx + 1) % 2000 == 0:
        print(f"Scanned {idx + 1} / {len(all_files)}...", flush=True)

print("\n" + "=" * 80)
print("EXHAUSTIVE SCAN COMPLETE")
print("=" * 80)

print(f"\n1. EMPTY FILES ({len(issues['empty_files'])} total):")
if issues['empty_files']:
    for fname in issues['empty_files']:
        print(f"   - {fname}")
else:
    print("   None found")

print(f"\n2. NON-NUMERIC VALUES IN 'value' COLUMN ({len(issues['non_numeric_values'])} files):")
if issues['non_numeric_values']:
    for fname, count, sample in issues['non_numeric_values']:
        print(f"   - {fname}: {count} non-numeric values (sample: '{sample}')")
else:
    print("   None found")

print(f"\n3. MISSING OR MISMATCHED COLUMNS ({len(issues['missing_columns'])} files):")
if issues['missing_columns']:
    for fname, cols in issues['missing_columns']:
        print(f"   - {fname}: columns={cols}")
else:
    print("   None found")

print(f"\n4. LOAD FAILURES ({len(issues['load_failures'])} files):")
if issues['load_failures']:
    for fname, err in issues['load_failures']:
        print(f"   - {fname}: {err}")
else:
    print("   None found")

print(f"\n5. OTHER ISSUES ({len(issues['other_issues'])} files):")
if issues['other_issues']:
    for fname, err in issues['other_issues']:
        print(f"   - {fname}: {err}")
else:
    print("   None found")

total_issues = sum(len(v) for v in issues.values())
print("\n" + "=" * 80)
print(f"TOTAL PROBLEMATIC FILES: {total_issues}")
print(f"SAFE FILES: {len(all_files) - total_issues}")
print(f"SUCCESS RATE: {((len(all_files) - total_issues) / len(all_files) * 100):.2f}%")
print("=" * 80)
