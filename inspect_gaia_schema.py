import os, glob, shutil
import py7zr

base = r'c:\AI-SRE\data\raw\gaia\MicroSS'
for ds in ['business', 'metric', 'trace']:
    print(f'\n=== {ds} ===')
    root = os.path.join(base, ds)
    zpath = os.path.join(root, f'{ds}_split.zip')
    target = os.path.join(root, f'{ds}_extracted')
    if os.path.exists(target):
        shutil.rmtree(target)
    with py7zr.SevenZipFile(zpath, mode='r') as zf:
        zf.extractall(path=target)
    files = sorted(glob.glob(os.path.join(target, '**', '*.csv'), recursive=True))
    print('csv_count=', len(files))
    for path in files[:3]:
        rel = os.path.relpath(path, target)
        print(f'--- {rel} ---')
        with open(path, 'r', encoding='utf-8', errors='replace') as fh:
            for i, line in enumerate(fh):
                print(line.rstrip())
                if i >= 4:
                    break
