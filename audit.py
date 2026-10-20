"""Verify publication files, selected weights, normalization and comparison data."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify(root=ROOT):
    root=Path(root).resolve()
    manifest=json.loads((root/'artifact_manifest.json').read_text())
    for name,record in manifest['files'].items():
        path=(root/name).resolve()
        path.relative_to(root)
        if path.stat().st_size!=record['bytes'] or digest(path)!=record['sha256']:
            raise ValueError('Artifact mismatch: '+name)
    run=root/'model/runs/selected'
    report=json.loads((run/'report.json').read_text())
    freeze=json.loads((root/'model/study_20261019/test_freeze.json').read_text())
    selected=next(r for r in freeze['checkpoints'] if r['run']==freeze['selection']['run'])
    if digest(run/'best_weights.pth')!=selected['sha256'] or selected['sha256']!=report['checkpoint']['sha256']:
        raise ValueError('Selected checkpoint mismatch')
    if digest(run/'normalization.npz')!=report['normalization_sha256']:
        raise ValueError('Normalization mismatch')
    if freeze['selection']['test_used_for_selection'] or report.get('checkpoint_partial_epoch'):
        raise ValueError('Checkpoint selection boundary is invalid')
    expected='9a21558378d6359f308e3b7b7a3068d053195fa9123792c1d8aa9ac177101022'
    if report['data_sha256']!=expected:
        raise ValueError('Unexpected training corpus')
    with np.load(run/'normalization.npz',allow_pickle=False) as n:
        if not np.isfinite(n['data_min']).all() or not np.isfinite(n['data_max']).all() or np.any(n['data_max']<=n['data_min']):
            raise ValueError('Invalid normalization bounds')
    summary=json.loads((root/'results/wind_comparison/annual_wind_summary.json').read_text())
    for label,row in summary.items():
        values=np.asarray(row['annual_means'])
        if len(values)!=row['count'] or not np.isclose(values.std(),row['annual_mean_SD'],atol=1e-12):
            raise ValueError('Annual comparison arithmetic mismatch: '+label)
    return {'status':'passed','files_verified':len(manifest['files']),
            'checkpoint_sha256':selected['sha256'],'data_sha256':expected,
            'training_years':[1980,2015],'validation_years':[2016,2019],'reporting_years':[2020,2024],
            'scope':'Artifact and recorded data-flow verification; reporting years were previously inspected.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-files',action='store_true',help='Accepted for compatibility; files are always verified')
    parser.parse_args()
    print(json.dumps(verify(),indent=2))

if __name__=='__main__':main()
