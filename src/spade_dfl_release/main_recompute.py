"""Recompute main-comparison statistics from saved run records; never train."""
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from .reference import DATASETS, METHODS, PARTITIONS, MAIN_ROOTS, aggregate_main_curves, verify_hash, write_csv

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def read_index(root):
    return read_csv(Path(root)/'reference_results/main_seed_records/run_index.csv')

def validate_index(rows):
    keys=[(r['dataset'],r['partition'],r['method'],int(r['fold']),int(r['seed'])) for r in rows]
    expected={(d,p,m,f,s) for d in DATASETS for p in PARTITIONS for m in METHODS for f in range(5 if d=='alzheimer' else 1) for s in range(30)}
    if len(keys)!=len(set(keys)) or set(keys)!=expected or len({r['run_id'] for r in rows})!=len(rows):
        raise ValueError('incomplete or duplicate run identities')

def read_curve(rows):
    if [int(r['round']) for r in rows]!=list(range(51)):raise ValueError('expected ordered rounds 0..50')
    values=np.asarray([float(r['accuracy']) for r in rows])
    if not np.all(np.isfinite(values)) or np.any((values<0)|(values>1)):raise ValueError('invalid accuracy')
    return values

def compare(actual,expected):
    def key(r):return tuple(str(r[k]) for k in ('dataset','partition','method','round'))
    a={key(r):r for r in actual};b={key(r):r for r in expected}
    if len(a)!=len(actual) or len(b)!=len(expected) or set(a)!=set(b):raise ValueError('aggregate identities differ')
    worst=0.0
    for k,r in a.items():
        for f in ('mean_percent','sd_percent','n_trajectories','ddof'):
            delta=abs(float(r[f])-float(b[k][f]));worst=max(worst,delta)
            if delta!=0:raise ValueError('statistics differ: '+repr((k,f,delta)))
    return worst

def recompute(root):
    root=Path(root);base=root/'reference_results/main_seed_records'
    manifest=json.loads((base/'MANIFEST.json').read_text(encoding='utf-8'))
    for f in manifest['files']:verify_hash(base/f['path'],f['sha256'])
    index=read_index(root);validate_index(index)
    configurations=json.loads((base/'configurations.json').read_text(encoding='utf-8'))
    environments=json.loads((base/'environments.json').read_text(encoding='utf-8'))
    raw=defaultdict(list)
    for path in sorted({r['result_path'] for r in index}):
        for row in read_csv(base/path):raw[row['run_id']].append(row)
    if set(raw)!={r['run_id'] for r in index}:raise ValueError('curve run identities differ')
    groups=defaultdict(list)
    for row in index:
        if row['configuration_id'] not in configurations or row['environment_id'] not in environments:
            raise ValueError('configuration/environment identity missing')
        curve=read_curve(raw[row['run_id']])
        if curve[-1]!=float(row['final_accuracy']):raise ValueError('summary/history endpoint differs')
        groups[(row['dataset'],row['partition'],row['method'])].append((int(row['fold']),int(row['seed']),curve))
    final=[];curves=[]
    for dataset,partition,method in itertools.product(DATASETS,PARTITIONS,METHODS):
        ordered=sorted(groups[(dataset,partition,method)],key=lambda x:x[:2])
        matrix=np.asarray([r[2] for r in ordered]);mean,sd=aggregate_main_curves(matrix)
        source,prefix=MAIN_ROOTS[dataset]
        common=dict(dataset=dataset,partition=partition,method=method,n_trajectories=len(matrix),ddof=0,source_id=source,source_member_prefix=prefix)
        final.append(dict(common,round=50,mean_percent=float(mean[-1]),sd_percent=float(sd[-1])))
        curves.extend(dict(common,round=k,mean_percent=float(mean[k]),sd_percent=float(sd[k])) for k in range(51))
    return final,curves,dict(runs=len(index),points=sum(len(v) for v in raw.values()))

def verify(root,output=None):
    root=Path(root);final,curves,info=recompute(root)
    worst=max(compare(final,read_csv(root/'reference_results/aggregates/main_final_accuracy.csv')),
              compare(curves,read_csv(root/'reference_results/aggregates/main_roundwise.csv')))
    result=dict(info,final_rows=len(final),curve_rows=len(curves),max_numeric_difference=worst,training_executed=False)
    if output is not None:
        output=Path(output).resolve()
        if output==root.resolve() or root.resolve() in output.parents:
            raise ValueError('output must be outside the release directory')
        output.mkdir(parents=True,exist_ok=False)
        write_csv(output/'main_final_accuracy.csv',final)
        write_csv(output/'main_roundwise.csv',curves)
        (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result
