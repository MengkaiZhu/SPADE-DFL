"""Frozen S15 suites and independently checked paired roundwise evidence."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from scipy import stats

from .paths import release_root
from .publication import verify_publication_file
from .reference import csv_rows, require_keys, verify_hash


# Canonical archive-verified (member, packaged path, bytes, SHA-256) closure.
# This is an independent portable trust anchor, not a value supplied by manifest.
_CLOSURE_SHA256='415ecc4e742b83c2dbd5644f00acb4dc54024f015315cc9ba6604163fa5ea04b'
_EXTERNAL_INPUT_IDENTITY=('data/mnist.npz',11490434,'731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1')


def expected_comparison_count(methods_against_reference, partitions, rounds):
    values=(methods_against_reference,partitions,rounds)
    if any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in values):
        raise ValueError('comparison dimensions must be positive integers')
    return methods_against_reference*partitions*rounds


def benjamini_hochberg(p_values):
    p=np.asarray(p_values,dtype=np.float64)
    if p.ndim!=1 or not p.size or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('p values must be a nonempty finite probability vector')
    order=np.argsort(p,kind='stable');q=np.empty_like(p);ceiling=1.
    for index in range(p.size-1,-1,-1):
        ceiling=min(ceiling,float(p[order[index]])*(p.size/(index+1)))
        q[order[index]]=ceiling
    return q


def _hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):digest.update(data)
    return digest.hexdigest()


def _vector_hash(values):
    return hashlib.sha256(np.asarray(values,dtype='<f8').tobytes()).hexdigest()


def _text_hash(values):
    return hashlib.sha256(json.dumps(values,separators=(',',':'),ensure_ascii=True).encode('ascii')).hexdigest()


def _vendor_root():
    return release_root()/'src/spade_dfl_release/vendor/heterogeneity'


def _verify_packaged_files(root,manifest):
    root=Path(root).resolve()
    prefix='src/spade_dfl_release/vendor/heterogeneity/'
    identities=[(r['path'],r['bytes'],r['sha256'].lower()) for r in manifest['required_inputs']]
    if identities!=[_EXTERNAL_INPUT_IDENTITY]:raise ValueError('S15 external-input identity mismatch')
    external={prefix+_EXTERNAL_INPUT_IDENTITY[0]:_EXTERNAL_INPUT_IDENTITY[1:]}
    for row in manifest['files']:
        relative=row['packaged_path'];path=(root/relative).resolve()
        if root not in path.parents:raise ValueError('invalid S15 vendor path/size')
        if relative in external:
            if external[relative]!=(row['bytes'],row['sha256'].lower()):
                raise ValueError('S15 external-input provenance mismatch')
            if not path.is_file():continue
        if not path.is_file():
            raise ValueError('invalid S15 vendor path/size')
        verify_publication_file(path,row['sha256'],row['bytes'],root=root)


def load_manifest():
    root=release_root().resolve()
    manifest=json.loads((root/'configs/heterogeneity/manifest.json').read_text(encoding='utf-8'))
    closure=sorted((r['member'],r['packaged_path'],r['bytes'],r['sha256']) for r in manifest['files'])
    if len(closure)!=36 or _text_hash(closure)!=_CLOSURE_SHA256:
        raise ValueError('S15 frozen runtime closure mismatch')
    require_keys([r['member'] for r in manifest['files']],{r['member'] for r in manifest['files']})
    sources=json.loads((root/'reference_results/raw_manifests/source_manifest.json').read_text(encoding='utf-8'))['files']
    archives=[r for r in sources if r['source_id']==manifest['source_id']]
    if len(archives)!=1 or archives[0]['sha256'].lower()!=manifest['archive_sha256'].lower():
        raise ValueError('S15 archive provenance hash mismatch')
    _verify_packaged_files(root,manifest)
    return manifest


def _vendor_call(action,payload=None,*,root=None):
    request=dict(action=action,root=str(Path(root or _vendor_root()).resolve()),payload=payload)
    environment=os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',PYTHONPATH=str(release_root()/'src'))
    result=subprocess.run([sys.executable,'-B','-m','spade_dfl_release.heterogeneity'],input=json.dumps(request),text=True,encoding='utf-8',capture_output=True,env=environment,check=True)
    return json.loads(result.stdout)


def _check_imports():
    load_manifest()
    return _vendor_call('imports')


def _historical_bh(values):
    load_manifest()
    return np.asarray(_vendor_call('bh',list(values)))


def _integer(value):
    x=float(value)
    if not np.isfinite(x) or x!=int(x):raise ValueError('noninteger S15 identity')
    return int(x)


def _same_or_adjacent(a,b):
    a,b=np.float64(a),np.float64(b)
    return bool(np.isfinite(a) and np.isfinite(b) and (a==b or np.nextafter(a,b)==b))


def verify(reference_root: Path):
    root=Path(reference_root);manifest=load_manifest()
    verification=json.loads((root/'paper_values/verification.json').read_text(encoding='utf-8'))
    if verification['status']!='verified' or any(verification[k] for k in ('difference_count','missing_cells','duplicate_cells')):
        raise ValueError('frozen paper verification failed')
    data={}
    for stem in ('roundwise','distribution_summary','client_proportions','wilcoxon_bh'):
        name='s15_'+stem+'.csv';path=root/'aggregates'/name
        verify_hash(path,verification['normalized_sha256'][name]);data[stem]=csv_rows(path.read_bytes())
    specs=manifest['suites'];seeds=tuple(range(30));points=tuple(range(51))
    runs={}
    for suite,spec in specs.items():
        require_keys(spec['seeds'],seeds)
        expected=set(itertools.product(spec['partitions'],spec['methods'],seeds))
        evidence=[r for r in manifest['run_evidence'] if r['suite']==suite]
        require_keys([(r['partition'],r['method'],r['seed']) for r in evidence],expected)
        for r in evidence:runs[(suite,r['partition'],r['method'],r['seed'])]=r
    expected_points={key+(i,) for key in runs for i in points}
    rows=data['roundwise'];keys=[(r['suite'],r['partition'],r['method'],_integer(r['seed']),_integer(r['round'])) for r in rows]
    require_keys(keys,expected_points)
    trajectories={key:np.empty(len(points)) for key in runs}
    for row,key in zip(rows,keys):
        value=float(row['accuracy']);record=runs[key[:-1]];central=row['method']=='zo2_dp_sgd'
        if record['n_clients']!=(1 if central else (1000 if row['suite']=='distribution' else 31)):
            raise ValueError('distribution/method client-count boundary violated')
        if not np.isfinite(value) or not 0<=value<=1:raise ValueError('invalid S15 accuracy')
        if row['axis_semantics']!=('centralized_optimizer_step' if central else 'communication_round'):
            raise ValueError('DP-SGD index semantic exception lost')
        if _integer(row['communication_round'])!=(0 if central else key[-1]) or _integer(row['n_clients'])!=record['n_clients']:
            raise ValueError('S15 communication/client accounting mismatch')
        if record['communication_rounds']!=(0 if central else 50) or central and record['n_clients']!=1:
            raise ValueError('invalid frozen communication evidence')
        trajectories[key[:-1]][key[-1]]=value
    for key,curve in trajectories.items():
        if _vector_hash(curve)!=runs[key]['accuracy_sha256']:raise ValueError('normalized trajectory differs from frozen raw history')
    dist=specs['distribution'];method=dist['methods'][0]
    summaries=data['distribution_summary'];require_keys([(r['partition'],_integer(r['round'])) for r in summaries],set(itertools.product(dist['partitions'],points)))
    for row in summaries:
        index=_integer(row['round']);v=np.asarray([trajectories[('distribution',row['partition'],method,s)][index] for s in seeds])*100
        mean=float(v.mean());half=float(stats.t.ppf(.975,len(v)-1))*float(v.std(ddof=1))/np.sqrt(len(v))
        expected=dict(mean_percent=mean,sd_percent=float(v.std(ddof=0)),ci95_low=mean-half,ci95_high=mean+half)
        if any(float(row[k])!=value for k,value in expected.items()):raise ValueError('distribution statistics differ')
    clients=data['client_proportions'];client_count={r['partition']:r['n_clients'] for r in runs.values() if r['suite']=='distribution'}
    require_keys([(r['partition'],_integer(r['seed']),_integer(r['client'])) for r in clients],{(p,s,c) for p in dist['partitions'] for s in seeds for c in range(client_count[p])})
    proportions={(p,s):np.empty(client_count[p]) for p in dist['partitions'] for s in seeds}
    for row in clients:
        value=float(row['p_y7'])
        if not np.isfinite(value) or not 0<=value<=1:raise ValueError('invalid client digit-7 proportion')
        proportions[(row['partition'],_integer(row['seed']))][_integer(row['client'])]=value
    for (p,s),values in proportions.items():
        if _vector_hash(values)!=runs[('distribution',p,method,s)]['p_y7_sha256']:raise ValueError('client proportions differ from frozen partitions')
    spec=specs['method'];methods=spec['methods'];reference=methods[0]
    if reference!='zo2_lt_admm_vr':raise ValueError('frozen method reference changed')
    tests=data['wilcoxon_bh'];expected=set(itertools.product(spec['partitions'],methods[1:],range(1,51)))
    require_keys([(r['partition'],r['method'],_integer(r['round'])) for r in tests],expected)
    if len(tests)!=expected_comparison_count(len(methods)-1,len(spec['partitions']),50):raise ValueError('incorrect global comparison family')
    for row in tests:
        p,m,t=row['partition'],row['method'],_integer(row['round'])
        differences=np.asarray([trajectories[('method',p,m,s)][t]*100-trajectories[('method',p,reference,s)][t]*100 for s in seeds])
        if _integer(row['n_pairs'])!=len(seeds) or not np.array_equal(differences,json.loads(row['paired_differences_pp'])):
            raise ValueError('paired seed identity differences changed')
        statistic,pvalue=(0.,1.) if np.all(differences==0) else stats.wilcoxon(differences,alternative='two-sided',zero_method='wilcox',method='auto')
        if float(row['wilcoxon_statistic'])!=statistic or float(row['raw_p_value'])!=pvalue or float(row['mean_paired_accuracy_difference_pp'])!=float(differences.mean()):
            raise ValueError('independent paired Wilcoxon result differs')
    p=np.asarray([float(r['raw_p_value']) for r in tests]);q=benjamini_hochberg(p)
    if not np.array_equal(q,_historical_bh(p)):raise ValueError('BH differs from frozen helper')
    # The reference recomputation uses (p*m)/rank; the historical helper uses p*(m/rank).
    # Verify exact operation orders, not a newly invented tolerance.
    order=np.argsort(p,kind='stable');alternate=np.empty_like(p)
    alternate[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1])
    evidence=manifest['statistical_evidence']
    key=lambda r:(r['partition'],r['method'],_integer(r['round']))
    canonical=sorted(range(len(tests)),key=lambda i:key(tests[i]))
    if _text_hash([key(tests[i]) for i in canonical])!=evidence['identities_sha256']:raise ValueError('archived comparison identities changed')
    if _vector_hash([json.loads(tests[i]['paired_differences_pp']) for i in canonical])!=evidence['paired_differences_sha256']:raise ValueError('archived paired seed differences changed')
    if _vector_hash([float(tests[i]['wilcoxon_statistic']) for i in canonical])!=evidence['statistic_sha256']:raise ValueError('archived Wilcoxon statistic changed')
    if _text_hash([tests[i]['classification'] for i in canonical])!=evidence['classification_sha256']:raise ValueError('archived classification changed')
    historical_p=p.copy();lookup={key(r):i for i,r in enumerate(tests)}
    exceptions=evidence['adjacent_float64'];require_keys([key(r) for r in exceptions],{key(r) for r in exceptions})
    for item in exceptions:
        index=lookup[key(item)];old=float.fromhex(item['historical_p_hex'])
        if p[index]!=float.fromhex(item['current_p_hex']) or not _same_or_adjacent(old,p[index]):raise ValueError('frozen test evidence differs beyond one ULP')
        historical_p[index]=old
    if _vector_hash(historical_p[canonical])!=evidence['historical_p_sha256']:raise ValueError('archived p vector differs')
    historical_q=_historical_bh(historical_p)
    if _vector_hash(historical_q[canonical])!=evidence['historical_q_sha256']:raise ValueError('archived p/q do not match historical BH exactly')
    ulp_records=[]
    for index,row in enumerate(tests):
        if float(row['adjusted_q_value'])!=alternate[index] or not _same_or_adjacent(historical_q[index],q[index]):raise ValueError('joint BH adjusted values differ')
        if historical_p[index]!=p[index] or historical_q[index]!=q[index]:
            ulp_records.append(dict(partition=row['partition'],method=row['method'],round=_integer(row['round']),
                                    historical_p=float(historical_p[index]).hex(),current_p=float(p[index]).hex(),historical_q=float(historical_q[index]).hex(),current_q=float(q[index]).hex()))
        classification='unresolved' if q[index]>=.05 or float(row['mean_paired_accuracy_difference_pp'])==0 else ('better' if float(row['mean_paired_accuracy_difference_pp'])>0 else 'worse')
        if row['classification']!=classification:raise ValueError('volcano classification differs')
        if float(row['minus_log10_q'])!=float(-np.log10(max(alternate[index],np.finfo(float).tiny))):raise ValueError('volcano q axis differs')
    return dict(status='verified',distribution_runs=sum(k[0]=='distribution' for k in runs),method_runs=sum(k[0]=='method' for k in runs),
                distribution_rows=len(summaries),client_proportions=len(clients),comparisons=len(tests),paired_seeds=len(seeds),observations=len(points),
                reference_method=reference,differences=0,cross_environment_adjacent_float64=ulp_records,dp_sgd_semantics='N=1, zero communication; centralized optimizer-step index')


def _require_inputs(manifest):
    for row in manifest['required_inputs']:
        path=_vendor_root()/row['path']
        if not path.is_file():raise FileNotFoundError('missing frozen S15 input: '+row['path'])
        if path.stat().st_size!=row['bytes']:raise ValueError('frozen S15 input size mismatch')
        verify_hash(path,row['sha256'])


def _resolved_inputs(root,manifest):
    required={(root/r['path']).resolve():r for r in manifest['required_inputs']}
    result={}
    for suite,spec in manifest['suites'].items():
        cfg=root/spec['config'];raw=json.loads(cfg.read_text(encoding='utf-8'))
        base=json.loads((cfg.parent/raw['base_config']).read_text(encoding='utf-8'))
        path=(root/base['data_path']).resolve()
        if root not in path.parents or path not in required:raise ValueError('S15 input escaped exact vendored data path')
        if not path.is_file():raise FileNotFoundError('missing frozen S15 input: '+base['data_path'])
        if path.stat().st_size!=required[path]['bytes']:raise ValueError('frozen S15 input size mismatch')
        verify_hash(path,required[path]['sha256'])
        result[suite]=dict(path=path.relative_to(root).as_posix(),sha256=_hash(path))
    return result


def run(output: Path):
    output=Path(output).resolve();references=(release_root()/'reference_results').resolve()
    if output.exists() or output==references or references in output.parents:raise ValueError('S15 run requires a new directory outside reference_results')
    manifest=load_manifest();_require_inputs(manifest)
    output.mkdir(parents=True,exist_ok=False);runtime=output/'runtime'
    shutil.copytree(_vendor_root(),runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
    checked=_vendor_call('run',dict(output=str(output)),root=runtime)
    result=dict(checked,status='completed',archive_sha256=manifest['archive_sha256'],manifest_path='run_manifest.json')
    with (output/'run_manifest.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,sort_keys=True,indent=2)
    return result


def _execute_frozen_suites(root,output,loaded,manifest):
    """Execute the two frozen configs once, audit, then return actual artifacts."""
    import contextlib
    root,output=Path(root),Path(output)
    configs=(('distribution','configs/mnist67_spade_distribution_50round_30seed.json'),
             ('method','configs/spade_mnist_50round_server.json'))
    if set(manifest['suites'])!={'distribution','method'} or any(manifest['suites'][suite]['config']!=config for suite,config in configs):
        raise ValueError('S15 requires exactly the two frozen suite configs')
    with contextlib.redirect_stdout(sys.stderr):
        for suite,config in configs:
            loaded['paper_suite'].run_paper_suite(root/config,output/(suite+'_suite'))
        distribution=output/'distribution_suite';method=output/'method_suite'
        distribution_count=len(json.loads((distribution/'suite_manifest.json').read_text(encoding='utf-8'))['runs'])
        method_count=len(json.loads((method/'suite_manifest.json').read_text(encoding='utf-8'))['runs'])
        if (distribution_count,method_count)!=(150,840):raise ValueError('S15 actual suite counts must be 150/840')
        selected=loaded['mnist67_selected_three_figures']
        selected.audit_formal_results(distribution,method)
        selected.build_selected_three_figures(distribution,method,output/'paper_figures')
    curves,seeds=loaded['mnist67_method_volcano'].load_server_accuracy(method)
    records=loaded['mnist67_method_volcano'].build_method_volcano_records(curves,seeds)
    if len(records)!=1200:raise ValueError('S15 actual comparison count must be 1200')
    if not np.array_equal(benjamini_hochberg([r.raw_p_value for r in records]),[r.adjusted_q_value for r in records]):raise ValueError('new-run BH verification failed')
    files={p.relative_to(output).as_posix():_hash(p) for base in (distribution,method,output/'paper_figures') for p in sorted(base.rglob('*')) if p.is_file()}
    return dict(distribution_runs=distribution_count,method_runs=method_count,comparisons=len(records),files=files,
                distribution_n_clients=1000,method_n_clients=31,dp_sgd_n_clients=1,dp_sgd_communication_rounds=0,dp_sgd_axis='centralized_optimizer_step')


def _worker():
    import importlib
    request=json.load(sys.stdin);root=Path(request['root']).resolve();sys.path.insert(0,str(root/'src'));os.chdir(root)
    modules=['paper_suite','mnist67_selected_three_figures','mnist67_method_volcano','mnist67_paper_evidence']
    loaded={name:importlib.import_module('spade_dfl.'+name) for name in modules}
    paths=[Path(m.__file__).resolve() for name,m in sys.modules.items() if (name=='spade_dfl' or name.startswith('spade_dfl.')) and getattr(m,'__file__',None)]
    if any(root not in path.parents for path in paths):raise ValueError('S15 import escaped packaged runtime')
    action=request['action'];payload=request['payload']
    if action=='imports':result=sorted(p.relative_to(root).as_posix() for p in paths)
    elif action=='bh':result=loaded['mnist67_paper_evidence'].benjamini_hochberg(np.asarray(payload)).tolist()
    elif action=='inputs':result=_resolved_inputs(root,load_manifest())
    elif action=='golden':
        import ast
        import inspect
        manifest=load_manifest();_resolved_inputs(root,manifest)
        suite=payload['suite'];spec=manifest['suites'][suite];ps=loaded['paper_suite']
        config_path=root/spec['config'];raw=json.loads(config_path.read_text(encoding='utf-8'))
        base=ps.apply_suite_base_overrides(ps.load_config(config_path.parent/raw['base_config']),raw)
        parameters=ps.resolve_method_parameters(raw,config_path,expected_methods=raw['methods'])
        method='spade_dfl_private' if suite=='distribution' else 'spade_dfl';panel='iid';seed=raw['seeds'][0]
        if seed!=0 or method not in raw['methods'] or panel not in raw['partitions']:raise ValueError('fixed S15 golden identity absent')
        mode,alpha=ps._partition_spec(panel,task_type=base.task_type)
        config=ps.configure_suite_method(base,method,tuple(raw['seeds']),mode,alpha,protocol=raw.get('protocol'),override=parameters.get(method))
        tree=ast.parse(inspect.getsource(ps.run_paper_suite))
        salts=[n.args[0].elts[-1].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='SeedSequence']
        if len(salts)!=1:raise ValueError('ambiguous frozen partition RNG salt')
        dataset=ps.load_dataset_for_suite(base,raw)
        partition=ps.partition_labels(dataset.train_y,ps.suite_partition_client_count(method,config,base,protocol=raw.get('protocol')),mode,
                                      np.random.default_rng(np.random.SeedSequence([seed,ps.partition_seed_index(panel),salts[0]])),alpha=alpha,pathological_minor_fraction=base.pathological_minor_fraction)
        result_run=ps.run_method(method,config,dataset,partition,seed)
        out=Path(payload['output']);out.mkdir(parents=True,exist_ok=False)
        np.savez(out/'result.npz',states=result_run.states,**result_run.history)
        np.savez(out/'partition.npz',**{f'client_{i}':v for i,v in enumerate(partition.indices)},label_histograms=partition.label_histograms)
        np.savez(out/'preprocessing.npz',mean=dataset.pca.mean,components=dataset.pca.components,scale=dataset.pca.scale)
        convert=lambda value:value.tolist() if isinstance(value,np.ndarray) else value.item()
        for name,value in [('config',config.to_dict()),('metadata',result_run.metadata)]:
            (out/(name+'.json')).write_text(json.dumps(value,sort_keys=True,default=convert),encoding='utf-8')
        result=dict(suite=suite,seed=seed,partition=panel,method=method,queries=result_run.query_count,
                    dataset_sha256=dataset.sha256,states_shape=list(result_run.states.shape),observations=len(result_run.history['accuracy']))
    elif action=='run':
        output=Path(payload['output']);manifest=load_manifest();_resolved_inputs(root,manifest)
        result=_execute_frozen_suites(root,output,loaded,manifest)
    else:raise ValueError('unknown private S15 operation')
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':_worker()
