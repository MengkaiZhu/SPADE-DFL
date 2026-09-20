"""Recovered historical preparation functions; no training or data I/O on import."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from preprocessing import fit_scale, apply_scale

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT.parents[1]
sys.path.insert(0, str(PUBLIC / 'src'))
from spade_dfl_release.publication import historical_module_hashes
OLD = ROOT
S18 = ROOT
VENDOR_ROOT = PUBLIC / 'src/spade_dfl_release/vendor/sensitivity/table34_n30_eps8_working'
OUTPUT_ROOT = None

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str(value.dtype).encode('ascii'))
    h.update(json.dumps(list(value.shape)).encode('ascii'))
    h.update(value.tobytes(order='C'))
    return h.hexdigest()

def save_npz(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)

def split_training(labels, fraction, seed):
    # S18 calibration.py::split_training; independently cross-checked on preparation.
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    target = int(math.floor(len(labels) * fraction + .5))
    ideal = counts * fraction
    allocation = np.clip(np.floor(ideal).astype(int), 1, counts - 1)
    while allocation.sum() < target:
        eligible = allocation < counts - 1
        index = int(np.argmax(np.where(eligible, ideal - allocation, -np.inf)))
        allocation[index] += 1
    while allocation.sum() > target:
        eligible = allocation > 1
        index = int(np.argmax(np.where(eligible, allocation - ideal, -np.inf)))
        allocation[index] -= 1
    rng = np.random.default_rng(seed)
    inner, validation = [], []
    for label, count in zip(classes, allocation):
        shuffled = rng.permutation(np.flatnonzero(labels == label))
        validation.extend(shuffled[:count])
        inner.extend(shuffled[count:])
    return np.sort(np.array(inner, dtype=np.int64)), np.sort(np.array(validation, dtype=np.int64))

def verify_modules(job):
    package = VENDOR_ROOT / 'vendor' / job['dataset'] / 'src' / 'spade_dfl'
    hashes = historical_module_hashes(package)
    actual = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if actual != job['algorithm_source']['all_python_module_hash_map_sha256']:
        raise ValueError('Original source module hash mismatch')
    return package

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)

def image_training(path,classes):
    with np.load(path,allow_pickle=False) as data:
        labels=data['y_train'];mask=np.isin(labels,classes)
        x=data['x_train'][mask]
        y=np.where(labels[mask]==classes[1],1.,-1.)
    return x.reshape(len(x),-1).astype(np.float64)/255.,y

def norm_stats(x):
    norms=np.linalg.norm(x,axis=1)
    return dict(shape=list(x.shape),norm_quantiles=np.quantile(norms,[0,.25,.5,.75,1]).tolist(),
        rms_norm=float(np.sqrt(np.mean(norms**2))),mean_square_columns=np.mean(x*x,axis=0).tolist())

def build_group(group):
    base=read(OLD/'configs'/f'{group}_n30_eps8_seed0.json');ds=base['dataset']
    package=verify_modules(base);sys.path.insert(0,str(package.parent))
    from spade_dfl.data import PCAState
    source=dict(base['training_data']);source['path']=str(ROOT/source['path']);assert sha(source['path'])==source['sha256']
    with np.load(source['path']) as cache:
        old_y=cache[source['label_key']]
        outer_cache_x=cache[source['feature_key']] if ds!='brain' else None
    manifest=read(S18/'data/data_manifest.json')['datasets']
    sources={source['path']:source['sha256']}
    if ds in ('mnist','fashion'):
        raw=ROOT/'inputs/raw'/('mnist.npz' if ds=='mnist' else 'fashion-mnist.npz')
        assert sha(raw)==manifest[ds]['source_sha256'];sources[str(raw)]=sha(raw)
        x,y=image_training(raw,[6,7] if ds=='mnist' else [0,1])
    elif ds=='brain':
        raw=Path(source['path'])
        with np.load(raw) as d:x=d['features'].astype(np.float64);y=d['labels']
        assert array_hash(y)==source['training_labels_sha256']
    else:
        import spade_dfl.alzheimer_data as az
        raw=ROOT/'inputs/raw/2、Alzheimer disease dataset.zip'
        entry=manifest[group];assert sha(raw)==entry['source_sha256'];sources[str(raw)]=sha(raw)
        rows,digest=az.read_alzheimer_zip(raw);continuous,binary,categories,labels=az._rows_to_arrays(rows)
        rng=np.random.default_rng(entry['preprocessing']['split_seed']);fold=base['fold']
        hold=[]
        for label in (-1.,1.):
            hold.extend(np.array_split(rng.permutation(np.flatnonzero(labels==label)),entry['preprocessing']['n_folds'])[fold])
        outer=np.setdiff1d(np.arange(len(labels)),np.sort(hold),assume_unique=True)
        c,b,cat=continuous[outer],binary[outer],categories[outer];y=labels[outer]
        original_state=az._fit_preprocessor(c)
        np.testing.assert_allclose(az._transform(c,b,cat,original_state),outer_cache_x,rtol=1e-13,atol=1e-13)
    assert np.array_equal(y,old_y)
    inner,valid=split_training(y,base['holdout']['fraction'],base['holdout']['seed'])
    assert array_hash(inner)==source['inner_indices_sha256']
    assert array_hash(valid)==source['validation_indices_sha256']
    state_info={};arrays=dict(train_y=y[inner],validation_y=y[valid],inner_indices=inner,validation_indices=valid)
    if ds=='alzheimer':
        state=az._fit_preprocessor(c[inner])
        train=az._transform(c[inner],b[inner],cat[inner],state)
        validation=az._transform(c[valid],b[valid],cat[valid],state)
        pca=az._identity_pca()
        arrays.update(continuous_mean=state.continuous_mean,continuous_scale=state.continuous_scale,outer_indices=outer)
        state_info['continuous_fit_scope']='inner_train_only'
        candidates={'baseline':(train,validation)}
        gs=fit_scale(train,'group_rms',continuous=len(az.CONTINUOUS_COLUMNS))
        candidates['group_balance']=(apply_scale(train,gs),apply_scale(validation,gs));state_info['group_balance']=gs
    else:
        if ds=='brain':
            from spade_dfl.brain_tumor_resnet18_data import fit_training_pca
            pca=fit_training_pca(x[inner],10)
        else:
            from spade_dfl.data import fit_pca_train_only
            pca=fit_pca_train_only(x[inner],10)
        assert np.allclose(pca.mean,x[inner].mean(axis=0))
        train_scores=(x[inner]-pca.mean)@pca.components.T
        validation_scores=(x[valid]-pca.mean)@pca.components.T
        train=train_scores/pca.scale;validation=validation_scores/pca.scale
        cs=fit_scale(train_scores,'pca_common');rs=fit_scale(train,'row_rms')
        candidates={'baseline':(train,validation),
            'common_scale':(apply_scale(train_scores,cs),apply_scale(validation_scores,cs)),
            'row_norm':(apply_scale(train,rs),apply_scale(validation,rs))}
        state_info.update(common_scale=cs,row_norm=rs,pca_score_sd=train_scores.std(axis=0).tolist(),
            retained_variance_fraction=float(np.sum(train_scores.var(axis=0))/np.sum(x[inner].var(axis=0))))
        np.testing.assert_allclose(train.std(axis=0),1.,rtol=1e-12)
    arrays.update(pca_mean=pca.mean,pca_components=pca.components,pca_scale=pca.scale)
    for candidate,(tx,vx) in candidates.items():
        assert tx.shape==train.shape and vx.shape==validation.shape and np.isfinite(tx).all() and np.isfinite(vx).all()
        arrays['train_x_'+candidate]=tx;arrays['validation_x_'+candidate]=vx
    metadata=read(OLD/'prepared'/base['job_id']/'bundle_metadata.json')
    info=dict(group=group,dataset=ds,fold=base['fold'],fit_scope='inner_training_only',
        candidates=list(candidates),states=state_info,training_stats={k:norm_stats(v[0]) for k,v in candidates.items()},
        sources=sources,outer_test_arrays_used=False,base=str(OLD/'configs'/f'{group}_n30_eps8_seed0.json'),
        bundle_metadata=metadata,arrays={k:array_hash(v) for k,v in arrays.items()},
        privacy='conditional release accountant only; data-dependent preprocessing not certified end-to-end')
    return arrays,info,base

def prepare_group(group):
    out=OUTPUT_ROOT/group
    if out.exists():raise FileExistsError('Never overwrite prepared data')
    arrays,info,_=build_group(group);out.mkdir(parents=True)
    save_npz(out/'data.npz',**arrays);info['data_sha256']=sha(out/'data.npz');save(out/'metadata.json',info)
    print('PREPARED',group,info['states'],flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', required=True, choices=sorted(p.stem.removesuffix('_n30_eps8_seed0') for p in (ROOT/'configs').glob('*.json')))
    parser.add_argument('--output', type=Path, help='New output root; preparation writes group/data.npz and metadata.json.')
    parser.add_argument('--check-code', action='store_true', help='Only verify preserved module hashes; no data loading or preparation.')
    args = parser.parse_args()
    if args.check_code:
        verify_modules(read(ROOT/'configs'/f'{args.group}_n30_eps8_seed0.json'))
        print('Preserved module hashes match; preparation was not executed.')
        return
    if args.output is None:
        parser.error('--output is required for preparation')
    global OUTPUT_ROOT
    OUTPUT_ROOT = args.output.resolve()
    if OUTPUT_ROOT.exists():
        raise FileExistsError('Use a new output directory')
    if PUBLIC == OUTPUT_ROOT or PUBLIC in OUTPUT_ROOT.parents:
        raise ValueError('Write reconstructed data outside the release package')
    prepare_group(args.group)

if __name__ == '__main__':
    main()
