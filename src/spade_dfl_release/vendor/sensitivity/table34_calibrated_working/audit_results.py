"""Independent result recomputation and mechanical CSV/TeX export; no trainer calls."""
import csv
import json
from pathlib import Path
import numpy as np
from study_runtime import ROOT,OLD,read,sha,array_hash,check_budget,verify_modules,plan,conditions,select_candidate


def aggregate(rows, expected):
    if len(rows)!=len(expected) or {(r['fold'],r['seed']) for r in rows}!=expected:
        raise ValueError('Incomplete or duplicate fold-seed grid')
    values=np.array([r['accuracy'] for r in rows])
    if not np.isfinite(values).all() or (values<0).any() or (values>1).any():
        raise ValueError('Invalid accuracy; do not drop a failed run')
    return dict(mean_pct=float(values.mean()*100),sd_pct=float(values.std(ddof=0)*100),count=len(rows))


def tex_cell(cell):
    if cell['mean_pct'] is None:
        if cell.get('status')!='blocked_partition':raise ValueError('Undocumented missing cell')
        return 'N/A'
    return rf"${cell['mean_pct']:.2f}\pm{cell['sd_pct']:.2f}$"


def prediction(cfg,states,x):
    theta=states.mean(axis=0)
    if cfg['objective_model']=='paper_sigmoid':return np.where(x@theta>0,-1.,1.)
    if cfg['objective_model']=='lt_admm_nonconvex_logistic':return np.where(x@theta>=0,1.,-1.)
    assert cfg['objective_model']=='multiclass_softmax'
    return np.argmax(x@theta.reshape(cfg['feature_dimension'],cfg['n_classes']),axis=1)


def check_run(path,data):
    r=read(path/'summary.json');cfg=read(path/'config.json');start=read(path/'STARTED.json');meta=read(path/'metadata.json')
    assert r['status']=='completed' and not (path/'FAILED.json').exists()
    assert sha(path/'raw.npz')==r['raw_sha256'] and sha(path/'metadata.json')==r['metadata_sha256']
    assert sha(path/'config.json')==r['config_sha256']==start['config_sha256']
    tx,ty,ex,ey=data
    for key,value in [('train_x',tx),('train_y',ty),('eval_x',ex),('eval_y',ey)]:
        assert array_hash(value)==start[key+'_sha256']
    with np.load(path/'raw.npz') as raw,np.load(path/'partition.npz') as parts:
        assert all(np.isfinite(raw[k]).all() for k in raw.files)
        assert raw['states'].shape==(cfg['n_clients'],cfg['dimension']) and len(raw['accuracy'])==51
        predicted=prediction(cfg,raw['states'],ex)
        correct=int(np.count_nonzero(predicted==ey))
        balanced=float(np.mean([np.mean(predicted[ey==v]==v) for v in np.unique(ey)]))
        assert balanced==r['balanced']
        assert correct/len(ey)==r['accuracy']==raw['accuracy'][-1]
        assert np.array_equal(raw['accuracy'][1:],raw['released_proxy_accuracy'])
        assert raw['loss'][-1]==r['loss']
        assert np.array_equal(np.sort(np.concatenate([parts[f'client_{i}'] for i in range(cfg['n_clients'])])),np.arange(len(ty)))
        assert [array_hash(parts[f'client_{i}']) for i in range(cfg['n_clients'])]==start['partition_hashes']
        assert [len(parts[f'client_{i}']) for i in range(cfg['n_clients'])]==meta['client_sample_counts']
        assert array_hash(parts['adjacency'])==start['adjacency_sha256']
        assert np.array_equal(parts['adjacency'],parts['adjacency'].T)
        edges=int(parts['adjacency'].sum())
    eps=check_budget(meta['clip_norm_schedule'],meta['release_noise_std_schedule'],cfg['target_delta'],cfg['target_epsilon'])
    assert np.allclose(meta['clip_norm_schedule'],cfg['clip_norm']*np.arange(1.,51.)**(-cfg['clip_exponent']),rtol=1e-14,atol=0)
    assert meta['stage']==cfg['stage']=='spade_dfl_private'
    assert np.isclose(eps,r['epsilon']) and np.isclose(eps,meta['epsilon'])
    assert r['queries']==len(ty)+cfg['n_clients']*cfg['tau']*cfg['batch_size']*50
    return r,dict(path=str(path),passed=True,raw_sha256=r['raw_sha256'],summary_sha256=sha(path/'summary.json'),
        correct=correct,eval_count=len(ey),epsilon_recalculated=eps,directed_edges=edges)


def verify_all():
    p=plan();formal=read(ROOT/'FORMAL_PLAN.json');rows=[];checks=[];calibration=[];calibration_checks=[]
    blocked=read(ROOT/'BLOCKED_CONDITIONS.json');blocked_jobs=[]
    assert blocked['formal_plan_sha256']==sha(ROOT/'FORMAL_PLAN.json') and len(blocked['tasks'])==10
    for task,record in blocked['tasks'].items():
        assert sha(record['log'])==record['log_sha256']
        assert record['group'].startswith('alzheimer_fold') and record['n'] in [50,100]
        assert not list((ROOT/'formal'/task).glob('seed*/STARTED.json'))
    assert sha(ROOT/'formal_runtime.py')==formal['code_sha256']
    for group,g in p['groups'].items():
        base=read(OLD/g['base_config']);verify_modules(base)
        with np.load(Path(g['prepared'])/'data.npz') as data:
            arrays=[data[k] for k in ['train_x','train_y','validation_x','validation_y']]
            for phase in ['calibration','extension']:
                paths=list((ROOT/phase/group).glob('*/seed*/summary.json'));assert len(paths)==45
                for path in paths:
                    r,c=check_run(path.parent,arrays);calibration.append(r);calibration_checks.append(c)
        if group=='brain_fold0':
            with np.load(ROOT/'formal_data/brain_inner.npz') as data:
                arrays=[data[k] for k in ['train_x','train_y','validation_x','validation_y']]
                paths=list((ROOT/'brain_server_calibration').glob('*/seed*/summary.json'));assert len(paths)==90
                for path in paths:
                    r,c=check_run(path.parent,arrays);calibration.append(r);calibration_checks.append(c)
        fg=formal['groups'][group];assert sha(fg['selection_path'])==fg['selection_sha256']
        chosen=read(fg['selection_path']);best=select_candidate(chosen['rows'])
        assert best=={k:v for k,v in chosen['selected'].items() if k!='fields'} and not chosen['test_used']
        with np.load(fg['data']['path']) as data:
            assert sha(fg['data']['path'])==fg['data']['sha256']
            arrays=[data[k] for k in ['train_x','train_y','test_x','test_y']]
            jobs=[j for j in formal['jobs'] if j['group']==group];assert len(jobs)==210
            shared_inputs={}
            for job in jobs:
                if job['task'] in blocked['tasks']:
                    blocked_jobs.append(job)
                    continue
                path=ROOT/'formal'/job['task']/f"seed{job['seed']}"
                cfg=read(path/'config.json');assert sha(job['config'])==job['config_sha256'] and cfg==read(job['config'])
                assert cfg['clip_norm']==fg['chosen']['fields']['clip'] and cfg['clip_exponent']==fg['chosen']['fields']['exponent']
                r,c=check_run(path,arrays)
                assert r['identity']['formal_plan_sha256']==sha(ROOT/'FORMAL_PLAN.json')
                start=read(path/'STARTED.json')
                key=(job['n'],job['seed'])
                value=(start['partition_hashes'],start['adjacency_sha256'])
                if key in shared_inputs:assert shared_inputs[key]==value
                else:shared_inputs[key]=value
                rows.append(dict(dataset=g['dataset'],fold=g['fold'],n=job['n'],epsilon=job['epsilon'],seed=job['seed'],
                    accuracy=r['accuracy'],accuracy_pct=100*r['accuracy'],final_loss=r['loss'],queries=r['queries'],
                    clip_norm=cfg['clip_norm'],clip_exponent=cfg['clip_exponent'],mean_clip_fraction=r['mean_clip_fraction'],
                    source_summary=str(path/'summary.json'),source_raw=str(path/'raw.npz'),source_config=str(path/'config.json')))
                checks.append(c)
    assert len(rows)==1380 and len(blocked_jobs)==300 and len(calibration)==810
    assert len(list((ROOT/'formal').rglob('STARTED.json')))==1380 and not list((ROOT/'formal').rglob('FAILED.json'))
    cells=[]
    for dataset in ['mnist','fashion','alzheimer','brain']:
        expected={(f,s) for f in range(5 if dataset=='alzheimer' else 1) for s in range(30)}
        for n,epsilon in conditions():
            selected=[r for r in rows if r['dataset']==dataset and r['n']==n and r['epsilon']==epsilon]
            if not selected:
                assert dataset=='alzheimer' and n in [50,100] and epsilon==8
                assert len([j for j in blocked_jobs if j['n']==n])==150
                stat=dict(mean_pct=None,sd_pct=None,count=0,status='blocked_partition')
            else:stat=aggregate(selected,expected)
            cells.append(dict(dataset=dataset,n=n,epsilon=epsilon,**stat,source_summaries=[r['source_summary'] for r in selected]))
    audit=dict(status='VERIFIED_1380_AVAILABLE_FORMAL_AND_810_CALIBRATION_300_NOT_RUN',formal_runs=1380,calibration_runs=810,
        planned_formal_runs=1680,blocked_unrun_jobs=300,blocked_condition_count=2,blocked_evidence_sha256=sha(ROOT/'BLOCKED_CONDITIONS.json'),
        additional_training_in_verifier=False,all_final_predictions_recomputed=True,population_sd_ddof=0,
        formal_total_queries=sum(r['queries'] for r in rows),calibration_total_queries=sum(r['queries'] for r in calibration),
        formal_checks=checks,calibration_checks=calibration_checks,source_files_unchanged=True,
        no_monotonicity_enforced=True,shared_reference_single_source=True)
    return rows,cells,audit


def save_check(path,text):
    if path.exists():
        with path.open(encoding='utf-8',newline='') as f:assert f.read()==text
    else:
        with path.open('x',encoding='utf-8',newline='') as f:f.write(text)


def csv_text(rows):
    from io import StringIO
    stream=StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return stream.getvalue()


def table_tex(cells,kind):
    names={'mnist':r'MNIST (6/7)','fashion':r'Fashion-MNIST (0/1)','alzheimer':r'Alzheimer (clinical)','brain':r'Brain Tumor MRI'}
    is_clients=kind=='clients'
    caption=(r'Test accuracy (\%) versus the number of clients at $\epsilon=8$.' if is_clients else r'Test accuracy (\%) versus the privacy budget at $N=30$.')
    label='tab:spade_clients' if is_clients else 'tab:spade_privacy'
    headers=[rf'$N={n}$' for n in [10,30,50,100]] if is_clients else [rf'$\epsilon={e:g}$' for e in [.5,2,8,32]]
    conditions_used=[(n,8.) for n in [10,30,50,100]] if is_clients else [(30,e) for e in [.5,2,8,32]]
    out=[r'\begin{table*}[t]',r'\centering',r'\caption{'+caption+r'}',r'\label{'+label+r'}',r'\small',
        r'\begin{tabular}{lcccc}',r'\toprule','Dataset & '+' & '.join(headers)+r' \\',r'\midrule']
    for ds,name in names.items():
        cs=[next(c for c in cells if (c['dataset'],c['n'],c['epsilon'])==(ds,n,e)) for n,e in conditions_used]
        out.append(name+' & '+' & '.join(tex_cell(c) for c in cs)+r' \\')
    out += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip',r'\begin{minipage}{0.98\textwidth}',r'\footnotesize',
        r'Available entries are mean $\pm$ population standard deviation ($\mathrm{ddof}=0$) of final-round accuracy after 50 rounds: 30 seeds (0--29), or 5 folds $\times$ 30 seeds for Alzheimer. The latter describes fold--seed variability, not 150 independent samples.',
        r'Dirichlet $\alpha=0.3$ and $\delta=10^{-5}$ were fixed; MNIST/Fashion used balanced-capacity partitions, and Alzheimer/Brain used the original unbalanced partitions. The original clipped-update private-release branch was used; clipping parameters were selected on training holdouts and then fixed across all $N$ and $\epsilon$ within each dataset/fold.',
        r'The $(N,\epsilon)=(30,8)$ entries in the two sensitivity tables use the same runs. These are new finite-budget experiments, not a relabelling of Table~II, whose source runs used 31 clients without the private-release mechanism.',
        (r'N/A denotes an unavailable complete seed grid: the original Alzheimer partition routine exhausted its 100 attempts to produce nonempty clients at the first seed for $N=50$ and $N=100$. No successful-seed subset was substituted.' if is_clients else r'The privacy budget applies to the original release accounting conditional on fixed preprocessing and partitioning, not to the combined disclosure of calibration and all experiment outputs.'),
        r'\end{minipage}',r'\end{table*}']
    return '\n'.join(out)+'\n'


if __name__=='__main__':
    rows,cells,audit=verify_all()
    save_check(ROOT/'FORMAL_RESULTS.csv',csv_text(rows))
    calibration_rows=[]
    for c in audit['calibration_checks']:
        r=read(Path(c['path'])/'summary.json');cfg=read(Path(c['path'])/'config.json')
        calibration_rows.append(dict(phase=r['identity']['phase'],group=r['identity']['group'],candidate=r['candidate'],seed=r['seed'],
            clip_norm=cfg['clip_norm'],clip_exponent=cfg['clip_exponent'],accuracy_pct=r['accuracy']*100,
            balanced_accuracy_pct=r['balanced']*100,final_loss=r['loss'],source_summary=str(Path(c['path'])/'summary.json')))
    save_check(ROOT/'CALIBRATION_RESULTS.csv',csv_text(calibration_rows))
    save_check(ROOT/'TABLE_CELLS.json',json.dumps(cells,ensure_ascii=False,indent=2)+'\n')
    save_check(ROOT/'FINAL_VERIFICATION.json',json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    save_check(ROOT/'table3_clients.tex',table_tex(cells,'clients'))
    save_check(ROOT/'table4_privacy.tex',table_tex(cells,'privacy'))
    provenance=[]
    for table,cs in [('Table3',[(n,8.) for n in [10,30,50,100]]),('Table4',[(30,e) for e in [.5,2,8,32]])]:
        for ds in ['mnist','fashion','alzheimer','brain']:
            for n,e in cs:
                c=next(c for c in cells if (c['dataset'],c['n'],c['epsilon'])==(ds,n,e))
                provenance.append(dict(table=table,dataset=ds,n=n,epsilon=e,tex=tex_cell(c),count=c['count'],
                    statistic='final_round_50_mean_population_SD',source_summaries=json.dumps(c['source_summaries'],ensure_ascii=False)))
    save_check(ROOT/'TABLE_DATA_PROVENANCE.csv',csv_text(provenance))
    print(json.dumps({k:v for k,v in audit.items() if not k.endswith('_checks')},indent=2))
    print(json.dumps([{k:v for k,v in c.items() if k!='source_summaries'} for c in cells],indent=2))
