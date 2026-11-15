"""User-approved moderate/wide epsilon grid; no original algorithm edits."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import helpers as h

ROOT=Path(__file__).resolve().parent
TRUE=ROOT.parent/'table34_true_dimension_working'
PROBE=ROOT.parent/'table34_accuracy_floor_probe_working'
sys.path.insert(0,str(TRUE))
import experiment as e


def make_jobs(groups,settings,anchors,counts,phase):
    if phase not in ('validation','formal'):raise ValueError('Unknown phase')
    jobs=[]
    for group,g in groups.items():
        choice=settings[g['dataset']];anchor=anchors[group];m=counts[group]
        for n,ep in h.conditions():
            for seed in range(5 if phase=='validation' else 30):
                cfg=e.adapt(dict(anchor,n_clients=n,target_epsilon=ep,n_rounds=choice['rounds'],seeds=[seed]),g['dataset'],choice['retained'])
                q=n*cfg['n_rounds']*cfg['tau']*cfg['batch_size']+m*(cfg['n_rounds'] if cfg['vr_mode']=='full_refresh' else 1)
                task=f'{group}_n{n}_eps{ep:g}'
                jobs.append(dict(group=group,candidate=f"pc{choice['retained']}_k{choice['rounds']}",retained=choice['retained'],
                    seed=seed,cfg=cfg,budget=dict(actual=q,target=q,unused=0),reference=None,task=task,
                    out=str(ROOT/phase/task/f'seed{seed}')))
    return jobs


def freeze():
    old=e.plan('validation');sources=dict(old['sources']);probe=e.read(PROBE/'PLAN.json')
    for path,value in probe['sources'].items():assert e.sha(path)==value,path
    sources.update(probe['sources'])
    for task in dict.fromkeys(j['task'] for j in probe['jobs']):
        a=e.read(PROBE/'audit'/(task+'.json'));b=e.read(PROBE/'recheck'/(task+'.json'));assert a==b and a['verified']
        for row in b['rows']:
            for name,value in row['artifact_hashes'].items():
                path=Path(row['source_summary']).parent/name
                assert e.sha(path)==value; sources[str(path)]=value
    summary=e.read(PROBE/'SUMMARY.json');assert summary['gate_passed'] and summary['record_count']==95
    ranked=summary['ranked_by_minimum_accuracy'];winner=ranked[0]['candidate']
    matched=[r for r in summary['statistics'] if r['candidate']==winner]
    assert len(matched)==2 and min(r['mean_percent'] for r in matched)>=h.FLOORS['brain']
    settings={'mnist':dict(retained=10,rounds=10),'fashion':dict(retained=10,rounds=10),
              'alzheimer':dict(retained=5,rounds=25),'brain':dict(retained=matched[0]['dimension']//4,rounds=matched[0]['rounds'])}
    groups={name:g for ds in h.NAMES for name,g in old['groups'].items() if g['dataset']==ds}
    anchors={group:e.read(e.RING/'pilot'/group/'ring/seed0/config.json') for group in groups}
    assert settings['alzheimer']['retained']==e.read(TRUE/'SELECTIONS.json')['selected']['alzheimer']['retained']
    for ds in ('mnist','fashion','alzheimer'):
        assert all(a['n_rounds']==settings[ds]['rounds'] for g,a in anchors.items() if groups[g]['dataset']==ds)
    counts={}
    for group in groups:
        with np.load(e.s.PRE/'prepared'/group/'data.npz') as data:counts[group]=len(data['train_y'])
    for path in [ROOT/f for f in ('study.py','helpers.py','test_helpers.py','test_study.py','PROTOCOL_CN.md','IMPLEMENTATION_PLAN.md')]+[
        PROBE/'PLAN.json',PROBE/'SUMMARY.json',PROBE/'EXECUTION_RECORD.md',TRUE/'PLAN.json',TRUE/'SELECTIONS.json',TRUE/'MNIST_ZERO_SD_CHECK.json']:
        sources[str(path)]=e.sha(path)
    jobs=make_jobs(groups,settings,anchors,counts,'validation')
    assert len(jobs)==200 and not (ROOT/'PLAN.json').exists()
    e.save(ROOT/'PLAN.json',dict(groups=groups,settings=settings,anchors=anchors,sources=sources,jobs=jobs,
        phase='validation',seeds=list(range(5)),validation_only=True,previous_test_exposure=True,end_to_end_privacy_certified=False))
    print('FROZEN',len(jobs),'validation runs;',sum(j['budget']['actual'] for j in jobs),'queries',flush=True)


def plan(phase):
    p=e.read(ROOT/'PLAN.json')
    for path,value in p['sources'].items():assert e.sha(path)==value,path
    e.check_schema_diff()
    if phase=='formal':
        f=e.read(ROOT/'FORMAL_PLAN.json')
        assert f['validation_plan_sha256']==e.sha(ROOT/'PLAN.json')
        assert f['validation_summary_sha256']==e.sha(ROOT/'SUMMARY_validation.json')
        return f
    assert phase=='validation'
    return p


def worker(phase,task,audit=False):
    p=plan(phase);jobs=[j for j in p['jobs'] if j['task']==task];assert jobs
    bundle,base,state=e.prepare(jobs[0]['group'],jobs[0]['retained'],phase=='formal')
    e.s.release.ROOT=ROOT;rows=[];cache={}
    for j in jobs:
        out=Path(j['out'])
        if not audit:
            e.s.release.run_one(j,bundle,base,phase)
            e.save(out/'TRANSFORM.json',dict(state=state,train_sha256=e.array_hash(bundle.train_x),
                eval_sha256=e.array_hash(bundle.test_x),actual_model_dimension=j['cfg']['dimension'],labels_unchanged=True))
        row=e.verify(j,bundle,base,state,phase,cache)
        row['artifact_hashes']={name:e.sha(out/name) for name in ('raw.npz','config.json','summary.json','metadata.json',
                                                               'STARTED.json','TRANSFORM.json','partition.npz')}
        rows.append(row)
    dest=ROOT/(('recheck_' if audit else 'audit_')+phase)/(task+'.json');assert not dest.exists()
    e.save(dest,dict(verified=True,rows=rows));print('VERIFIED',phase,task,len(rows),flush=True)


def dispatch(phase,audit=False):
    p=plan(phase);tasks=list(dict.fromkeys(j['task'] for j in p['jobs']))
    tag=('recheck_' if audit else 'run_')+phase
    def child(task):
        log=ROOT/('logs_'+tag)/(task+'.log');log.parent.mkdir(exist_ok=True)
        env=os.environ.copy();env.pop('PYTHONUTF8',None)
        env.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
        env.update({key:'1' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
        args=[sys.executable,'-B',__file__,'worker','--phase',phase,'--task',task]+(['--audit'] if audit else [])
        with log.open('x',encoding='utf-8') as f:code=subprocess.run(args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        return dict(task=task,exit_code=code,log=str(log))
    pending=iter(tasks);results=[];failed=False
    with ThreadPoolExecutor(max_workers=4) as pool:
        active={pool.submit(child,next(pending)) for _ in range(min(4,len(tasks)))}
        while active:
            done,active=wait(active,return_when=FIRST_COMPLETED);finished=[f.result() for f in done];results+=finished
            failed=failed or any(r['exit_code'] for r in finished)
            for r in finished:print(len(results),'/',len(tasks),r,flush=True)
            if not failed:
                for _ in finished:
                    task=next(pending,None)
                    if task is not None:active.add(pool.submit(child,task))
    e.save(ROOT/f'DISPATCH_{tag}.json',results)
    assert not failed and len(results)==len(tasks),'Failures retained; do not retry automatically'


def collect(phase):
    p=plan(phase);rows=[]
    for task in dict.fromkeys(j['task'] for j in p['jobs']):
        first=e.read(ROOT/('audit_'+phase)/(task+'.json'));second=e.read(ROOT/('recheck_'+phase)/(task+'.json'))
        assert first==second and first['verified']
        for row in second['rows']:
            out=Path(row['source_summary']).parent
            for name,value in row['artifact_hashes'].items():assert e.sha(out/name)==value,(out,name)
        rows+=second['rows']
    key=lambda r:(r['group'],r['candidate'],r['n'],r['epsilon'],r['seed'])
    expected={(j['group'],j['candidate'],j['cfg']['n_clients'],j['cfg']['target_epsilon'],j['seed']) for j in p['jobs']}
    assert len(rows)==len(expected) and {key(r) for r in rows}==expected
    return p,rows


def summarize(phase):
    p,rows=collect(phase);cells=[]
    for ds in h.NAMES:
        folds=[g['fold'] for g in p['groups'].values() if g['dataset']==ds]
        for n,ep in h.conditions():
            rr=[r for r in rows if (r['dataset'],r['n'],r['epsilon'])==(ds,n,ep)]
            cells.append(dict(dataset=ds,n=n,epsilon=ep,**h.cell(rr,folds,p['seeds']),
                source_summaries=[r['source_summary'] for r in rr]))
    assert len(cells)==20
    q=h.qualification(cells)
    summary=dict(phase=phase,cells=cells,qualification=q,records=len(rows),
        queries=sum(r['queries'] for r in rows),trainer_seconds=sum(r['trainer_seconds'] for r in rows),
        settings=p['settings'],formal_table_ready=(phase=='formal' and q['passed']),
        previous_test_exposure=True,end_to_end_privacy_certified=False)
    e.save(ROOT/f'SUMMARY_{phase}.json',summary)
    flat=lambda rr:[{key:value for key,value in r.items() if not isinstance(value,(list,dict))} for r in rr]
    e.save_check(ROOT/f'{phase.upper()}_RUNS.csv',e.csv_text(flat(rows)))
    e.save_check(ROOT/f'CELL_RESULTS_{phase}.csv',e.csv_text(flat(cells)))
    lines=[f'# {"内部验证" if phase=="validation" else "正式测试"}结果','',
        f"总运行数：{len(rows)}；所有单元格准确率下限与非零SD显示检查结果：{q['passed']}。",'']
    for title,grid in [('固定ε32，客户端数量变化',[(30,32.),(50,32.),(100,32.)]),
                       ('固定N100，隐私预算变化',[(100,16.),(100,24.),(100,32.)])]:
        lines += ['## '+title,'','| 数据集 | '+' | '.join(f'N={n}, ε={ep:g}' for n,ep in grid)+' |',
                  '|---|'+'---:|'*3]
        for ds,name in h.NAMES.items():
            ss=[next(r for r in cells if (r['dataset'],r['n'],r['epsilon'])==(ds,n,ep)) for n,ep in grid]
            lines.append('| '+name+' | '+' | '.join(f"{r['mean_percent']:.2f} ± {r['sd_percent']:.2f}" for r in ss)+' |')
        lines += ['']
    lines += [f"单位%；{len(p['seeds'])}个固定运行种子的均值±样本SD，ddof=1；Alzheimer先在seed内合并5折correct/count。",
        'MNIST为6/7二分类，Fashion为0/1二分类，Alzheimer为临床表格二分类，Brain为四分类。',
        '最后一轮私有发布客户端模型的平均模型准确率，不使用最佳轮；两表共享条件使用完全同一组原始运行。',
        '固定本地batch，不是固定全网络计算预算；N越大查询量越大。没有强制趋势。',
        '隐私仅为条件化发布会计，预处理/分区/选参及全部运行的端到端隐私未证明。',
        '历史测试暴露仍存在；Alzheimer跨折全局选表示不是严格嵌套评估，本批应作为探索性实验披露。',
        f"总优化查询数：{summary['queries']}；训练器累计秒数：{summary['trainer_seconds']:.3f}（不是并行墙钟时间）。",'']
    if q['failures']:
        lines+=['## 未达标项','']+[f"- {r['dataset']} / N={r['n']} / ε={r['epsilon']:g}：{r['reason']}；{r}" for r in q['failures']]
    if phase=='validation':lines+=['','本文件不是正式测试结果，不可直接替换论文表3/4。未达标则不启动正式批次。']
    else:
        e.save_check(ROOT/'tables34_results.tex',h.tex(cells))
        lines+=['','检查结果为False时，表示至少一个单元格未通过准确率下限或非零显示样本标准差检查；实测值和已冻结的候选值选择流程保持不变。']
    e.save_check(ROOT/f'REPORT_{phase.upper()}_CN.md','\n'.join(lines)+'\n')
    print(json.dumps(dict(qualification=q,cells=flat(cells),records=len(rows)),ensure_ascii=False,indent=2),flush=True)


def freeze_formal():
    p,rows=collect('validation');summary=e.read(ROOT/'SUMMARY_validation.json')
    assert summary['qualification']['passed'] and summary['records']==200
    # Recompute the gate from the freshly verified records, not only its saved boolean.
    cells=[]
    for ds in h.NAMES:
        folds=[g['fold'] for g in p['groups'].values() if g['dataset']==ds]
        for n,ep in h.conditions():
            rr=[r for r in rows if (r['dataset'],r['n'],r['epsilon'])==(ds,n,ep)]
            cells.append(dict(dataset=ds,n=n,epsilon=ep,**h.cell(rr,folds,p['seeds'])))
    assert h.qualification(cells)['passed']
    for calculated,saved in zip(cells,summary['cells']):
        assert all(saved[key]==value for key,value in calculated.items())
    entries=e.read(e.s.CAL/'FORMAL_DATA.json')['entries'];counts={}
    for group in p['groups']:
        path=e.s.relocate(entries[group]['path']);assert e.sha(path)==entries[group]['sha256']
        with np.load(path) as data:counts[group]=len(data['train_y'])
    jobs=make_jobs(p['groups'],p['settings'],p['anchors'],counts,'formal')
    assert len(jobs)==1200 and not (ROOT/'FORMAL_PLAN.json').exists()
    e.save(ROOT/'FORMAL_PLAN.json',dict(groups=p['groups'],settings=p['settings'],jobs=jobs,seeds=list(range(30)),
        validation_plan_sha256=e.sha(ROOT/'PLAN.json'),validation_summary_sha256=e.sha(ROOT/'SUMMARY_validation.json'),
        phase='formal',selection_frozen_before_formal=True))
    print('FROZEN',len(jobs),'formal runs;',sum(j['budget']['actual'] for j in jobs),'queries',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run','recheck','worker','summarize','freeze_formal'])
    ap.add_argument('--phase',choices=['validation','formal'],default='validation');ap.add_argument('--task');ap.add_argument('--audit',action='store_true');a=ap.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='worker':worker(a.phase,a.task,a.audit)
    elif a.action=='summarize':summarize(a.phase)
    elif a.action=='freeze_formal':freeze_formal()
    else:dispatch(a.phase,a.action=='recheck')
