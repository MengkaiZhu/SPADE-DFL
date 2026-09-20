import copy
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest

from spade_dfl_release.communication import simulation as c


def test_exact_budget_schedule_and_cycle_scalar_accounting():
    rows=c.budget_schedule()
    assert [r['T'] for r in rows]==[1000,3375,10648,29791]
    assert [r['tau'] for r in rows]==[10,15,22,31]
    assert [r['rounds'] for r in rows]==[100,225,484,961]
    assert c.transmitted_scalars(20,20,29791)==23832800
    assert c.transmitted_scalars(20,20,961)==768800
    assert c.transmitted_scalars(3,2,7)==84


@pytest.mark.parametrize('args',[(True,20,1),(20,1.5,1),(2,20,1),(20,0,1),(20,20,0),(20,20,-1),(20,20,np.nan)])
def test_scalar_arguments_are_strict_positive_integers(args):
    with pytest.raises(ValueError):c.transmitted_scalars(*args)


def test_portable_exact_source_closure_and_imports(release_root):
    manifest=c.load_manifest()
    assert len(manifest['files'])==5
    assert {r['source_path'] for r in manifest['files']}=={'spade_simulation.py','run_experiment.py','results/communication_cost_per_seed.csv','results/communication_cost_summary.csv','results/experiment_metadata.json'}
    runtime,runner=c._runtime()
    for module in (runtime,runner):assert Path(module.__file__).resolve().is_relative_to(release_root/'src/spade_dfl_release/communication')
    for row in manifest['files']:
        data=(release_root/row['packaged_path']).read_bytes()
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']


@pytest.mark.parametrize('mutation',['missing','duplicate','source_path','package_path','hash'])
def test_source_lock_rejects_recomputed_manifest_mutations(monkeypatch,mutation):
    original=json.loads
    def read(data,*a,**k):
        value=original(data,*a,**k)
        if isinstance(value,dict) and 'plotting_sources' in value:
            if mutation=='missing':value['files'].pop()
            elif mutation=='duplicate':value['files'][-1]=dict(value['files'][0])
            elif mutation=='source_path':value['files'][0]['source_path']='make_figure.py'
            elif mutation=='package_path':value['files'][0]['packaged_path']='../external/spade_simulation.py'
            else:value['files'][0]['sha256']='0'*64
        return value
    monkeypatch.setattr(c.json,'loads',read)
    with pytest.raises(ValueError,match='closure'):c.load_manifest()


@pytest.mark.parametrize('mutation',['query_matched','diagnostic','n_seeds','plotting_segments'])
def test_complete_manifest_semantics_fail_closed(monkeypatch,mutation):
    original=json.loads
    def read(data,*args,**kwargs):
        value=original(data,*args,**kwargs)
        if isinstance(value,dict) and 'plotting_sources' in value:
            if mutation=='query_matched':value['semantics']['query_matched']=True
            elif mutation=='diagnostic':value['semantics']['diagnostic']='private'
            elif mutation=='n_seeds':value['n_seeds']=1
            else:value['plotting_sources'][0]['segments']=[]
        return value
    monkeypatch.setattr(c.json,'loads',read)
    with pytest.raises(ValueError,match='manifest semantic'):c.load_manifest()


def test_release_excludes_private_paths_and_superseded_content(release_root):
    import ast,re
    root=release_root/'src/spade_dfl_release/communication'
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.suffix!='.pyc'}
    assert actual=={'__init__.py','simulation.py','plotting.py','_frozen/__init__.py','_frozen/spade_simulation.py','_frozen/run_experiment.py'}
    for relative in actual:
        source=(root/relative).read_text(encoding='utf-8')
        assert not re.search(r'[A-Za-z]:[\\/]|/home/|/Users/',source)
        for node in ast.walk(ast.parse(source)):
            assert not (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='home')
        assert '.codex' not in source and 'style_comparison' not in source


def test_paired_dataset_seed_not_full_query_rng_matching():
    import ast
    runtime,_=c._runtime();tree=ast.parse(inspect.getsource(runtime.run_spade))
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='SeedSequence']
    assert len(calls)==1
    names={node.id for node in ast.walk(calls[0]) if isinstance(node,ast.Name)}
    assert 'seed' in names and 'tau' in names
    x,y=runtime.generate_synthetic_data(0);other_x,other_y=runtime.generate_synthetic_data(0)
    assert np.array_equal(x,other_x) and np.array_equal(y,other_y)
    changed_x,_=runtime.generate_synthetic_data(1)
    assert not np.array_equal(x,changed_x)


def test_verify_240_records_eight_summaries_and_four_adjacent_ci(reference_root):
    result=c.verify(reference_root)
    assert result['records']==240 and result['summaries']==8
    assert result['paired_dataset_seeds']==30 and result['differences']==0
    assert {(r['T'],r['tau'],r['field']) for r in result['adjacent_float64']}=={(1000,10,'stationarity_ci95_low'),(10648,22,'stationarity_ci95_low'),(29791,1,'stationarity_ci95_low'),(29791,31,'stationarity_ci95_low')}


@pytest.mark.parametrize('mutation',['missing','duplicate','seed','tau','rounds','schedule','query','one_direction','network_multiplier','criterion','negative','nan','config','noise','reduction'])
def test_scientific_mutations_fail_after_checksum_stage(monkeypatch,mutation):
    rows,summary,metadata=c._fixtures();rows=copy.deepcopy(rows);summary=copy.deepcopy(summary);metadata=copy.deepcopy(metadata)
    if mutation=='missing':rows.pop()
    elif mutation=='duplicate':rows[-1]=dict(rows[0])
    elif mutation=='seed':rows[0]['seed']=30
    elif mutation=='tau':rows[0]['tau']=2
    elif mutation=='rounds':rows[0]['rounds']+=1
    elif mutation=='schedule':rows[0]['schedule']='tau=T^(1/3)'
    elif mutation=='query':rows[0]['query_count']+=1
    elif mutation=='one_direction':rows[0]['transmitted_scalars']//=2
    elif mutation=='network_multiplier':rows[0]['transmitted_scalars']*=20
    elif mutation=='criterion':rows[0]['criterion']=rows[0]['stationarity']
    elif mutation=='negative':rows[0]['consensus']=-1
    elif mutation=='nan':rows[0]['stationarity']=float('nan')
    elif mutation=='config':metadata['config']['dimension']+=1
    elif mutation=='noise':metadata['query_noise']=1
    elif mutation=='reduction':summary[0]['communication_reduction_vs_tau1']=2
    with pytest.raises(ValueError):c._verify_records(rows,summary,metadata)


def test_no_public_scientific_overrides_and_existing_output_rejected(tmp_path):
    assert tuple(inspect.signature(c.run).parameters)==('output',)
    with pytest.raises(ValueError,match='new directory'):c.run(tmp_path)


@pytest.mark.parametrize('mutation',['two_ulp','other_field','other_identity'])
def test_ci_boundary_rejects_unapproved_differences(mutation):
    _,old,_=c._fixtures();current=copy.deepcopy(old)
    row=next(r for r in current if (r['T'],r['tau'])==((1000,1) if mutation=='other_identity' else (1000,10)))
    field='criterion_ci95_low' if mutation=='other_field' else 'stationarity_ci95_low'
    row[field]=float(np.nextafter(row[field],np.inf))
    if mutation=='two_ulp':row[field]=float(np.nextafter(row[field],np.inf))
    with pytest.raises(ValueError):c._cross_chain(old,current)


@pytest.mark.parametrize('mutation',['mean','normal','population'])
def test_wrong_summary_statistical_choices_fail(mutation):
    rows,summary,metadata=c._fixtures();row=summary[0];v=np.array([r['criterion'] for r in rows if (r['T'],r['tau'])==(row['T'],row['tau'])])
    if mutation=='mean':row['criterion_mean']=float(np.median(v))
    else:
        from scipy import stats
        critical=stats.norm.ppf(.975) if mutation=='normal' else stats.t.ppf(.975,29)
        row['criterion_ci95_low']=float(v.mean()-critical*v.std(ddof=0 if mutation=='population' else 1)/np.sqrt(len(v)))
    with pytest.raises(ValueError):c._verify_records(rows,summary,metadata)


def test_portable_plot_dynamic_savings_and_structure(tmp_path):
    from spade_dfl_release.communication import plotting as p
    _,rows,_=c._fixtures();fig=p.build_figure(rows)
    result=p.verify_figure(fig,rows)
    assert result['panels']==1 and result['alignment']=='not_applicable_single_panel'
    files=p.export_figure(fig,tmp_path,rows)
    assert {x.name for x in files}=={'fig_communication_saving_band'+suffix for suffix in ('.pdf','.svg','.png','.tiff')}
    original={text.get_text() for text in fig.axes[0].texts}
    for row in rows:
        if row['tau']!=1:row['transmitted_scalars']/=2
    changed=p.build_figure(rows)
    assert p.saving_label(rows) in {t.get_text() for t in changed.axes[0].texts}
    assert p.saving_label(rows) not in original
    p.plt.close(fig);p.plt.close(changed)


@pytest.mark.parametrize('mutation',['xlabel','units','shading','saving','line','fill'])
def test_figure_mutation_fails(mutation):
    from spade_dfl_release.communication import plotting as p
    _,rows,_=c._fixtures();fig=p.build_figure(rows);axis=fig.axes[0]
    if mutation=='xlabel':axis.set_xlabel('Queries')
    elif mutation=='units':axis.set_ylabel('Bytes / messages')
    elif mutation=='shading':next(t for t in axis.texts if 'Shaded' in t.get_text()).set_text('95% confidence interval')
    elif mutation=='saving':next(t for t in axis.texts if '% saved' in t.get_text()).set_text('incorrect saved')
    elif mutation=='line':axis.lines[0].set_ydata(axis.lines[0].get_ydata()*2)
    elif mutation=='fill':axis.collections[0].set_gid('confidence-band')
    with pytest.raises(ValueError):p.verify_figure(fig,rows)
    p.plt.close(fig)


def _author_sources():
    from spade_dfl_release.reference import Sources,resolve_source_index
    root=c.release_root();spec=resolve_source_index()
    if spec is None:pytest.skip('author source index is not configured; see docs/OPTIONAL_SOURCES.md')
    sources=Sources(spec,root/'reference_results/raw_manifests/source_manifest.json')
    if not sources.paths['communication_figure10'].is_dir():pytest.skip('author communication source unavailable')
    return sources


@pytest.mark.skipif(os.environ.get('SPADE_DFL_AUTHOR_ARCHIVE_QA')!='1',reason='opt-in read-only author source QA')
def test_author_sources_exact():
    sources=_author_sources()
    try:
        for row in c.load_manifest()['files']:
            data=sources.file('communication_figure10',row['source_path'])
            assert data==(c.release_root()/row['packaged_path']).read_bytes()
    finally:sources.close()


@pytest.mark.skipif(os.environ.get('SPADE_DFL_AUTHOR_ARCHIVE_QA')!='1',reason='opt-in author selected-plot artist golden')
def test_author_selected_plot_artist_golden():
    import types,sys
    from spade_dfl_release.communication import plotting as p
    try:p.font_manager.findfont('Times New Roman',fallback_to_default=False)
    except ValueError:pytest.skip('Times New Roman unavailable; author artist golden requires original font')
    sources=_author_sources();names=['communication_cost_figure','communication_cost_figure.make_style_comparison','_communication_original_plot'];previous={name:sys.modules.get(name) for name in names}
    try:
        package=types.ModuleType(names[0]);package.__path__=[str(sources.paths['communication_figure10'])];sys.modules[names[0]]=package
        for name,file in [(names[1],'make_style_comparison.py'),(names[2],'make_final_saving_band.py')]:
            module=types.ModuleType(name);module.__file__=str(sources.paths['communication_figure10']/file);sys.modules[name]=module
            exec(compile(sources.file('communication_figure10',file),file,'exec'),module.__dict__)
        _,rows,_=c._fixtures();first=sys.modules[names[2]].build_figure(rows);second=p.build_figure(rows)
        a,b=first.axes[0],second.axes[0]
        for figure in (first,second):figure.canvas.draw()
        assert (a.get_xlabel(),a.get_ylabel(),a.get_legend_handles_labels()[1])==(b.get_xlabel(),b.get_ylabel(),b.get_legend_handles_labels()[1])
        assert [(t.get_text(),t.get_position()) for t in a.texts]==[(t.get_text(),t.get_position()) for t in b.texts]
        assert [t.get_text() for t in a.get_xticklabels()+a.get_yticklabels()]==[t.get_text() for t in b.get_xticklabels()+b.get_yticklabels()]
        for x,y in zip(a.lines,b.lines):
            assert np.array_equal(x.get_xydata(),y.get_xydata())
            assert (x.get_color(),x.get_marker(),x.get_linestyle())==(y.get_color(),y.get_marker(),y.get_linestyle())
        assert np.array_equal(a.collections[0].get_paths()[0].vertices,b.collections[0].get_paths()[0].vertices)
        assert np.array_equal(a.collections[0].get_facecolors(),b.collections[0].get_facecolors())
        assert np.array_equal(first.get_size_inches(),second.get_size_inches())
        assert a.get_position().bounds==b.get_position().bounds
        p.plt.close(first);p.plt.close(second)
    finally:
        sources.close()
        for name,value in previous.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value


def _fake_complete_design(monkeypatch,failure=None):
    _,runner=c._runtime();rows,summary,metadata=c._fixtures();calls=[]
    def execute(output):
        calls.append(output)
        if failure=='training':raise RuntimeError('injected training failure')
        if failure=='records':rows.pop()
        if failure=='summaries':summary.pop()
        runner.write_csv(output/'communication_cost_per_seed.csv',rows);runner.write_csv(output/'communication_cost_summary.csv',summary)
        (output/'experiment_metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
        if failure!='copied_fixtures':
            cache=output/'cache';cache.mkdir()
            for row in rows:
                result=dict(row,config=metadata['config'])
                (cache/'seed_{:02d}_T_{}_tau_{}.json'.format(row['seed'],row['T'],row['tau'])).write_text(json.dumps(result),encoding='utf-8')
        return rows,summary
    monkeypatch.setattr(runner,'run_all',execute)
    if failure=='aggregate':monkeypatch.setattr(c,'_verify_records',lambda *a:(_ for _ in ()).throw(ValueError('injected aggregation failure')))
    from spade_dfl_release.communication import plotting as p
    if failure in ('figure','figure_check','export'):
        monkeypatch.setattr(p,{'figure':'build_figure','figure_check':'verify_figure','export':'export_figure'}[failure],lambda *a:(_ for _ in ()).throw(ValueError('injected plot failure')))
    return calls


def test_full_design_orchestration_and_actual_manifest(tmp_path,monkeypatch):
    calls=_fake_complete_design(monkeypatch);output=tmp_path/'run';result=c.run(output)
    assert calls==[output]
    assert result['status']=='completed' and result['records']==240 and result['summaries']==8
    assert result['executed_jobs']==240 and result['figure']['series']==2
    assert json.loads((output/'run_manifest.json').read_text(encoding='utf-8'))==result
    assert len(result['files'])>=247
    for name,digest in result['files'].items():assert hashlib.sha256((output/name).read_bytes()).hexdigest()==digest


@pytest.mark.parametrize('failure',['training','records','summaries','copied_fixtures','aggregate','figure','figure_check','export'])
def test_run_partial_failure_never_completed(tmp_path,monkeypatch,failure):
    calls=_fake_complete_design(monkeypatch,failure);output=tmp_path/'run'
    with pytest.raises((ValueError,RuntimeError)):c.run(output)
    assert calls==[output] and not (output/'run_manifest.json').exists()


@pytest.mark.skipif(os.environ.get('SPADE_DFL_RUN_GOLDEN')!='1',reason='opt-in eight full-budget original/packaged seed-0 goldens')
@pytest.mark.parametrize('total,tau',[(1000,1),(1000,10),(3375,1),(3375,15),(10648,1),(10648,22),(29791,1),(29791,31)])
def test_seed_zero_scientific_golden(total,tau):
    import types,sys
    sources=_author_sources();name='_communication_original_golden'
    module=types.ModuleType(name);sys.modules[name]=module
    try:
        data=sources.file('communication_figure10','spade_simulation.py');exec(compile(data,'spade_simulation.py','exec'),module.__dict__)
        original=module.run_spade(0,total,tau);runtime,runner=c._runtime();packaged=runtime.run_spade(0,total,tau)
        assert original==packaged
        frozen=next(r for r in c._fixtures()[0] if (r['seed'],r['T'],r['tau'])==(0,total,tau))
        packaged['schedule']=runner._schedule_label(total,tau)
        assert {k:packaged[k] for k in runner.RAW_FIELDS}==frozen
    finally:sources.close();sys.modules.pop(name,None)
