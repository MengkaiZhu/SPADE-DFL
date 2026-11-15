"""Actual feature/model dimension adaptation; no optimizer or release edits."""
from dataclasses import fields, replace
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'table34_lowdim_working'))
from lowdim import transform as transform_arrays

def adapt(cfg,dataset,k):
    if k is None:return dict(cfg)
    if isinstance(k,bool) or not isinstance(k,int) or k<=0:raise ValueError('Positive integer dimension required')
    width=k+(dataset=='alzheimer')
    result=dict(cfg,dimension=width*(4 if dataset=='brain' else 1))
    if dataset in ('brain','alzheimer'):result['feature_dimension']=width
    return result

def load_brain_config():
    name='brain_config_working'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name].ExperimentConfig

def activate_brain_config():
    load_brain_config()
    sys.modules['spade_dfl.config']=sys.modules['brain_config_working']

def transform_bundle(bundle,dataset,k):
    tx,vx,state=transform_arrays(bundle.train_x,bundle.test_x,
        'baseline' if k is None else ('clinical' if dataset=='alzheimer' else 'image'),k)
    if k is None:return bundle,state
    updates=dict(train_x=tx,test_x=vx,model_dimension=tx.shape[1]*(4 if dataset=='brain' else 1),pca_dimension=k)
    accepted={f.name for f in fields(bundle)}
    return replace(bundle,**{key:value for key,value in updates.items() if key in accepted}),state
