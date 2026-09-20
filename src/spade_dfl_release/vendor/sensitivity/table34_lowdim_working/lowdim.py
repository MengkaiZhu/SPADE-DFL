"""Label-free train-only dimensionality probes."""
import numpy as np

def transform(train,valid,kind,k):
    x=np.asarray(train,dtype=np.float64);v=np.asarray(valid,dtype=np.float64)
    if x.ndim!=2 or v.ndim!=2 or min(x.shape)==0 or min(v.shape)==0 or x.shape[1]!=v.shape[1]:
        raise ValueError('Compatible nonempty matrices required')
    if not np.isfinite(x).all() or not np.isfinite(v).all():raise ValueError('Nonfinite features')
    if kind=='baseline':return x.copy(),v.copy(),dict(kind=kind,input_dimension=x.shape[1])
    available=x.shape[1]-(kind=='clinical')
    if not isinstance(k,int) or not 1<=k<=available:raise ValueError('Invalid retained dimension')
    if kind=='image':return x[:,:k].copy(),v[:,:k].copy(),dict(kind=kind,retained=k,input_dimension=x.shape[1],refit=False)
    if kind!='clinical':raise ValueError('Unknown representation')
    if not np.array_equal(x[:,-1],np.ones(len(x))) or not np.array_equal(v[:,-1],np.ones(len(v))):
        raise ValueError('Last column must be the original intercept')
    mean=x[:,:-1].mean(axis=0)
    _,_,vt=np.linalg.svd(x[:,:-1]-mean,full_matrices=False)
    components=vt[:k].copy()
    for row in components:
        if row[np.argmax(np.abs(row))]<0:row*=-1
    def project(a):return np.column_stack(((a[:,:-1]-mean)@components.T,np.ones(len(a))))
    return project(x),project(v),dict(kind=kind,retained=k,mean=mean.tolist(),components=components.tolist(),fit_scope='inner_train_only',rescaled=False)

def config_for(cfg,width):
    if width<=0:raise ValueError('Positive width required')
    return dict(cfg,feature_dimension=width,dimension=width*(cfg['n_classes'] if cfg.get('task_type')=='multiclass' else 1))

