"""Historical scaling functions recovered without mathematical changes."""
import numpy as np

def matrix(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or min(x.shape)==0 or not np.isfinite(x).all():
        raise ValueError('Nonempty finite matrix required')
    return x

def fit_scale(train,kind,continuous=15):
    x=matrix(train);d=x.shape[1]
    s=dict(kind=kind,dimension=d)
    if kind=='identity':return s
    if kind in ('pca_white','pca_common'):
        sd=x.std(axis=0)
        scale=np.where(sd>1e-12,sd,1.) if kind=='pca_white' else np.full(d,np.sqrt(np.mean(sd**2)))
        if not (scale>0).all():raise ValueError('Degenerate PCA scale')
        s['scale']=scale.tolist()
    elif kind=='row_rms':
        target=float(np.sqrt(np.mean(np.sum(x*x,axis=1))))
        if target<=0:raise ValueError('Degenerate row norm')
        s['target']=target
    elif kind=='group_rms':
        if not 0<continuous<d-1 or not np.array_equal(x[:,-1],np.ones(len(x))):
            raise ValueError('Continuous and encoded groups with last intercept required')
        first=float(np.mean(x[:,:continuous]**2));second=float(np.mean(x[:,continuous:-1]**2))
        if min(first,second)<=0:raise ValueError('Degenerate group energy')
        weights=np.ones(d);weights[continuous:-1]=np.sqrt(first/second)
        correction=np.sqrt(np.sum(x[:,:-1]**2)/np.sum((x[:,:-1]*weights[:-1])**2))
        weights[:-1]*=correction;s['weights']=weights.tolist();s['continuous']=continuous
    else:raise ValueError('Unknown scale variant')
    return s

def apply_scale(values,state):
    x=matrix(values)
    if x.shape[1]!=state['dimension']:raise ValueError('Feature dimension changed')
    kind=state['kind']
    if kind=='identity':return x.copy()
    if kind in ('pca_white','pca_common'):return x/np.asarray(state['scale'])
    if kind=='group_rms':return x*np.asarray(state['weights'])
    if kind=='row_rms':
        norms=np.linalg.norm(x,axis=1,keepdims=True)
        return x*state['target']/np.where(norms>0,norms,1.)
    raise ValueError('Unknown scale variant')

