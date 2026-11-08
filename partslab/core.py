"""Nonnegative factors with inspectable cyclic coordinate minimization.

All floating-point certificates below are numerical diagnostics, not exact bounds.
"""
import numpy as np


def matrix(x,name='matrix',nonnegative=True):
    a=np.asarray(x,dtype=float)
    if a.ndim!=2 or not a.size or not np.isfinite(a).all() or (nonnegative and (a<0).any()):
        raise ValueError(name+' must be finite, nonempty and '+('nonnegative' if nonnegative else 'real'))
    return a


def objective(x,a,b):
    r=x-a@b
    return float(np.sum(r*r)/2)


def gradients(x,a,b):
    r=a@b-x
    return r@b.T,a.T@r


def projected_gradient(a,g):
    return np.where((a>0)|(g<0),g,0.)


def kkt(x,a,b):
    ga,gb=gradients(x,a,b)
    return {'coefficient_pg_inf':float(np.max(np.abs(projected_gradient(a,ga)))),
            'basis_pg_inf':float(np.max(np.abs(projected_gradient(b,gb)))),
            'complementarity_inf':float(max(np.max(np.abs(a*ga)),np.max(np.abs(b*gb))))}


def sweep(a,gram,cross):
    """One exact scalar-coordinate minimizer at a time, vectorized across rows.

    Gram has shape rank x rank; cross is X B.T. No epsilon in denominators:
    a zero basis row has zero squared norm and zero cross, so its coefficient
    can be set to zero without changing the reconstructed vector.
    """
    g=a@gram-cross
    for j in range(a.shape[1]):
        old=a[:,j].copy()
        if gram[j,j]>0:a[:,j]=np.maximum(0,old-g[:,j]/gram[j,j])
        else:a[:,j]=0.
        g+=(a[:,j]-old)[:,None]*gram[j]
    return a


def encode(x,b,max_sweeps=1000,tol=1e-8,initial=None,history=False):
    """Fixed-basis convex NNLS, explicit per-row numerical KKT stopping.

    Unique reconstruction may coexist with nonunique coefficients. No implicit
    convergence claim on budget exhaustion; return the unfinished row mask.
    Relative criterion uses max(1, max(abs(X B.T))) separately for each row.
    """
    x=matrix(x,'inputs');b=matrix(b,'basis')
    if x.shape[1]!=b.shape[1] or type(max_sweeps) is not int or max_sweeps<0 or not np.isfinite(tol) or tol<0:raise ValueError('dimensions, sweep budget or tolerance')
    a=np.zeros((len(x),len(b))) if initial is None else matrix(initial,'initial').copy()
    if a.shape!=(len(x),len(b)):raise ValueError('initial shape')
    gram=b@b.T;cross=x@b.T;threshold=tol*np.maximum(1,np.max(np.abs(cross),axis=1))
    losses=[objective(x,a,b)] if history else None
    done=np.zeros(len(x),dtype=bool);iterations=0
    for it in range(max_sweeps+1):
        # Recompute from the actual matrix product, not accumulated updates.
        g=(a@b-x)@b.T;pg=np.max(np.abs(projected_gradient(a,g)),axis=1);done=pg<=threshold
        if done.all() or it==max_sweeps:break
        a=sweep(a,gram,cross);iterations+=1
        if history:losses.append(objective(x,a,b))
    return {'coefficients':a,'reconstruction':a@b,'sweeps':iterations,
            'converged':done,'projected_gradient_inf':pg,'threshold':threshold,'history':losses}


def normalize(a,b):
    """Fix the display gauge: each nonzero basis row gets unit Euclidean norm.

    This preserves A B, but not the coordinate scale of gradient norms.
    Zero components are left at zero, without claiming identifiability.
    """
    scale=np.linalg.norm(b,axis=1);safe=np.where(scale>0,scale,1)
    return a*safe,b/safe[:,None]


def fit(x,rank,seed=11,outer=100,inner=3,tol=1e-5):
    x=matrix(x,'training pixels')
    if type(rank) is not int or not 1<=rank<=min(x.shape) or type(outer) is not int or outer<0 or type(inner) is not int or inner<1 or tol<0 or not np.isfinite(tol):raise ValueError('invalid fit protocol')
    rng=np.random.default_rng(seed);b=rng.random((rank,x.shape[1]))+.01;a=rng.random((len(x),rank))*.1;a,b=normalize(a,b)
    history=[]
    def record(step):
        diag=kkt(x,a,b);history.append({'step':step,'half_squared_error':objective(x,a,b),**diag})
    record(0);initial=max(history[0]['coefficient_pg_inf'],history[0]['basis_pg_inf'],1.)
    for it in range(outer):
        gram=b@b.T;cross=x@b.T
        for _ in range(inner):a=sweep(a,gram,cross)
        bt=b.T.copy();gram=a.T@a;cross=x.T@a
        for _ in range(inner):bt=sweep(bt,gram,cross)
        b=bt.T;a,b=normalize(a,b);record(it+1)
        if max(history[-1]['coefficient_pg_inf'],history[-1]['basis_pg_inf'])<=tol*initial:break
    return {'basis':b,'coefficients':a,'history':history,'seed':seed,'rank':rank,
            'stationary_within_tolerance':max(history[-1]['coefficient_pg_inf'],history[-1]['basis_pg_inf'])<=tol*initial,'initial_pg_scale':initial,'relative_tolerance':tol}


def multiplicative_coefficients(x,b,a,steps):
    """Deliberately plain squared-error MU control; zeros stay zero.

    If denominator is zero, leave the entry unchanged. This control is not used
    by training or inference and is not a claim about modified MU methods.
    """
    a=a.copy();cross=x@b.T;gram=b@b.T
    for _ in range(steps):
        den=a@gram;ratio=np.divide(cross,den,out=np.ones_like(cross),where=den>0);a*=ratio
    return a
