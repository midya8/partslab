import numpy as np
from .core import encode,multiplicative_coefficients,objective,gradients,normalize

def controls():
    x=np.array([[1.,1.]]);b=np.eye(2);a=np.array([[1.,0.]])
    mu=multiplicative_coefficients(x,b,a,20);cd=encode(x,b,20,1e-12,initial=a)
    # Duplication makes coefficients nonunique even with a fixed dictionary.
    db=np.array([[1.,0.],[1.,0.]]);d1=np.array([[1.,0.]]);d2=np.array([[0.,1.]])
    scale=np.array([2.,.25]);ga,gb=normalize(a,b)
    return {'zero_lock':{'x':x,'basis':b,'initial':a,'mu_coefficients':mu,'mu_loss':objective(x,mu,b),'mu_gradient':gradients(x,mu,b)[0],'cd_coefficients':cd['coefficients'],'cd_loss':objective(x,cd['coefficients'],b),'cd_converged':bool(cd['converged'].all())},'duplicate_basis':{'basis':db,'coefficients_a':d1,'coefficients_b':d2,'reconstruction_a':d1@db,'reconstruction_b':d2@db},'scaling':{'basis':b,'coefficients':a,'scaled_basis':b/scale[:,None],'scaled_coefficients':a*scale,'original_reconstruction':a@b,'scaled_reconstruction':(a*scale)@(b/scale[:,None])},'scope':'Numerical optimization and explicit nonidentifiability controls; not semantic explanations or global NMF optima.'}
