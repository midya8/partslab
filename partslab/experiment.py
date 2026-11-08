"""Fixed before holdout: three ranks, two seeds, 100 outer/3 inner sweeps."""
import numpy as np
from . import data
from .core import fit,encode,kkt
from .io import ROOT,digest,read,write

PROTOCOL={'ranks':[8,16,32],'seeds':[11,23],'outer':100,'inner':3,'fit_relative_pg_tolerance':1e-5,'encode_max_sweeps':1000,'encode_relative_pg_tolerance':1e-8,'selection':'minimum reencoded training MSE within each fixed rank; lower seed breaks exact ties','holdout':'original 1797-row optdigits.tes, never opened while fitting','scale':'integer pixels divided by 16','pca':'centered training SVD, no whitening; signed unclipped reconstruction','svd':'uncentered training SVD, signed unclipped reconstruction','mean':'training pixel mean; no fitted test parameters'}


def encoding(x,b):return encode(x,b,PROTOCOL['encode_max_sweeps'],PROTOCOL['encode_relative_pg_tolerance'])


def diagnostics(x,e):
    mse=np.mean((x-e['reconstruction'])**2,axis=1)
    return {'mse':float(mse.mean()),'row_mse':mse,'sweeps':e['sweeps'],'converged_rows':int(e['converged'].sum()),'rows':len(x),'max_pg_inf':float(e['projected_gradient_inf'].max()),'pg_inf':e['projected_gradient_inf'],'threshold':e['threshold'],'converged_mask':e['converged']}


def train():
    x=data.load('train');arms={};mean=x.mean(axis=0)
    _,_,pc=np.linalg.svd(x-mean,full_matrices=False);_,_,sv=np.linalg.svd(x,full_matrices=False)
    for rank in PROTOCOL['ranks']:
        runs=[]
        for seed in PROTOCOL['seeds']:
            r=fit(x,rank,seed,PROTOCOL['outer'],PROTOCOL['inner'],PROTOCOL['fit_relative_pg_tolerance']);e=encoding(x,r['basis']);runs.append({'seed':seed,'basis':r['basis'],'history':r['history'],'stationary_within_tolerance':r['stationary_within_tolerance'],'initial_pg_scale':r['initial_pg_scale'],'fit_kkt':kkt(x,r['coefficients'],r['basis']),'reencoded_training':diagnostics(x,e)})
        chosen=min(range(len(runs)),key=lambda i:(runs[i]['reencoded_training']['mse'],runs[i]['seed']))
        arms[str(rank)]={'rank':rank,'runs':runs,'chosen':chosen,'pca_components':pc[:rank],'svd_components':sv[:rank]}
    models={'schema':1,'protocol_sha256':digest(PROTOCOL),'train_rows':len(x),'mean':mean,'arms':arms}
    return models,{'schema':1,'protocol_sha256':digest(PROTOCOL),'models_sha256':digest(models)}


def evaluate(models,guard):
    if guard['models_sha256']!=digest(models) or guard['protocol_sha256']!=digest(PROTOCOL) or models['protocol_sha256']!=digest(PROTOCOL):raise ValueError('frozen training artifact/protocol changed')
    x=data.load('test');mean=np.array(models['mean']);arms={}
    for name,a in models['arms'].items():
        runs=[]
        for r in a['runs']:runs.append({'seed':r['seed'],**diagnostics(x,encoding(x,np.array(r['basis'])))})
        pc=np.array(a['pca_components']);sv=np.array(a['svd_components']);pr=(x-mean)@pc.T@pc+mean;sr=x@sv.T@sv
        arms[name]={'chosen_seed':a['runs'][a['chosen']]['seed'],'nmf_runs':runs,'pca_row_mse':np.mean((x-pr)**2,axis=1),'pca_mse':float(np.mean((x-pr)**2)),'svd_row_mse':np.mean((x-sr)**2,axis=1),'svd_mse':float(np.mean((x-sr)**2))}
    return {'schema':1,'models_sha256':guard['models_sha256'],'test_rows':len(x),'mean_mse':float(np.mean((x-mean)**2)),'mean_row_mse':np.mean((x-mean)**2,axis=1),'arms':arms}


def predict(pixels,models,rank):
    x=np.asarray(pixels,dtype=float)
    if x.shape!=(64,) or not np.isfinite(x).all() or (x<0).any() or (x>1).any():raise ValueError('64 finite pixels in [0,1] required')
    a=models['arms'][str(rank)];r=a['runs'][a['chosen']];b=np.array(r['basis']);e=encoding(x[None,:],b);mean=np.array(models['mean']);pc=np.array(a['pca_components']);sv=np.array(a['svd_components']);pca=(x-mean)@pc.T@pc+mean;svd=x@sv.T@sv
    return {'rank':int(rank),'seed':r['seed'],'coefficients':e['coefficients'][0],'reconstruction':e['reconstruction'][0],'nnls_converged':bool(e['converged'][0]),'nnls_sweeps':e['sweeps'],'projected_gradient_inf':float(e['projected_gradient_inf'][0]),'threshold':float(e['threshold'][0]),'nmf_mse':float(np.mean((x-e['reconstruction'][0])**2)),'pca_reconstruction':pca,'pca_mse':float(np.mean((x-pca)**2)),'svd_reconstruction':svd,'svd_mse':float(np.mean((x-svd)**2)),'mean_mse':float(np.mean((x-mean)**2))}
