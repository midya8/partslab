from pathlib import Path
import sys,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from partslab.io import ROOT,read,dumps
from partslab.data import load
from partslab.experiment import predict
m=read(ROOT/'artifacts/models.json');report=read(ROOT/'artifacts/report.json');x=load('test')[:20];arms={};refs={}
for rank,a in m['arms'].items():
    runs=[];refs[rank]=[]
    for i,r in enumerate(a['runs']):
        runs.append({k:r[k] for k in ['seed','basis','history','stationary_within_tolerance','initial_pg_scale','fit_kkt']}|{'train_mse':r['reencoded_training']['mse'],'train_converged_rows':r['reencoded_training']['converged_rows']})
        temp=copy.deepcopy(m);temp['arms'][rank]['chosen']=i;refs[rank].append([predict(row,temp,int(rank)) for row in x])
    arms[rank]={'rank':a['rank'],'chosen':a['chosen'],'runs':runs,'pca_components':a['pca_components'],'svd_components':a['svd_components']}
b={'schema':1,'mean':m['mean'],'arms':arms,'examples':[{'row':i,'pixels':v} for i,v in enumerate(x)],'reference':refs,'report':report,'audit':read(ROOT/'artifacts/audit.json'),'protocol':read(ROOT/'artifacts/protocol.json')}
Path('docs/data.js').write_text('globalThis.PARTS_DATA='+dumps(b).rstrip()+';\n')
