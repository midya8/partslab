"""Rebuild exact certificates for the 120 already-public model/example pairs."""
import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from partslab.io import ROOT,read,write
from partslab.data import load
from partslab.experiment import predict
from partslab.certificate import certify,verify
m=read(ROOT/'artifacts/models.json');x=load('test')[:20];cases=[]
for rank,a in m['arms'].items():
    for ri,r in enumerate(a['runs']):
        temp=copy.deepcopy(m);temp['arms'][rank]['chosen']=ri
        for i,row in enumerate(x):
            p=predict(row,temp,int(rank));w=certify(row,r['basis'],p['coefficients']);assert verify(w)
            del w['problem'];cases.append({'rank':int(rank),'run':ri,'row':i,'certificate':w})
controls=[certify([1,1],[[1,0],[0,1]],[1,0]),certify([1,0],[[1,0],[1,0]],[1,0]),certify([1,0],[[1,0],[1,0]],[0,1]),certify([1,1],[[0,0]],[0]),certify([1,1],[[1,0],[0,1]],[0,0])]
assert all(verify(w) for w in controls)
write(ROOT/'artifacts/certificates.json',{'schema':1,'scope':'120 previously published cases only; not a full-holdout gap claim','cases':cases,'controls':controls})
print('120 exact fixed-basis certificates and five controls exported')
