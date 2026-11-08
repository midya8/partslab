#!/usr/bin/env bash
set -euo pipefail
installed=$1
source_root=$2
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
export PYTHONPATH="$installed"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python - <<'PY'
import importlib,pkgutil,partslab
mods=sorted(m.name for m in pkgutil.walk_packages(partslab.__path__,partslab.__name__+'.'))
for m in mods:importlib.import_module(m)
assert len(mods)==9,mods
print('all ten package modules including root imported outside checkout')
PY
python -m partslab train --out models.json --guard guard.json
python -m partslab evaluate --models models.json --guard guard.json --out report.json
python -m partslab audit --out audit.json
python -m partslab demo >demo.json
python - <<'PY'
import json
json.dump([0.]*64,open('pixels.json','w'))
PY
python -m partslab predict --pixels pixels.json --out predict.json
python - <<'PY'
import json
json.dump({'pixels':[1.,1.],'basis':[[1.,0.],[0.,1.]],'coefficients':[1.,0.]},open('problem.json','w'))
PY
python -m partslab certify --problem problem.json --out certificate.json
python -m partslab verify-certificate --witness certificate.json >verified.json
python - <<'PYFACE'
import json
json.dump({'pixels':[.5,.5],'basis':[[1.,1.],[1.,0.]],'support':[0],'pixel':0},open('face-problem.json','w'))
PYFACE
python -m partslab face --problem face-problem.json --out face.json
python -m partslab verify-face --witness face.json >face-verified.json
python -m partslab face-export --out face-reference.json
python - "$source_root" <<'PY'
import json,sys,numpy as np
from pathlib import Path
from partslab.io import digest
source=Path(sys.argv[1])/'partslab/artifacts'
def close(a,b,path=''):
    if isinstance(a,dict):
        assert set(a)==set(b),path
        for k in a:
            if k not in ['models_sha256']:close(a[k],b[k],path+'/'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):close(x,y,path+'/'+str(i))
    elif type(a) in [int,float] and type(b) in [int,float]:assert abs(a-b)<=2e-9+2e-8*abs(b),(path,a,b)
    else:assert a==b,(path,a,b)
for name in ['models','report','audit']:close(json.load(open(name+'.json')),json.load(open(source/(name+'.json'))))
models=json.load(open('models.json'));guard=json.load(open('guard.json'));assert guard['models_sha256']==digest(models)
p=json.load(open('predict.json'));assert p['nmf_mse']==0 and p['nnls_converged'] and p['nnls_sweeps']==0
assert json.load(open('demo.json'))['test_rows']==1797
assert json.load(open('verified.json'))['verified']
from partslab.certificate import verify
artifact=json.load(open(source/'certificates.json'));assert all(verify(w) for w in artifact['controls'])
assert json.load(open('face-verified.json'))['verified']
assert json.load(open('face-reference.json'))==json.load(open(source/'face-reference.json'))
print('all ten installed CLIs, six complete refits, fixed holdout, controls and frozen guard pass')
PY
