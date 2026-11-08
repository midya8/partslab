import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent

def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x

def dumps(x):return json.dumps(plain(x),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
def digest(x):return hashlib.sha256(dumps(x).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def write(p,x):Path(p).write_text(dumps(x))
