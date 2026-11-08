"""Original UCI split, integer pixels /16, labels deliberately not returned."""
import hashlib
import numpy as np
from .io import ROOT

def load(split):
    if split not in ['train','test']:raise ValueError('split must be train or test')
    name='optdigits.tra' if split=='train' else 'optdigits.tes';p=ROOT/'data'/name
    # Expected source hashes are stored directly; inspect manifest at build time.
    expected=HASHES[name]
    if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('dataset hash mismatch')
    rows=np.loadtxt(p,delimiter=',',dtype=int)
    if rows.shape!=(3823 if split=='train' else 1797,65) or (rows[:,:64]<0).any() or (rows[:,:64]>16).any():raise ValueError('source shape/range')
    return rows[:,:64].astype(float)/16

HASHES={'optdigits.tra': 'e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd', 'optdigits.tes': '6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8'}
