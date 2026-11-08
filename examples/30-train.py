from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from partslab.experiment import train,evaluate,PROTOCOL
from partslab.io import ROOT,write,read
from partslab.audit import controls
m,g=train();write(ROOT/'artifacts/models.json',m);write(ROOT/'artifacts/guard.json',g);write(ROOT/'artifacts/protocol.json',PROTOCOL)
write(ROOT/'artifacts/report.json',evaluate(read(ROOT/'artifacts/models.json'),read(ROOT/'artifacts/guard.json')));write(ROOT/'artifacts/audit.json',controls())
