from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from partslab.io import ROOT,read

def render(models,report):
    a=models['arms']['16'];r=a['runs'][a['chosen']];s=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1110" viewBox="0 0 1200 1110">','<rect width="1200" height="1110" fill="#101e29"/>']
    def text(x,y,value,size=18,color='#f3f1e6',family='sans-serif'):s.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="{family}" font-size="{size}">{value}</text>')
    text(40,40,'MIDYA / PARTSLAB / NONNEGATIVE IS NOT NONAMBIGUOUS',13,'#d1ee91','monospace');text(40,100,'Add without subtracting.',48,family='serif');text(40,138,'Six manual NMF fits. Original 3,823 fitting / 1,797 held-out UCI digits. No labels in training.',18,'#b5c5c4');text(40,177,'Rank 16 / training-selected seed '+str(r['seed'])+' / each basis display-rescaled separately',17,'#d1ee91')
    for i,row in enumerate(r['basis']):
        x=40+(i%8)*142;y=208+(i//8)*165;scale=max(row) or 1
        s.append(f'<rect x="{x}" y="{y}" width="126" height="148" fill="#172b38" stroke="#38515c"/>');text(x+9,y+20,'PART '+str(i),12,'#d1ee91','monospace')
        for p,v in enumerate(row):
            n=round(max(0,min(1,v/scale))*255);s.append(f'<rect x="{x+15+(p%8)*12}" y="{y+33+(p//8)*12}" width="12" height="12" fill="rgb({n},{n},{n})"/>')
    text(40,580,'Official holdout reconstruction MSE / no clipping in scores',27,family='serif');text(40,622,'RANK',15,'#d1ee91','monospace');text(180,622,'CHOSEN SEED',15,'#d1ee91','monospace');text(420,622,'NMF',15,'#d1ee91','monospace');text(650,622,'PCA',15,'#d1ee91','monospace');text(880,622,'SVD',15,'#d1ee91','monospace')
    for i,(name,a) in enumerate(sorted(report['arms'].items(),key=lambda item:int(item[0]))):
        y=666+i*45;v=next(r for r in a['nmf_runs'] if r['seed']==a['chosen_seed'])
        for x,value in [(40,name),(180,str(a['chosen_seed'])),(420,f'{v["mse"]:.7f}'),(650,f'{a["pca_mse"]:.7f}'),(880,f'{a["svd_mse"]:.7f}')]:text(x,y,value,20,family='monospace')
    text(40,822,'PCA and SVD win at every fixed rank. This is not digit classification.',19,'#d1ee91');text(40,856,'Every NMF fit exhausts its joint stationarity budget; lower loss is not a global optimum.',18,'#b5c5c4');text(40,900,'ZERO-LOCK CONTROL',14,'#d1ee91','monospace');text(40,933,'Target [1,1], identity basis, initial coefficients [1,0].',19);text(40,966,'Plain multiplicative updates stay at loss 0.5 with a negative blocked gradient.',19);text(40,999,'Coordinate minimization reaches [1,1], loss 0. Same objective, explicit boundary step.',19);text(40,1046,'Scaling, permutation and duplicate bases can preserve a sum but change its coefficients.',17,'#b5c5c4');text(40,1080,'Alpaydin &amp; Kaynak (1998), UCI Optical Digits. CC BY 4.0 data. MIT code. No endorsement.',14,'#b5c5c4')
    s.append('</svg>');return '\n'.join(s)+'\n'
if __name__=='__main__':Path('assets/parts-audit.svg').write_text(render(read(ROOT/'artifacts/models.json'),read(ROOT/'artifacts/report.json')))
