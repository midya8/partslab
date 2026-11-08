"""Independent determinant/Cramer active-set oracles for exact face intervals."""
import copy
import itertools
import json
import math
import unittest
from fractions import Fraction as F
from pathlib import Path
from partslab.face import response, verify


def determinant(a):
    if not a:
        return F(1)
    return sum(((-1)**j * a[0][j] * determinant([r[:j]+r[j+1:] for r in a[1:]]) for j in range(len(a))), F())


def oracle(x, basis):
    """Enumerate independent column sets with Cramer, never use face.solve."""
    best = None
    for bits in itertools.product([0, 1], repeat=len(basis)):
        s = [k for k, yes in enumerate(bits) if yes]
        gram = [[sum(u*v for u,v in zip(basis[k], basis[l])) for l in s] for k in s]
        det = determinant(gram)
        if not det:
            continue
        rhs = [sum(u*v for u,v in zip(basis[k],x)) for k in s]
        c = [F()] * len(basis)
        for j,k in enumerate(s):
            mat = [[rhs[i] if q==j else v for q,v in enumerate(row)] for i,row in enumerate(gram)]
            c[k] = determinant(mat)/det
        if any(v<0 for v in c):
            continue
        residual = [sum(c[k]*basis[k][j] for k in range(len(basis)))-x[j] for j in range(len(x))]
        grad = [sum(u*v for u,v in zip(row,residual)) for row in basis]
        if any(g<0 or g*v!=0 for g,v in zip(grad,c)):
            continue
        objective = sum(v*v for v in residual)/2
        if best is not None:
            assert best == objective
        best = objective
    return best


def frac(v):
    return F(int(v['numerator']),int(v['denominator']))


class FaceTests(unittest.TestCase):
    def test_exhaustive_binary_bases(self):
        count = 0
        for flat in itertools.product([0,1], repeat=6):
            basis = [list(map(F, flat[i:i+2])) for i in range(0,6,2)]
            for bits in itertools.product([0,1], repeat=3):
                s = [k for k,v in enumerate(bits) if v]
                for pixel in [0,1]:
                    try:
                        w = response([.25,.75],[[float(v) for v in row] for row in basis],s,pixel)
                    except ValueError:
                        continue
                    self.assertTrue(verify(w))
                    lo,hi = frac(w['interval']['lower']),frac(w['interval']['upper'])
                    for t in [lo,hi,(lo+hi)/2,F()]:
                        x=[F(1,4),F(3,4)];x[pixel]+=t
                        c=[frac(v['intercept'])+frac(v['slope'])*t for v in w['coefficients']]
                        r=[sum(c[k]*basis[k][j] for k in range(3))-x[j] for j in range(2)]
                        self.assertEqual(sum(v*v for v in r)/2,oracle(x,basis))
                    for t in [lo-F(1,100),hi+F(1,100)]:
                        if 0<=F([.25,.75][pixel])+t<=1:
                            self.assertTrue(any(frac(v['intercept'])+frac(v['slope'])*t<0 for v in w['inequalities']))
                    count+=1
        self.assertEqual(count,200)

    def test_all54_source_bound_native_exports(self):
        from partslab.face_export import export
        from partslab.io import ROOT,read
        self.assertEqual(export(),read(ROOT/'artifacts/face-reference.json'))

    def test_subnormal_and_binary64_not_decimal(self):
        for v in [math.ulp(0.),2.**-1022,.1]:
            w=response([v],[[1]],[0],0)
            self.assertTrue(verify(w))
            self.assertEqual(frac(w['coefficients'][0]['intercept']),F(v))
        self.assertNotEqual(frac(response([.1],[[1]],[0],0)['coefficients'][0]['intercept']),F('0.1'))

    def test_all120_source_nominations_and_immutable_bundle(self):
        from partslab.io import ROOT,read
        from partslab.data import load
        m=read(ROOT/'artifacts/models.json');c=read(ROOT/'artifacts/certificate-candidates.json')['cases'];x=load('test')
        bundle=json.loads((ROOT.parent/'docs/data.js').read_text().removeprefix('globalThis.PARTS_DATA=').strip().removesuffix(';'))
        before=copy.deepcopy((m,c))
        for case in c:
            rank=str(case['rank']);run=case['run'];row=case['row']
            b=m['arms'][rank]['runs'][run]['basis'];self.assertEqual(b,bundle['arms'][rank]['runs'][run]['basis'])
            self.assertEqual(x[row].tolist(),bundle['examples'][row]['pixels'])
            self.assertEqual(case['coefficients'],bundle['reference'][rank][run][row]['coefficients'])
            w=response(x[row].tolist(),b,[k for k,v in enumerate(case['coefficients']) if v>0],27)
            self.assertTrue(verify(w))
        self.assertEqual((m,c),before)

    def test_cli_and_duplicate_json_rejection(self):
        import subprocess,sys,tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'problem.json';w=root/'witness.json'
            p.write_text(json.dumps({'pixels':[.5],'basis':[[1]],'support':[0],'pixel':0}))
            subprocess.run([sys.executable,'-m','partslab','face','--problem',str(p),'--out',str(w)],check=True,capture_output=True)
            out=subprocess.run([sys.executable,'-m','partslab','verify-face','--witness',str(w)],check=True,capture_output=True,text=True)
            self.assertTrue(json.loads(out.stdout)['verified'])
            p.write_text('{'+'"pixels":[0.5],"basis":[[1]],"support":[0],"pixel":0,"pixel":1}')
            bad=subprocess.run([sys.executable,'-m','partslab','face','--problem',str(p),'--out',str(w)],capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0);self.assertIn('duplicate JSON',bad.stderr)

    def test_controls(self):
        # Duplicate components: one face works but coefficients need not be unique.
        w=response([.5],[[1],[1]],[0],0);self.assertTrue(verify(w))
        self.assertEqual(w['inequalities'][-1]['slope'],{'numerator':'0','denominator':'1'})
        with self.assertRaisesRegex(ValueError,'singular'):
            response([.5],[[1],[1]],[0,1],0)
        w=response([0,0],[[1,0]],[],0)
        self.assertTrue(w['interval']['singleton']);self.assertTrue(verify(w))
        w=response([.5,.5],[[1,1],[1,0]],[0],0)
        self.assertEqual(frac(w['interval']['upper']),0)
        self.assertEqual(w['interval']['upper_limits'],[{'kind':'gradient','index':1}])
        with self.assertRaisesRegex(ValueError,'starting'):
            response([.5],[[1]],[],0)

    def test_strict_inputs_and_nonmutation(self):
        x=[.5,.5];b=[[1,0],[0,1]];s=[0,1];old=copy.deepcopy((x,b,s))
        response(x,b,s,0);self.assertEqual((x,b,s),old)
        for support in [[True],[0,0],[1,0],[-1],[2],(0,)]:
            with self.assertRaises(ValueError):response(x,b,support,0)
        for pixel in [True,-1,2,.5]:
            with self.assertRaises(ValueError):response(x,b,[],pixel)
        for pixels in [[True,.5],[math.nan,.5],[1.01,.5],[-.1,.5]]:
            with self.assertRaises(ValueError):response(pixels,b,[],0)

    def test_verifier_tampering(self):
        w=response([.5,.5],[[1,1],[1,0]],[0],0)
        for change in ['slope','bound','limit','stationarity','canonical','schema','short']:
            v=copy.deepcopy(w)
            if change=='slope':v['coefficients'][0]['slope']['numerator']='0'
            if change=='bound':v['interval']['upper']['numerator']='1'
            if change=='limit':v['interval']['upper_limits']=[]
            if change=='stationarity':v['coefficients'][1]['slope']['numerator']='1'
            if change=='canonical':v['coefficients'][0]['slope']={'numerator':'2','denominator':'4'}
            if change=='schema':v['schema']=True
            if change=='short':v['coefficients']=[]
            self.assertFalse(verify(v),change)


if __name__=='__main__':unittest.main()
