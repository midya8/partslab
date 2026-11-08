import copy,itertools,json,math,random,unittest
from fractions import Fraction as F
from partslab.certificate import certify,verify,decimal_bound
from partslab.io import ROOT,read
from partslab.data import load
from partslab.experiment import predict


def solve_linear(a,y):
    a=[list(row)+[v] for row,v in zip(a,y)];n=len(y)
    for j in range(n):
        pivot=next((i for i in range(j,n) if a[i][j]),None)
        if pivot is None:return None
        a[j],a[pivot]=a[pivot],a[j];p=a[j][j];a[j]=[v/p for v in a[j]]
        for i in range(n):
            if i!=j:
                p=a[i][j];a[i]=[u-p*v for u,v in zip(a[i],a[j])]
    return [row[-1] for row in a]


def exact_oracle(x,b):
    x=list(map(F,x));b=[list(map(F,row)) for row in b];best=None
    for bits in itertools.product([0,1],repeat=len(b)):
        ids=[i for i,bit in enumerate(bits) if bit];c=[F()]*len(b)
        if ids:
            g=[[sum((u*v for u,v in zip(b[i],b[j])),F()) for j in ids] for i in ids]
            cross=[sum((u*v for u,v in zip(b[i],x)),F()) for i in ids];v=solve_linear(g,cross)
            if v is None or min(v)<0:continue
            for i,a in zip(ids,v):c[i]=a
        residual=[x[j]-sum((c[i]*b[i][j] for i in range(len(b))),F()) for j in range(len(x))]
        if any(sum((v*r for v,r in zip(row,residual)),F())>0 for row in b):continue
        loss=sum((r*r/2 for r in residual),F());best=loss if best is None else min(best,loss)
    assert best is not None
    return best


def value(w,k):return F(int(w[k]['numerator']),int(w[k]['denominator']))


class CertificateTests(unittest.TestCase):
    def test_80_exact_active_set_oracles_and_feasible_intervals(self):
        rng=random.Random(991)
        for rank in range(1,5):
            for _ in range(20):
                x=[rng.randrange(17)/16 for _ in range(4)];b=[[rng.randrange(9)/8 for _ in range(4)] for _ in range(rank)];c=[rng.randrange(17)/16 for _ in range(rank)]
                w=certify(x,b,c);self.assertTrue(verify(w));opt=exact_oracle(x,b)
                self.assertLessEqual(value(w,'dual'),opt);self.assertLessEqual(opt,value(w,'primal'));self.assertLessEqual(value(w,'primal')-opt,value(w,'gap'))
    def test_exact_binary_values_not_rounded_decimal_surrogate(self):
        w=certify([.1,.3],[[.7,.2]],[.9]);self.assertTrue(verify(w));p=sum((F(x)-F(.9)*F(b))**2/2 for x,b in zip([.1,.3],[.7,.2]));self.assertEqual(value(w,'primal'),p)
        self.assertNotEqual(p,sum((F(str(x))-F('.9')*F(str(b)))**2/2 for x,b in zip([.1,.3],[.7,.2])))
    def test_zero_gap_does_not_imply_unique_coefficients(self):
        a=certify([1,0],[[1,0],[1,0]],[1,0]);b=certify([1,0],[[1,0],[1,0]],[0,1]);self.assertTrue(verify(a) and verify(b));self.assertEqual(value(a,'gap'),0);self.assertEqual(value(b,'gap'),0);self.assertNotEqual(a['problem']['coefficients'],b['problem']['coefficients'])
    def test_zero_rows_zero_input_unfinished_and_loose_bound(self):
        for x,b,c in [([0,0],[[1,0]],[0]),([1,1],[[0,0]],[0]),([1,1],[[1,0],[0,1]],[0,0]),([1,1],[[1,0],[0,1]],[1,0])]:
            w=certify(x,b,c);self.assertTrue(verify(w));self.assertLessEqual(value(w,'dual'),exact_oracle(x,b));self.assertGreaterEqual(value(w,'primal'),exact_oracle(x,b))
        w=certify([1,1],[[1,0],[0,1]],[1,0]);self.assertEqual(w['dual_source'],'zero-fallback');self.assertEqual(value(w,'gap'),F(1,2))
    def test_weak_duality_gap_sum_of_squares_identity(self):
        w=certify([.8,.3,.7],[[.2,.7,.1],[.3,.1,.6]],[.4,.9]);p=w['problem'];x=list(map(F,p['pixels']));b=[list(map(F,row)) for row in p['basis']];c=list(map(F,p['coefficients']));y=[F(int(v),int(w['dual_denominator'])) for v in w['dual_numerators']];r=[x[j]-sum(c[k]*b[k][j] for k in range(2)) for j in range(3)]
        squared=sum((u-v)**2/2 for u,v in zip(r,y));slack=-sum(c[k]*sum(a*v for a,v in zip(b[k],y)) for k in range(2));self.assertGreaterEqual(slack,0);self.assertEqual(squared+slack,value(w,'gap'))
    def test_outward_decimal_rounding_never_understates_upper(self):
        for f in [F(0),F(1,3),F(9999999,1000000),F(1,10**80),F(2**211,3**151),F(9999999,1000000000000000000000000)]:
            lo,hi=decimal_bound(f),decimal_bound(f,True);self.assertLessEqual(F(lo),f);self.assertGreaterEqual(F(hi),f)
    def test_subnormal_and_large_exponents_exact(self):
        for a in [math.ulp(0.),2.**-1022,2.**-500,2.**500]:
            w=certify([a],[[1.]],[0.]);self.assertTrue(verify(w));self.assertEqual(value(w,'primal'),F(a)**2/2)
    def test_tampering_and_invalid_problem_rejected(self):
        original=certify([.8,.3],[[.2,.7]],[.4]);self.assertTrue(verify(original))
        for key in ['primal','dual','gap']:
            w=copy.deepcopy(original);w[key]['numerator']=str(int(w[key]['numerator'])+1);self.assertFalse(verify(w))
        for key,v in [('dual_denominator','0'),('scope','joint global NMF'),('dual_numerators',['1','1'])]:
            w=copy.deepcopy(original);w[key]=v;self.assertFalse(verify(w))
        w=copy.deepcopy(original);w['display']['suboptimality_upper']='0';self.assertFalse(verify(w))
        for x,b,c in [([],[[]],[0]),([1],[[1,0]],[0]),([1],[[1]],[math.nan]),([1],[[-1]],[0]),([True],[[1]],[0]),(['1'],[[1]],[0])]:
            with self.assertRaises(ValueError):certify(x,b,c)
    def test_120_saved_exact_witnesses_and_five_controls_reproduce(self):
        artifact=read(ROOT/'artifacts/certificates.json');m=read(ROOT/'artifacts/models.json');x=load('test')[:20];self.assertEqual(len(artifact['cases']),120)
        for case in artifact['cases']:
            rank=str(case['rank']);ri=case['run'];i=case['row'];temp=copy.deepcopy(m);temp['arms'][rank]['chosen']=ri;p=predict(x[i],temp,int(rank));w=certify(x[i],m['arms'][rank]['runs'][ri]['basis'],p['coefficients']);self.assertTrue(verify(w));del w['problem'];self.assertEqual(w,case['certificate'])
        self.assertEqual(len(artifact['controls']),5);self.assertTrue(all(verify(w) for w in artifact['controls']))

if __name__=='__main__':unittest.main()
