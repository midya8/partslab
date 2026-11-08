import itertools,unittest
import numpy as np
from partslab.core import matrix,objective,gradients,projected_gradient,sweep,encode,normalize,fit,kkt
from partslab.audit import controls


def active_set_oracle(x,b):
    best=float('inf');answer=None
    for bits in itertools.product([0,1],repeat=len(b)):
        ids=np.flatnonzero(bits);a=np.zeros(len(b))
        if len(ids):
            # Enumerate independent supports; every feasible cone projection has
            # an independent-support representation. No normal-equation inverse.
            if np.linalg.matrix_rank(b[ids])<len(ids):continue
            coef=np.linalg.lstsq(b[ids].T,x,rcond=None)[0]
            if np.min(coef)<-1e-10:continue
            a[ids]=np.maximum(0,coef)
        loss=float(np.sum((x-a@b)**2)/2)
        if loss<best:best=loss;answer=a
    return best,answer


class CoreTests(unittest.TestCase):
    def test_objective_scalar_independent(self):
        x=np.array([[1.,2.,0.],[.3,.2,.1]]);a=np.array([[1.,.5],[0.,.2]]);b=np.array([[.4,1.,0.],[1.,0.,.3]])
        expected=sum((x[i,j]-sum(a[i,k]*b[k,j] for k in range(2)))**2/2 for i in range(2) for j in range(3));self.assertAlmostEqual(objective(x,a,b),expected)
    def test_gradients_central_differences(self):
        rng=np.random.default_rng(19);x=rng.random((4,5));a=rng.random((4,3));b=rng.random((3,5));ga,gb=gradients(x,a,b)
        for z,g in [(a,ga),(b,gb)]:
            for idx in np.ndindex(z.shape):
                old=z[idx];z[idx]=old+1e-6;hi=objective(x,a,b);z[idx]=old-1e-6;lo=objective(x,a,b);z[idx]=old
                self.assertAlmostEqual(g[idx],(hi-lo)/2e-6,places=7)
    def test_scalar_coordinate_exact_minimizer(self):
        x=np.array([[.7,.2]]);b=np.array([[.3,.5],[.9,.1]]);a=np.array([[.2,.4]]);g=b@b.T;c=x@b.T
        expected=a.copy()
        for j in range(2):
            residual=x[0]-sum(expected[0,k]*b[k] for k in range(2) if k!=j);expected[0,j]=max(0,float(residual@b[j]/(b[j]@b[j])))
        np.testing.assert_allclose(sweep(a,g,c),expected,atol=1e-14)
    def test_nnls_active_set_oracle_35_random_and_zero_cases(self):
        rng=np.random.default_rng(47)
        for rank in range(1,6):
            for _ in range(7):
                b=rng.random((rank,4));x=rng.random(4);e=encode(x[None,:],b,5000,1e-10);ref,_=active_set_oracle(x,b)
                self.assertAlmostEqual(objective(x[None,:],e['coefficients'],b),ref,places=8)
                self.assertTrue(e['converged'].all())
    def test_nnls_monotone_history_and_kkt(self):
        rng=np.random.default_rng(4);x=rng.random((9,6));b=rng.random((4,6));e=encode(x,b,1000,1e-9,history=True)
        self.assertTrue(e['converged'].all());self.assertTrue(all(y<=x+1e-12 for x,y in zip(e['history'],e['history'][1:])))
        g=(e['coefficients']@b-x)@b.T
        self.assertLessEqual(np.max(np.abs(projected_gradient(e['coefficients'],g))/e['threshold'][:,None]),1.01)
    def test_zero_basis_duplicate_basis_and_zero_input(self):
        b=np.array([[1.,0.],[1.,0.],[0.,0.]]);x=np.array([[1.,.5],[0.,0.]])
        e=encode(x,b);np.testing.assert_allclose(e['reconstruction'],[[1.,0.],[0.,0.]]);self.assertTrue(e['converged'].all());np.testing.assert_array_equal(e['coefficients'][:,2],[0,0])
    def test_zero_budget_returns_unfinished(self):
        e=encode([[1.,1.]],np.eye(2),0);self.assertFalse(e['converged'].any());self.assertEqual(e['sweeps'],0);self.assertGreater(e['projected_gradient_inf'][0],e['threshold'][0])
    def test_plain_mu_zero_locking_not_stationary(self):
        a=controls()['zero_lock'];self.assertEqual(a['mu_loss'],.5);self.assertEqual(a['mu_coefficients'][0,1],0);self.assertEqual(a['mu_gradient'][0,1],-1);self.assertEqual(a['cd_loss'],0);self.assertTrue(a['cd_converged'])
    def test_scale_permutation_and_duplicate_nonidentifiability(self):
        rng=np.random.default_rng(1);a=rng.random((6,3));b=rng.random((3,5));s=np.array([.01,7.,3.]);np.testing.assert_allclose((a*s)@(b/s[:,None]),a@b);p=[2,0,1];np.testing.assert_allclose(a[:,p]@b[p],a@b)
        n,m=normalize(a,b);np.testing.assert_allclose(n@m,a@b);np.testing.assert_allclose(np.linalg.norm(m,axis=1),1)
        c=controls()['duplicate_basis'];np.testing.assert_array_equal(c['reconstruction_a'],c['reconstruction_b']);self.assertFalse(np.array_equal(c['coefficients_a'],c['coefficients_b']))
    def test_fit_monotone_nonnegative_deterministic_budget(self):
        x=np.random.default_rng(17).random((18,8));r=fit(x,3,outer=20,inner=2,tol=0);z=fit(x,3,outer=20,inner=2,tol=0)
        np.testing.assert_array_equal(r['basis'],z['basis']);self.assertTrue((r['basis']>=0).all() and (r['coefficients']>=0).all());h=r['history'];self.assertEqual(len(h),21);self.assertFalse(r['stationary_within_tolerance']);self.assertTrue(all(v['half_squared_error']<=u['half_squared_error']+1e-11 for u,v in zip(h,h[1:])));self.assertAlmostEqual(h[-1]['half_squared_error'],objective(x,r['coefficients'],r['basis']))
    def test_zero_training_point_stationary_not_useful(self):
        r=fit(np.zeros((4,3)),2,outer=3);self.assertEqual(r['history'][-1]['half_squared_error'],0);self.assertTrue(r['stationary_within_tolerance']);np.testing.assert_array_equal(r['basis'],np.zeros((2,3)))
    def test_strict_inputs(self):
        for x in [[],[1,2],[[float('nan')]],[[float('inf')]],[[-1]]]:
            with self.assertRaises(ValueError):matrix(x)
        for kwargs in [{'max_sweeps':-1},{'max_sweeps':1.5},{'tol':-1},{'tol':float('nan')},{'initial':[[0,0,0]]}]:
            with self.assertRaises(ValueError):encode([[1,2]],[[1,0]],**kwargs)
        for kwargs in [{'rank':0},{'rank':4},{'rank':True},{'rank':1,'inner':0},{'rank':1,'outer':-1}]:
            with self.assertRaises(ValueError):fit(np.ones((2,3)),**kwargs)
if __name__=='__main__':unittest.main()
