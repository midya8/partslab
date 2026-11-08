import unittest
from copy import deepcopy
from unittest.mock import patch
import numpy as np
from partslab import data
from partslab.experiment import train,evaluate,predict,encoding,PROTOCOL
from partslab.io import ROOT,read,digest,dumps


class ExperimentTests(unittest.TestCase):
    def test_full_training_reproduction_forbids_test_loader_and_has_no_labels(self):
        original=data.load;calls=[]
        def guarded(split):
            calls.append(split)
            if split!='train':raise AssertionError('test access during training')
            return original(split)
        with patch('partslab.data.load',side_effect=guarded):models,guard=train()
        self.assertEqual(calls,['train']);saved=read(ROOT/'artifacts/models.json')
        for name,a in models['arms'].items():
            b=saved['arms'][name];self.assertEqual(a['chosen'],b['chosen'])
            for r,s in zip(a['runs'],b['runs']):
                np.testing.assert_allclose(r['basis'],s['basis'],atol=2e-10,rtol=2e-10);self.assertAlmostEqual(r['reencoded_training']['mse'],s['reencoded_training']['mse'],places=11)
        self.assertEqual(guard['models_sha256'],digest(models));self.assertEqual(models['train_rows'],3823)
    def test_frozen_guard_before_test_opens(self):
        m=read(ROOT/'artifacts/models.json');g=read(ROOT/'artifacts/guard.json')
        for mutate in ['model','protocol']:
            bad=deepcopy(m)
            if mutate=='model':bad['mean'][0]+=.1
            else:bad['protocol_sha256']='wrong'
            with patch('partslab.data.load',side_effect=AssertionError('must reject before test access')):
                with self.assertRaises(ValueError):evaluate(bad,g)
    def test_holdout_all_rows_and_controls_reproduced(self):
        m=read(ROOT/'artifacts/models.json');g=read(ROOT/'artifacts/guard.json');r=evaluate(m,g);s=read(ROOT/'artifacts/report.json');self.assertEqual(r['test_rows'],1797);self.assertAlmostEqual(r['mean_mse'],s['mean_mse'],places=13)
        for name,a in r['arms'].items():
            for key in ['pca_row_mse','svd_row_mse']:np.testing.assert_allclose(a[key],s['arms'][name][key],atol=1e-13)
            for v,w in zip(a['nmf_runs'],s['arms'][name]['nmf_runs']):
                np.testing.assert_allclose(v['row_mse'],w['row_mse'],atol=1e-12);self.assertEqual(v['converged_rows'],w['converged_rows'])
    def test_training_only_selection_all_seeds_and_ranks_remain(self):
        m=read(ROOT/'artifacts/models.json');self.assertEqual(sorted(map(int,m['arms'])),PROTOCOL['ranks'])
        for a in m['arms'].values():
            self.assertEqual([r['seed'] for r in a['runs']],PROTOCOL['seeds']);self.assertEqual(a['chosen'],min(range(2),key=lambda i:(a['runs'][i]['reencoded_training']['mse'],a['runs'][i]['seed'])))
            for r in a['runs']:self.assertEqual(len(r['history']),101);self.assertFalse(r['stationary_within_tolerance'])
    def test_pca_orthogonal_reconstruction_and_unclipped_scores(self):
        m=read(ROOT/'artifacts/models.json');x=data.load('test');mean=np.array(m['mean'])
        for a in m['arms'].values():
            pc=np.array(a['pca_components']);np.testing.assert_allclose(pc@pc.T,np.eye(len(pc)),atol=1e-12);re=(x-mean)@pc.T@pc+mean
            np.testing.assert_allclose((x-re)@pc.T,0,atol=1e-12);self.assertTrue((re<0).any())
    def test_custom_prediction_uses_new_pixels(self):
        m=read(ROOT/'artifacts/models.json');x=np.zeros(64);x[12]=1
        for rank in PROTOCOL['ranks']:
            r=predict(x,m,rank);a=np.array(r['coefficients']);b=np.array(m['arms'][str(rank)]['runs'][m['arms'][str(rank)]['chosen']]['basis']);np.testing.assert_allclose(a@b,r['reconstruction']);self.assertAlmostEqual(np.mean((x-a@b)**2),r['nmf_mse']);self.assertTrue(r['nnls_converged'])
        for bad in [[0]*63,[float('nan')]*64,[2]*64]:
            with self.assertRaises(ValueError):predict(bad,m,16)
    def test_dataset_manifest_and_scale(self):
        self.assertEqual(data.load('train').shape,(3823,64));self.assertEqual(data.load('test').shape,(1797,64))
        with self.assertRaises(ValueError):data.load('validation')
    def test_json_refuses_nan(self):
        with self.assertRaises(ValueError):dumps({'x':float('nan')})
if __name__=='__main__':unittest.main()
