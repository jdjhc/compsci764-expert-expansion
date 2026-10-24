import unittest
from unittest.mock import patch
import numpy as np
import run_neural as s

class NeuralChecks(unittest.TestCase):
 def test_all_three_layer_gradients(self):
  rng=np.random.default_rng(71);net=s.Network(23,(6,5,4,3),np.float64)
  for i in (1,3,5):net.p[i]+=0.13
  x=rng.normal(size=(9,6));y=rng.integers(0,3,9);loss,grads=net.loss_grad(x,y)
  for a,g in zip(net.p,grads):
   for j in np.ndindex(a.shape):
    old=a[j];a[j]=old+1e-6;hi=net.loss_grad(x,y)[0];a[j]=old-1e-6;lo=net.loss_grad(x,y)[0];a[j]=old
    self.assertAlmostEqual(g[j],(hi-lo)/2e-6,places=6)
 def test_network_size_and_clone_isolation(self):
  net=s.Network();self.assertEqual(sum(p.size for p in net.p),109386)
  other=net.clone();other.p[0][0,0]+=1;self.assertNotEqual(other.p[0][0,0],net.p[0][0,0]);self.assertEqual(net.nbytes(),437544)
 def test_alignment_and_near_zero(self):
  a=[np.array([1.,2.])];self.assertAlmostEqual(s.alignment(a,[-a[0]],1e-8)[0],1)
  self.assertEqual(s.alignment(a,a,1e-8)[0],0);self.assertEqual(s.alignment([np.zeros(2)],a,1e-8),(0.,False))
 def test_priority_memory_and_test_rejection(self):
  mem=s.Memory(3);b=dict(x=np.ones((5,784),dtype=np.float32),y=np.arange(5,dtype=np.int64),ids=np.arange(5,dtype=np.int64));mem.add(b,np.array([.1,.8,.3,.9,.7]));self.assertEqual(set(mem.ids),{1,3,4})
  b['ids']=-b['ids']-1
  with self.assertRaises(AssertionError):mem.add(b,np.ones(5))
 def test_future_and_test_data_rejected(self):
  cfg=s.Config(hidden1=4,hidden2=3);learner=s.Learner('dual',cfg,0)
  b=dict(x=np.ones((4,784),dtype=np.float32),y=np.arange(4,dtype=np.int64),ids=-np.arange(1,5,dtype=np.int64))
  with self.assertRaises(AssertionError):learner.train_window(b,np.ones(4),0)
  b['ids']=np.arange(4,8);learner.memory.add(b,np.arange(4));b['ids']=np.arange(4)
  with self.assertRaises(AssertionError):learner.train_window(b,np.ones(4),0)
 def test_early_allocation_preserves_old_anchor(self):
  cfg=s.Config(hidden1=4,hidden2=3,persistence=1,warmup_windows=1,novelty_threshold=1)
  learner=s.Learner('novelty',cfg,0)
  for t,level in enumerate([0,1]):
   b=dict(x=np.full((8,784),level,dtype=np.float32),y=np.zeros(8,dtype=np.int64),ids=np.arange(t*8,(t+1)*8));learner.train_window(b,np.arange(8)/10,t)
  self.assertEqual(learner.route(np.array([[0]*784,[1]*784],dtype=np.float32)).tolist(),[0,1])
 def test_retained_evidence_changes_delayed_decision(self):
  counts={}
  for policy in ['dual','dual_latched']:
   cfg=s.Config(hidden1=4,hidden2=3,updates=1,warmup_windows=1,novelty_threshold=1)
   learner=s.Learner(policy,cfg,0)
   with patch('run_neural.alignment',side_effect=[(v,True) for v in [1,0,0,1,1,1]]):
    for t,level in enumerate([0,1,1,1,1,1,1]):
     b=dict(x=np.full((8,784),level,dtype=np.float32),y=np.zeros(8,dtype=np.int64),ids=np.arange(t*8,(t+1)*8));learner.train_window(b,np.linspace(.1,.9,8),t)
   counts[policy]=len(learner.experts)
  self.assertEqual(counts,{'dual':1,'dual_latched':2})
 def test_stream_disjoint_images_and_matched_priorities(self):
  cfg=s.Config(batch_size=10,examples_per_segment=10);x=np.zeros((60000,784),dtype=np.float32);y=np.arange(60000,dtype=np.int64)%10
  a,_,h,ix=s.build_stream(x,y,'contrast',3,cfg);b,_,h2,ix2=s.build_stream(x,y,'contrast',3,cfg)
  self.assertEqual(len(np.unique(ix)),60);self.assertTrue(np.all(ix>=10000));self.assertEqual(h,h2);np.testing.assert_equal(ix,ix2)
  for aa,bb in zip(a,b):np.testing.assert_equal(aa[1],bb[1])

if __name__=='__main__':unittest.main(verbosity=2)
