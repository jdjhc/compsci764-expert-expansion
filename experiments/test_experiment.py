"""Meaningful checks of gradients, online isolation, budgets and routing assumptions."""
import unittest
import numpy as np
from run_experiment import Config, Learner, Memory, calibrate, loss_grad, conflict, make_data, run_one

class ExperimentChecks(unittest.TestCase):
    def test_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(41)
        x, y, w = rng.normal(size=(17,2)), rng.integers(0,2,17), rng.normal(size=3)
        analytic = loss_grad(w,x,y)[1]
        numerical = np.zeros(3)
        for i in range(3):
            step = np.zeros(3); step[i] = 1e-6
            numerical[i] = (loss_grad(w+step,x,y)[0]-loss_grad(w-step,x,y)[0])/2e-6
        np.testing.assert_allclose(analytic, numerical, atol=1e-8, rtol=1e-6)

    def test_conflict_direction_and_zero_gradient(self):
        self.assertAlmostEqual(conflict(np.array([1.,0]),np.array([-1.,0]),1e-6)[0],1.)
        self.assertEqual(conflict(np.array([1.,0]),np.array([1.,0]),1e-6)[0],0.)
        self.assertEqual(conflict(np.zeros(2),np.ones(2),1e-6),(0.,False))

    def test_simple_domain_is_learnable(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(1024,2)); y = (x[:,0]>.0).astype(float)
        w = np.zeros(3)
        for _ in range(120):
            w -= .2*loss_grad(w,x,y)[1]
        self.assertGreater(np.mean((x@w[:2]+w[2]>=0)==y),.98)

    def test_replay_rejects_evaluation_samples(self):
        cfg=Config(); _, tests, _, _=make_data('benign_shift',0,cfg)
        with self.assertRaises(AssertionError):
            Memory(16).add(tests[0],np.zeros(cfg.test_size))

    def test_replay_is_uniform_priority_reservoir_and_bounded(self):
        cfg=Config(); stream, _, priority, _=make_data('benign_shift',0,cfg)
        mem=Memory(71)
        for b,p in zip(stream[:5],priority[:5]): mem.add(b,p)
        all_p=np.concatenate(priority[:5])
        expected=np.argsort(all_p,kind='stable')[-71:]
        self.assertEqual(set(mem.ids),set(expected))
        self.assertEqual(len(mem.ids),71)

    def test_learner_rejects_nonpast_replay_and_test_batches(self):
        cfg=Config(); stream, tests, priority, _=make_data('benign_shift',0,cfg)
        learner=Learner('dual',cfg,0)
        with self.assertRaises(AssertionError): learner.train_window(tests[0],np.zeros(cfg.test_size),0)
        learner.memory.add(stream[1],priority[1])
        with self.assertRaises(AssertionError): learner.train_window(stream[0],priority[0],0)

    def test_cap_and_reproducibility(self):
        cfg=Config(expert_cap=2, windows_per_segment=8, warmup_windows=8, test_size=128)
        cfg.novelty_threshold,_=calibrate(cfg)
        data=make_data('conflicting_shift',0,cfg)
        a,ta,_=run_one('novelty','conflicting_shift',0,cfg,data)
        b,tb,_=run_one('novelty','conflicting_shift',0,cfg,data)
        self.assertEqual(ta,tb)
        self.assertEqual(a['final_accuracy'],b['final_accuracy'])
        self.assertLessEqual(a['experts'],2)
        self.assertTrue(all(t['replay_count']<=cfg.replay_size for t in ta))
        self.assertTrue(all(t['prototypes']<=cfg.prototype_cap for t in ta))
        self.assertGreater(a['blocked_events'],0)

    def test_early_novel_expansion_preserves_old_routing(self):
        cfg=Config(persistence=1, gallery_persistence=3, windows_per_segment=8, warmup_windows=8, test_size=128)
        cfg.novelty_threshold,_=calibrate(cfg)
        stream,_,priority,_=make_data('conflicting_shift',0,cfg)
        learner=Learner('novelty',cfg,0)
        for t in range(8): learner.train_window(stream[t],priority[t],t)
        learner.train_window(stream[8],priority[8],8)
        self.assertEqual(len(learner.experts),2)
        self.assertEqual(learner.route(np.array([[0.,0.],[5.,0.]])).tolist(),[0,1])

    def test_identical_contexts_cannot_route_opposite_labels(self):
        cfg=Config()
        learner=Learner('dual',cfg,0)
        learner.anchors=[np.array([0.,0.])];learner.anchor_experts=[0]
        learner.experts=[np.array([1.,.6,0.])]
        rng=np.random.default_rng(3)
        x=rng.normal(size=(500,2)); c=rng.normal(size=(500,2)); y=(x[:,0]+.6*x[:,1]>=0).astype(float)
        a={'x':x,'c':c,'y':y}; b={'x':x,'c':c,'y':1-y}
        self.assertAlmostEqual((learner.evaluate(a)+learner.evaluate(b))/2,.5)

if __name__=='__main__': unittest.main(verbosity=2)
