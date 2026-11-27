"""Focused invariants for a new P5 model; all test network calls separately logged."""
import os,sys,copy,unittest,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import torch,numpy as np
from src.p5_quality_v23 import *
from src.p5_metrics_v23 import ap_binary,diagnostics,summary
from scripts.p5_common_v23 import event,once,LOG,now

def hook(model,name):
    def pre(m,a):m.cid=uuid.uuid4().hex;event('forward_intent',id=m.cid,network=name,phase='engineering_tests')
    def post(m,a,o):event('forward_returned',id=m.cid,network=name,phase='engineering_tests')
    model.register_forward_pre_hook(pre);model.register_forward_hook(post)
class Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4);torch.manual_seed(23);self.x=torch.randn(5,64,80);self.x[3]=.4;self.x[4]=1.;self.x[3,0]=0;self.x[4,0]=0;self.r=dict(scene=3,frame='0000')
    def test_deterministic_and_immutable(self):
        before=self.x.clone();a,t,m=corrupt(self.x,self.r,0,'joint');b,u,n=corrupt(self.x,self.r,0,'joint');self.assertTrue(torch.equal(a,b) and torch.equal(before,self.x) and m==n);self.assertTrue(torch.equal(a[:3],self.x[:3]));self.assertTrue(torch.equal(t['q_supervised'],self.x[4:5]*a[4:5]));self.assertTrue(torch.equal(t['q_residual'],(a[3:4]-self.x[3:4])*2))
    def test_clean_reference(self):
        a,t,m=corrupt(self.x,self.r,0,'clean');self.assertTrue(torch.equal(a,self.x));self.assertEqual(t['q_residual'].count_nonzero(),0);self.assertEqual(t['q_anomaly'].count_nonzero(),0)
    def test_shift_no_wrap(self):
        a,t,m=corrupt(self.x,self.r,0,'shift3');self.assertEqual(a[4,:,:3].sum(),0);self.assertTrue(torch.equal(a[3,:,3:],self.x[3,:,:-3]));self.assertEqual(m['shift_px'],3)
    def test_missing_targets(self):
        a,t,m=corrupt(self.x,self.r,0,'missing15');self.assertTrue(torch.equal(a[3:4]*2,(self.x[3:4]*2)*a[4:5]));self.assertTrue(bool((a[3:4][a[4:5]==0]==0).all()))
    def test_pair_and_gradient_isolation(self):
        q=QualityNet();fixed=GatedModel(q,'fixed');quality=copy.deepcopy(fixed);quality.arm='quality';hook(fixed.backbone,'segmentation');hook(fixed.quality,'quality');hook(quality.backbone,'segmentation');hook(quality.quality,'quality')
        a=fixed(self.x[None]);b=quality(self.x[None]);self.assertTrue(all(torch.equal(a[k],b[k]) for k in ['foreground','centers','offsets']));self.assertEqual(sum(p.numel() for p in fixed.parameters()),sum(p.numel() for p in quality.parameters()));quality.strength=.5;c=quality(self.x[None]);self.assertTrue(bool(((c['effective_gate']>=.5)&(c['effective_gate']<=1)).all()));c['foreground'].sum().backward();self.assertTrue(all(p.grad is None for p in quality.quality.parameters()));self.assertTrue(any(p.grad is not None for p in quality.backbone.parameters()))
    def test_quality_loss_finite_and_gradients(self):
        q=QualityNet();hook(q,'quality');x,t,m=corrupt(self.x,self.r,0,'joint');out=q(x[None]);loss,_=quality_loss(out,{k:v[None] for k,v in t.items()});self.assertTrue(bool(torch.isfinite(loss)));loss.backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in q.parameters()))
    def test_average_precision_ties(self):
        self.assertEqual(ap_binary([1,0],[.5,.5]),.5);self.assertAlmostEqual(ap_binary([1,0,1],[.9,.8,.7]),5/6);self.assertIsNone(ap_binary([0,0],[.2,.1]))
    def test_gate_schedule(self):
        self.assertEqual(strength(1),0);self.assertEqual(strength(80),0);self.assertEqual(strength(240),.5);self.assertEqual(strength(800),.5)
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks));once(LOG/'component_tests.json',dict(at=now(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),new_research_calls=0));sys.exit(0 if result.wasSuccessful() else 1)
