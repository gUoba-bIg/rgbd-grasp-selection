"""Small synthetic optimizer/checkpoint cold-resume equality, counted separately."""
import sys,copy,uuid
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p5_common_v23 import *
from src.p5_quality_v23 import QualityNet,quality_loss
from src.checkpoints import CheckpointStore,fingerprint
import torch
def main():
    torch.set_num_threads(2);torch.manual_seed(20261023);a=QualityNet();opt=torch.optim.AdamW(a.parameters(),lr=.001);x=torch.randn(1,5,32,32);x[:,3]=.4;x[:,4]=1
    target=dict(q_valid=x[:,4:5],q_supervised=x[:,4:5],q_residual=torch.full_like(x[:,4:5],.001),q_anomaly=torch.zeros_like(x[:,4:5]))
    def step(m,o):
        cid=uuid.uuid4().hex;event('forward_intent',call_id=cid,network='quality',phase='engineering_resume_test');out=m(x);event('forward_returned',call_id=cid,network='quality',phase='engineering_resume_test');loss,_=quality_loss(out,target);o.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);oid=uuid.uuid4().hex;event('optimizer_intent',call_id=oid,phase='engineering_resume_test');o.step();event('optimizer_returned',call_id=oid,phase='engineering_resume_test',loss=float(loss.detach()));return float(loss.detach())
    step(a,opt);directory=RUN/'synthetic_resume_test';store=CheckpointStore(directory,fingerprint('P5synthetic'),fingerprint('synthetic32'),keep_last=2);store.save(a,opt,0,4,1,True);snapshot={p:stat(p) for p in directory.iterdir() if p.is_file()};b=QualityNet();other=torch.optim.AdamW(b.parameters(),lr=.001);loaded=store.load_latest(b,other);need(loaded['cursor']==dict(epoch=0,next_batch=4,global_step=1),'Resume cursor failed');need(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()),'Resume weights differ');la=step(a,opt);lb=step(b,other);need(la==lb and all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()),'Resumed optimizer trajectory differs');need(all(stat(p)==s for p,s in snapshot.items()),'Resume mutated checkpoint');once(LOG/'execution_tests.json',dict(passed=True,cold_model_optimizer_resume_exact=True,checkpoint_files_unchanged=True,synthetic_forward_returns=3,synthetic_optimizer_returns=3,new_research_updates=0))
if __name__=='__main__':main()
