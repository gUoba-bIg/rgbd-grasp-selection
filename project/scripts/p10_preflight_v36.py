"""Bounded old-development replay; never opens held-out images."""
import sys,json,time,uuid
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p10_common_v36 import *
PF=LOG/'preflight';START=time.monotonic();pc={}
def call(kind,fn,**kw):
 limits=dict(grasp_forward=1,quality_forward=1,segmentation_forward=1,corruption=1,official_score=6,upstream_oracle=1)
 need(time.monotonic()-START<600,'Preparation time limit');pc[kind]=pc.get(kind,0)+1;need(pc[kind]<=limits[kind],'Preparation call cap');cid=uuid.uuid4().hex
 def write(kind,**data):
  with (PF/'calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),kind=kind,call_id=cid,**data))+'\n');f.flush();os.fsync(f.fileno())
 write(kind+'_intent',**kw);t=time.monotonic();v=fn();write(kind+'_returned',seconds=time.monotonic()-t);return v,cid
def run():
 need(not (PF/'checks.json').exists(),'Preparation already sealed');PF.mkdir(parents=True,exist_ok=True);checks=[]
 from scripts.p9_close_recovery_v35r2 import verify_close
 verify_close();need(load(ROOT/'logs/P9/risk_seeds_v35/root_final_review.json')['passed'],'P9 root closure missing')
 import scripts.p10_generate_v36 as gen
 c7=read(ROOT/'configs/risk_calibration_v31.json');c=dict(c7,limits=dict(c7['limits'],rss_bytes=16*1024**3),phase_seconds=dict(generation=600));r=dict(split='validation',scene=2,frame=64,condition='clean');saved=ROOT/'results/P7/risk_v31/cases/validation/scene_0002_0064_clean/observations.npz'
 oldgen=read(saved.parent/'generation.json');need(stat(saved)==oldgen['artifact'],'Old source observation changed')
 sealed(PF/'generation/assets.json',dict(source='accepted old P7 only',formal_test_opened=False))
 gen.verify=lambda:c;gen.BASE=PF/'generation';gen.DATA=ROOT/c7['data_root'];gen.cases=lambda c:[r];gen.folder=lambda case:PF/'generation'/key(case);gen.invoke=call;gen.verify_case_inputs=lambda c,r:None;gen.status=lambda *a,**k:None;gen.run()
 import numpy as np,torch
 actual=PF/'generation'/key(r)/'observations.npz'
 with np.load(saved) as a,np.load(actual) as b:
  need(set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files),'New generation differs from accepted old frame')
 checks.append('Actual three upstream model forwards exactly replay old development frame')
 from src.risk_observations_v31 import prepare
 from scripts.p10_review_v36 import independent_depth
 from src.p5_quality_v23 import CORE,GRADED
 for cond in CORE+GRADED:
  rr=dict(r,condition=cond);x,K,size,_=prepare(ROOT/c7['data_root'],2,64,cond,c7['quality_screen_epoch']);z,v,kk,ss=independent_depth(c7,rr);need(np.array_equal(x[3]*2,z) and np.array_equal(x[4].astype(bool),v) and np.array_equal(K,kk) and size==ss,'Independent graded replay failed '+cond);checks.append('independent corruption '+cond)
 from src.p8_math_v32 import grasp_ap,probability,ranked_candidates,logits
 from scripts.p8_review_v32 import independent_probability,independent_selection
 from scripts.p7_review_v31 import close_values,independent_features
 from scripts.p10_geometry_v36 import Geometry,check_oracle
 from src.grasp_observed_filter_v29 import independent_mask
 for scores,col in [([],[]),([.2,-1.,.6],[False,False,True])]:
  prefix=[[sum(0<s<=mu and not co for s,co in list(zip(scores,col))[:k])/k for mu in [.2,.4,.6,.8,1.,1.2]] for k in range(1,51)];need(grasp_ap(scores,col)['prefix']==prefix,'AP fixed denominator differs');checks.append('AP empty/incomplete denominator '+str(len(scores)))
 for n in [0,14,15,16,151]:z=np.round(np.linspace(-5,5,n));y=np.arange(n)%2;close_values(probability(y,z),independent_probability(y,z));checks.append('ECE15 empty/ties '+str(n))
 a=np.zeros((4,17),np.float32);norm,ix=ranked_candidates(a,np.array([True,False,True,True]),np.array([1.,5.,1.,0.]));need(ix.tolist()==[0,2,3] and norm[:,0].tolist()==[3,2,1],'Stable ranking tie handling');checks.append('Deterministic ranking and ties')
 c9=read(ROOT/'configs/p9_risk_seeds_v35.json');geo=Geometry(c9)
 with np.load(saved) as a:obs={k:a[k].copy() for k in a.files}
 obs['candidates'][:,1]=np.clip(obs['candidates'][:,1],0,.1);cand=obs['candidates'];state=geo.scene(r);p8=ROOT/'results/P8/ablation_v32/cases/validation/scene_0002_0064_clean/whole';p9=ROOT/'results/P9/risk_seeds_v35/cases/scene_0002_0064_clean'
 with np.load(artifact(p8/'features.npz')) as a:f={k:a[k].copy() for k in a.files}
 need(np.array_equal(independent_mask(cand,f['voxels10']),f['keep']),'Observed filter replay differs')
 expected=independent_features(obs,np.arange(len(cand)),f['voxels05']);need(np.allclose(expected,f['features'],rtol=2e-6,atol=2e-7),'Independent feature replay differs');checks.append('All old frame features and observed collision masks')
 with np.load(artifact(p9/'predictions.npz')) as a:z=a['20261031'].copy()
 sel=load(p9/'selection.json')['methods']['20261031'];need(geo.select(cand,f['keep'],z,state)[0]==independent_selection(cand,f['keep'],z,geo,state)==sel,'Selection geometry replay differs');checks.append('New bounded geometry adapter and independent selection identity')
 labels={v['index']:v for v in load(p9/'labels.json')['records']}
 for ix in sorted(labels)[:6]:rec=labels[ix];v,cid=call('official_score',lambda:geo.score(cand[ix],rec['object_slot'],state),scene=2,index=ix);need(all(v[k]==rec[k] for k in v),'Physical adapter label mismatch')
 checks.append('Six actual official labels match accepted results with new mesh cache')
 norm,orig=ranked_candidates(cand,f['keep'],z);oracle,cid=call('upstream_oracle',lambda:geo.oracle(norm,state),scene=2);check_oracle(oracle,norm,orig,labels,sel);checks.append('Actual full upstream reference agrees')
 with np.load(artifact(ROOT/'results/P7/risk_v31/predictions/train.npz')) as a:mean=a['mean'];scale=a['scale']
 from scripts.p7_fit_v31 import model
 for seed in c9['seeds']:
  with np.load(artifact(ROOT/f'results/P9/risk_seeds_v35/seeds/{seed}/weights.npz')) as a:w={k:a[k].copy() for k in a.files}
  n=model();n.load_state_dict({k:torch.from_numpy(v) for k,v in w.items()});n.eval()
  with torch.inference_mode():pred=n(torch.from_numpy((f['features']-mean)/scale)).reshape(-1).numpy()
  need(np.allclose(logits(f['features'],w,mean,scale),pred,rtol=2e-5,atol=2e-5),'Independent risk network output differs');checks.append('Risk-network export '+str(seed))
 sealed(PF/'checks.json',dict(at=now(),passed=True,checks=checks,actual_calls=pc,extra_corruption_replays=24,extra_old_risk_identity_forwards=3,extra_independent_risk_identity_forwards=3,official_test_images_opened=0,seconds=time.monotonic()-START));print('P10 preflight passed:',len(checks),flush=True)
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()
