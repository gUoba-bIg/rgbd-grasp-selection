"""Read-only reconstruction of every sealed frame and all metric cells."""
import os,sys,json,hashlib,argparse
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v24 import ROOT,read,stat,sha,once,atomic,need,now
from scripts.p3_train_v30 import CONFIG,LOG,OUT,load_sealed,sealed,OLDLOG,OLDOUT,all_events
RESULT=ROOT/'results/P3/candidate_v30'

def run(stage,cold=False):
 import numpy as np
 from scipy.io import loadmat
 from PIL import Image
 c=read(CONFIG);need(sha(CONFIG)==read(LOG/'freeze.json')['sha256'],'Training protocol changed')
 for p,s in c['protected'].items():need(stat(ROOT/p)==s,'Protected prerequisite changed '+p)
 epoch=18
 calls=all_events();intents={r['call_id']:r for r in calls if r['kind']=='update_intent' and r['epoch']<=epoch};returns={r['call_id']:r for r in calls if r['kind']=='update_returned' and r['call_id'] in intents};generation_returns={r['call_id']:r for r in calls if r['kind']=='generation_returned'};score_returns={r['call_id']:r for r in calls if r['kind']=='score_returned'}
 need(set(returns)<=set(intents),'Update return lacks intent');steps={(intents[k]['epoch'],intents[k]['batch']) for k in returns if intents[k]['epoch']<=epoch}
 need(steps=={(e,b) for e in range(1,epoch+1) for b in range(560)},'Unique training steps incomplete')
 for e in range(1,epoch+1):need(stat((OLDOUT if e<=3 else OUT)/f'epoch_{e:02d}_final.pt')==read((OLDLOG if e<=3 else LOG)/f'epoch_{e:02d}.json')['checkpoint'],'Epoch checkpoint changed')
 frame_reports=[]
 for item in c[stage+'_evaluation']:
  scene,frame=item['scene'],item['frame'];d=RESULT/stage/f'scene_{scene:04d}_{frame:04d}';g=load_sealed(d/'generation.json');r=load_sealed(d/'score.json');cand=np.load(d/'candidates.npy');sample=np.load(d/'sampled_points.npy');pixels=np.load(d/'candidate_pixels.npy')
  need(g['call_id'] in generation_returns and generation_returns[g['call_id']]['candidates']==len(cand),'Generation lacks known return')
  for filename,key in [('candidates.npy','candidates'),('sampled_points.npy','sampled_points'),('candidate_pixels.npy','pixels')]:need(stat(d/filename)==g[key],'Generation array changed')
  for p,s in g['inputs'].items():need(stat(ROOT/p)==s,'Generation input changed')
  base=ROOT/c['data_root']/f'scenes/scene_{scene:04d}/realsense';depth=np.asarray(Image.open(base/f'depth/{frame:04d}.png'));m=loadmat(base/f'meta/{frame:04d}.mat');K=m['intrinsic_matrix'];factor=float(m['factor_depth'].item());vv,uu=np.indices(depth.shape);z=depth/factor;cloud=np.stack(((uu-K[0,2])*z/K[0,0],(vv-K[1,2])*z/K[1,1],z),axis=-1);valid=np.isfinite(cloud).all(-1)&(z>0)&(z<=2);indices=np.flatnonzero(valid);pick=np.random.RandomState(g['seed']).choice(len(indices),20000,replace=False);expected=cloud.reshape(-1,3)[indices[pick]].astype(np.float32)
  # Operation order differs from production (multiply then divide); compare at FP32 rounding resolution.
  need(np.allclose(sample,expected,rtol=2e-7,atol=1e-7),'Independent sampled geometry differs');need(pixels.shape==(len(cand),),'Candidate pixel count differs');need(np.allclose(cand[:,13:16],cloud.reshape(-1,3)[pixels].astype(np.float32),rtol=2e-7,atol=1e-7),'Decoded candidate seed provenance differs')
  from scripts.p3_filter_support_v30 import review_filter
  keep=review_filter(d,cand,cloud[valid].astype(np.float32));need(all(keep[row['index']] for row in r['records']),'Scored collision-filtered candidate')
  rows=r['records'];need(len(rows)==r['selected_count'] and len({x['index'] for x in rows})==len(rows),'Selected support duplicate/incomplete')
  for row in rows:
   need(row['call_id'] in score_returns and score_returns[row['call_id']]['score']==row['score'],'Used score lacks known return')
   need(load_sealed(d/'scores'/f'{row["index"]:04d}.json')==row,'Score record differs');need(np.array_equal(np.asarray(row['grasp'],dtype=cand.dtype),cand[row['index']]),'Candidate identity differs');need(row['score'] in [-1,.2,.4,.6,.8,1,1.2] and (not row['collision'] or row['score']==-1),'Score semantics differ')
  sorted_rows=sorted(enumerate(rows),key=lambda t:-np.float32(t[1]['grasp'][0]));conf=np.array([x['grasp'][0] for x in rows],dtype=np.float32);order=np.argsort(-conf);quality=np.asarray([rows[i]['score'] for i in order]);matrix=np.zeros((50,6))
  for k in range(1,51):
   for j,mu in enumerate([.2,.4,.6,.8,1,1.2]):matrix[k-1,j]=sum(0<q<=mu for q in quality[:k])/k
  need(np.array_equal(matrix,np.asarray(r['prefix_matrix'])) and float(matrix.mean())==r['prefix_mean'],'Prefix matrix differs')
  need(r['positive_mu04']==sum(0<q<=.4 for q in quality),'Positive count differs')
  if r['oracle_executed']:
   oracle=load_sealed(d/'upstream_oracle.json');need(sum(len(x) for x in oracle['scores'])==len(rows),'Oracle coverage differs')
   flat=[(score,collision,g) for gs,ss,cc in zip(oracle['groups'],oracle['scores'],oracle['collision']) for g,score,collision in zip(gs,ss,cc)]
   need(flat==[(q['score'],q['collision'],q['grasp']) for q in rows],'Independent oracle result differs')
  frame_reports.append(r)
 gate=load_sealed(RESULT/(stage+'_gate.json'));ps=len({r['scene'] for r in frame_reports if r['positive_mu04']>0});pc=sum(r['positive_mu04'] for r in frame_reports);expected_gate=ps>=c[stage+'_gate']['positive_scenes_min'] and pc>=c[stage+'_gate']['positive_candidates_min']
 need(gate['positive_scenes_mu04']==ps and gate['positive_candidates_mu04']==pc and gate['mean_frame_prefix_matrix']==float(np.mean([r['prefix_mean'] for r in frame_reports])),'Gate summary differs');need(gate['continue_training' if stage=='early' else 'P7_candidate_baseline_usable']==expected_gate,'Scientific decision differs')
 outputs={p.relative_to(ROOT).as_posix():stat(p) for p in (RESULT/stage).rglob('*') if p.is_file()};outputs.update({(RESULT/(stage+'_gate.json')).relative_to(ROOT).as_posix():stat(RESULT/(stage+'_gate.json'))})
 report=dict(at=now(),stage=stage,engineering_passed=True,scientific_usable=expected_gate,frames=len(frame_reports),unique_training_updates=len(steps),known_update_returns=len(returns),unknown_updates=sorted(set(intents)-set(returns)),files=outputs,new_training_updates=0,new_model_forwards=0,new_official_scoring_calls=0)
 if cold:
  old=read(LOG/(stage+'_readonly.json'));need(outputs==old['files'],'Cold reentry changed files');target=LOG/(stage+'_cold.json')
 else:target=LOG/(stage+'_readonly.json')
 if target.exists():need({k:v for k,v in read(target).items() if k!='at'}=={k:v for k,v in report.items() if k!='at'},'Repeated review result changed')
 else:once(target,report)
 print(json.dumps({k:v for k,v in report.items() if k!='files'}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['final'],required=True);p.add_argument('--cold',action='store_true');args=p.parse_args()
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run(args.stage,args.cold)
