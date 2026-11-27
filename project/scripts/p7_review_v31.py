"""Independent input/feature reconstruction, all labels and metrics, cold hashes."""
import os,sys,json,hashlib,argparse
from pathlib import Path
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/build/graspnetAPI_eval_v7'))
from scripts.p7_common_v31 import *

def observed_depth(c,case):
 import numpy as np
 from PIL import Image
 from scipy.io import loadmat
 d=ROOT/c['data_root']/f"scenes/scene_{case['scene']:04d}/realsense";meta=loadmat(d/'meta/0064.mat',variable_names=['factor_depth','intrinsic_matrix']);factor=float(meta['factor_depth'].item())
 with Image.open(d/'depth/0064.png') as im:size=im.size;z=np.asarray(im.resize((640,480),Image.Resampling.NEAREST),np.float32)/factor
 valid=np.isfinite(z)&(z>0);z=np.where(valid,np.minimum(z,2.)/2,0).astype(np.float32)*2;original=z.copy();v=valid.copy();condition=case['condition'];seed=int(hashlib.sha256(f"P5v23|20261023|{c['quality_screen_epoch']}|{case['scene']}|0064|{condition}".encode()).hexdigest()[:8],16);rng=np.random.RandomState(seed)
 if condition in ['shift3','joint']:z[:,3:]=original[:,:-3];z[:,:3]=0;v[:,3:]=valid[:,:-3];v[:,:3]=False
 if condition in ['noise5','joint']:z=np.where(v,np.clip(z+rng.normal(0,.005,z.shape).astype(np.float32),0,2),0).astype(np.float32);v&=z>0
 if condition in ['missing15','joint']:
  v&=rng.uniform(size=z.shape)>=.15;y=int(rng.randint(0,361));x=int(rng.randint(0,481));v[y:y+120,x:x+160]=False
 z[~v]=0
 return z,v,meta['intrinsic_matrix'],size

def independent_features(a,indices,voxels):
 import numpy as np
 from scipy.ndimage import maximum_filter
 cand=a['candidates'][indices];pixels=a['pixels'][indices];instances=a['instances'];sigma=a['sigma'];yy,xx=np.unravel_index(pixels,instances.shape);edge=np.zeros_like(instances,bool);edge[:-1]|=instances[:-1]!=instances[1:];edge[1:]|=instances[:-1]!=instances[1:];edge[:,:-1]|=instances[:,:-1]!=instances[:,1:];edge[:,1:]|=instances[:,:-1]!=instances[:,1:]
 # Two cross-structure dilations = Manhattan radius two, independent explicit shifts.
 expanded=np.zeros_like(edge)
 for dy in range(-2,3):
  for dx in range(-2,3):
   if abs(dy)+abs(dx)>2:continue
   sy=slice(max(0,-dy),min(480,480-dy));sx=slice(max(0,-dx),min(640,640-dx));ty=slice(max(0,dy),min(480,480+dy));tx=slice(max(0,dx),min(640,640+dx));expanded[ty,tx]|=edge[sy,sx]
 rows=[]
 for g,y,x in zip(cand,yy,xx):
  label=instances[y,x];mask=instances==label;mean=float(np.mean(sigma[mask],dtype=np.float64));fraction=float(mask.mean()) if label else 0.;q=(voxels-g[13:16])@g[4:13].reshape(3,3);height,width,depth=float(g[2]),float(g[1]),float(g[3]);vertical=(q[:,2]>-height/2)&(q[:,2]<height/2);fingers=(q[:,0]>depth-.06)&(q[:,0]<depth);left=vertical&fingers&(q[:,1]>-(width/2+.01))&(q[:,1]<-width/2);right=vertical&fingers&(q[:,1]<width/2+.01)&(q[:,1]>width/2);span=(q[:,1]>-(width/2+.01))&(q[:,1]<width/2+.01);bottom=vertical&span&(q[:,0]<=depth-.06)&(q[:,0]>depth-.07);approach=vertical&span&(q[:,0]<=depth-.07)&(q[:,0]>depth-.12);inner=vertical&fingers&(q[:,1]>=-width/2)&(q[:,1]<=width/2)
  lv=height*.06*.01/.005**3;bv=height*(width+.02)*.01/.005**3;av=height*(width+.02)*.05/.005**3;iv=height*.06*width/.005**3;ious=[(left|right|bottom|approach).sum()/(2*lv+bv+av+1e-6),left.sum()/(lv+1e-6),right.sum()/(lv+1e-6),bottom.sum()/(bv+1e-6),approach.sum()/(av+1e-6)];empty=bool(np.divide(float(inner.sum()),iv)<.01) if iv else False
  rows.append([g[0],g[1],g[3],g[15],g[10],sigma[y,x],mean,a['quality_valid'][y,x],a['quality_anomaly'][y,x],a['foreground_probability'][y,x],fraction,expanded[y,x],ious[0]>.01,empty,*ious])
 return np.asarray(rows,dtype=np.float32).reshape(-1,19)

def independent_probability(y,z):
 import numpy as np
 from scipy.special import expit
 p=expit(z.astype(np.float64));n=len(y);pos=y==1;neg=y==0;nll=sum(float(np.logaddexp(0,float(a)))-float(t)*float(a) for t,a in zip(y,z))/n;ece=0.;bins=[]
 for k in range(10):
  inds=[i for i,x in enumerate(p) if k/10<=x and (x<(k+1)/10 if k<9 else x<=1)];nn=len(inds);conf=float(np.mean(p[inds])) if nn else None;freq=float(np.mean(y[inds])) if nn else None
  if nn:ece+=nn/n*abs(conf-freq)
  bins.append(dict(lower=k/10,upper=(k+1)/10,n=nn,confidence=conf,frequency=freq))
 auc=sum(float((x>z[neg]).sum())+.5*float((x==z[neg]).sum()) for x in z[pos])/(pos.sum()*neg.sum());ap=0.;last=0
 for threshold in sorted(set(z.tolist()),reverse=True):
  selected=z>=threshold;tp=int(y[selected].sum());ap+=(tp-last)/pos.sum()*tp/selected.sum();last=tp
 return dict(n=n,positives=int(pos.sum()),nll=nll,brier=float(sum((a-float(b))**2 for a,b in zip(p,y))/n),ece=ece,auroc=float(auc),pr_auc=float(ap),bins=bins)

def close_values(a,b,path=''):
 import numpy as np
 if isinstance(a,dict):need(set(a)==set(b),'Metric keys differ '+path);[close_values(a[k],b[k],path+'/'+k) for k in a]
 elif isinstance(a,list):need(len(a)==len(b),'Metric length differs '+path);[close_values(x,y,path) for x,y in zip(a,b)]
 elif isinstance(a,(float,int)) and not isinstance(a,bool):need(np.isclose(a,b,rtol=1e-6,atol=1e-7),'Metric value differs '+path)
 else:need(a==b,'Metric result differs '+path)

def run(cold=False):
 import numpy as np,open3d as o3d
 from graspnetAPI import GraspGroup
 from scripts.verify_grasp_geometry_v10 import exact_indices
 from scripts.extract_mask_scores_v17 import decode_with_scores
 c=verify();previous=LOG/'readonly.json'
 if cold:
  r=read(previous);need(r['engineering_passed'],'No full numeric review')
  for p,s in r['files'].items():need(stat(ROOT/p)==s,'Cold file changed '+p)
  once(LOG/'cold.json',dict(at=now(),engineering_passed=True,files=r['files'],new_model_forwards=0,new_training_updates=0,new_official_scores=0));print('P7 cold read-only passed');return
 if previous.exists():
  r=read(previous)
  for p,s in r['files'].items():need(stat(ROOT/p)==s,'Readonly artifact changed '+p)
  print('Existing full P7 review verified');return
 b=Budget(c,'review');events=[json.loads(s) for s in (LOG/'calls.jsonl').read_text(encoding='utf-8').splitlines()];returned={r['call_id'] for r in events if r['kind'].endswith('_returned')};frames=0;candidate_total=0;datasets={s:dict(xs=[],ys=[],spans=[],n=0) for s in c['groups']}
 for case in cases(c):
  b.guard();verify_case_inputs(c,case);d=folder(case);g=load_sealed(d/'generation.json');need(stat(d/'observations.npz')==g['artifact'],'Observation artifact changed');a=np.load(d/'observations.npz');need(set(g['forward_call_ids']+g['identity_call_ids']+[g['corruption_call_id']])<=returned,'Used observation missing actual return');(z,v,K,size),_=invoke('corruption_audit',lambda:observed_depth(c,case),case=case);need(np.array_equal(z,a['observed_depth']) and np.array_equal(v,a['observed_valid']) and np.array_equal(K,a['K']) and np.array_equal(size,a['original_size']),'Independent observed input differs')
  heads={k:a[k] for k in ['foreground_probability','center_probability','offsets']};need(np.array_equal(decode_with_scores(heads,**c['decoder'])[0],a['instances']),'Predicted instance decode differs')
  from scripts.p7_pool_v31 import review_pool
  pool=load_sealed(d/'pool.json');cand=a['candidates'];expected=review_pool(d,a,GraspGroup);need(expected==pool['indices'],'Pool not outcome-blind filtered top20')
  pixels=a['pixels'];y,x=np.unravel_index(pixels,z.shape);oy=((2*y+1)*size[1])//960;ox=((2*x+1)*size[0])//1280;depth=z[y,x].astype(np.float64);xyz=np.column_stack(((ox-K[0,2])*depth/K[0,0],(oy-K[1,2])*depth/K[1,1],depth));need(np.allclose(xyz,cand[:,13:16],rtol=2e-7,atol=1e-7),'Candidate pixel provenance differs')
  f=load_sealed(d/'features.json');need(stat(d/'features.npz')==f['artifact'],'Feature artifact changed');features=np.load(d/'features.npz');vv,uu=np.indices(z.shape);oy=((2*vv+1)*size[1])//960;ox=((2*uu+1)*size[0])//1280;zz=z.astype(np.float64);cloud=np.stack(((ox-K[0,2])*zz/K[0,0],(oy-K[1,2])*zz/K[1,1],zz),-1);valid=v&(zz>0)&(zz<2);pc=o3d.geometry.PointCloud();pc.points=o3d.utility.Vector3dVector(cloud[valid].astype(np.float32));voxel=np.asarray(pc.voxel_down_sample(.005).points);need(np.array_equal(voxel,features['observed_voxel_points']),'Observed voxel cloud differs');expected_features=independent_features(a,expected,voxel);need(np.allclose(expected_features,features['features'],rtol=2e-6,atol=2e-7),'Independent feature values differ')
  labels=load_sealed(d/'labels.json');need(labels['indices']==expected and len(labels['records'])==len(expected),'Label coverage differs')
  ds=datasets[case['split']];ds['xs'].append(features['features']);ds['ys'].append(np.array([r['positive'] for r in labels['records']],np.float32));ds['spans'].append(dict(**case,start=ds['n'],end=ds['n']+len(expected)));ds['n']+=len(expected)
  for index,r in zip(expected,labels['records']):
   need(load_sealed(d/'labels'/f'{index:04d}.json')==r and r['call_id'] in returned,'Score lacks sealed actual return');need(r['grasp']==cand[index].tolist() and r['positive']==bool(0<r['score']<=.4 and not r['collision']),'Target semantics differ')
  if (d/'upstream_oracle.json').exists():
   o=load_sealed(d/'upstream_oracle.json');need(o['call_id'] in returned,'Oracle lacks return')
   for gs,ss,cc in zip(o['value']['groups'],o['value']['scores'],o['value']['collisions']):
    for row,s,co in zip(gs,ss,cc):need(any(r['grasp']==row and r['score']==s and r['collision']==co for r in labels['records']),'Oracle mismatch')
  frames+=1;candidate_total+=len(expected);a.close();features.close()
 fit=load_sealed(BASE/'fit_result.json')
 counts={s:dict(n=d['n'],positive=int(np.concatenate(d['ys']).sum()),negative=int(d['n']-np.concatenate(d['ys']).sum())) for s,d in datasets.items()};support=load_sealed(BASE/'class_support.json');need(counts==support['counts'] and support['adequate']==all(v['positive']>=30 and v['negative']>=30 for v in counts.values()),'Independent class support differs');need(fit['fitted']==support['adequate'],'Fit/support mismatch')
 if fit['fitted']:
  identity=load_sealed(BASE/'model_identity.json');need(identity['passed'],'Torch checkpoint/export identity missing')
  w=load_sealed(BASE/'selected_weights.json');need(stat(BASE/'selected_weights.npz')==w['artifact'],'Selected weights changed');weights=np.load(BASE/'selected_weights.npz');train=np.load(BASE/'predictions/train.npz');mean=train['features'].mean(0,dtype=np.float64).astype(np.float32);scale=np.maximum(train['features'].std(0,dtype=np.float64),1e-6).astype(np.float32)
  need(fit['simple_penalty_train_only']==float(np.ptp(train['features'][:,0])+1.),'Simple baseline scale not train-only');need(fit['selected_epoch']==min(fit['history'],key=lambda r:r['validation_nll'])['epoch'],'Selected epoch differs from prespecified validation NLL')
  for split in ['train','calibration','validation']:
   b.guard();p=BASE/'predictions'/f'{split}.npz';r=load_sealed(p.with_suffix('.json'));need(stat(p)==r['artifact'] and r['call_id'] in returned,'Prediction cache/return mismatch');a=np.load(p);need(np.array_equal(a['features'],np.concatenate(datasets[split]['xs'])) and np.array_equal(a['labels'],np.concatenate(datasets[split]['ys'])) and r['spans']==datasets[split]['spans'],'Fit dataset not identical to sealed inputs and true outcomes');need(np.array_equal(mean,a['mean']) and np.array_equal(scale,a['scale']),'Standardization not train-only');X=(a['features']-mean)/scale
   def forward():
    h=X
    for layer in ['0','2','4']:
     h=h@weights[layer+'.weight'].T+weights[layer+'.bias']
     if layer!='4':h=np.maximum(h,0)
    return h.reshape(-1)
   z,_=invoke('risk_audit_forward',forward,split=split);need(np.allclose(z,a['logits'],atol=2e-5,rtol=2e-5),'Independent NumPy risk logits differ');y=a['labels'];stored=a['logits'];report=fit['reports'][split];close_values(independent_probability(y,stored),report['uncalibrated']);close_values(independent_probability(y,stored/fit['temperature']),report['calibrated'])
   # Independently recompute every fixed top-k frame, including empty pools.
   simple=a['features'][:,0];penalized=simple-fit['simple_penalty_train_only']*((a['features'][:,12]>0)|(a['features'][:,13]>0))
   for name,scores in dict(raw=simple,collision_penalty=penalized,risk=stored,calibrated=stored/fit['temperature']).items():
    frame_rows=[]
    for span in r['spans']:
     start,end=span['start'],span['end'];indices=sorted(range(start,end),key=lambda i:(-float(scores[i]),i));row=dict(scene=span['scene'],condition=span['condition'],**{f'p{k}':sum(float(y[i]) for i in indices[:k])/k for k in [1,5,10]});frame_rows.append(row)
    scene_rows=[]
    for scene in c['groups'][split]:
     rows=[row for row in frame_rows if row['scene']==scene];scene_rows.append(dict(scene=scene,**{f'p{k}':sum(row[f'p{k}'] for row in rows)/len(rows) for k in [1,5,10]},clean_p5=next(row['p5'] for row in rows if row['condition']=='clean')))
    rr=dict(**{f'p{k}':sum(row[f'p{k}'] for row in scene_rows)/len(scene_rows) for k in [1,5,10]},clean_p5=sum(row['clean_p5'] for row in scene_rows)/len(scene_rows),scenes=scene_rows,frames=frame_rows);close_values(rr,report['ranking'][name])
   a.close()
  val=fit['reports']['validation'];rank=val['ranking'];gates=dict(ranking_gain=rank['risk']['p5']-max(rank[n]['p5'] for n in ['raw','collision_penalty'])>=.01,clean_noninferiority=rank['risk']['clean_p5']>=max(rank[n]['clean_p5'] for n in ['raw','collision_penalty'])-.01,calibration_nll=val['calibrated']['nll']<=val['uncalibrated']['nll']+.001,calibration_ece=val['calibrated']['ece']<=.1);need(gates==fit['gates'] and all(gates.values())==fit['scientific_usable'],'Scientific decision mismatch')
  cal=np.load(BASE/'predictions/calibration.npz');T=fit['temperature'];need(.05<=T<=20,'Temperature outside preregistered bounds');best=fit['reports']['calibration']['calibrated']['nll']
  for alternative in [.05,20.,max(.05,T*.999),min(20.,T*1.001)]:
   z=cal['logits']/alternative;nll=float(np.mean(np.logaddexp(0,z.astype(np.float64))-cal['labels']*z));need(best<=nll+1e-6,'Calibration does not minimize its independent NLL')
 b.save('complete');paths=[p for base in [BASE,ROOT/'experiments/risk_v31'] for p in base.rglob('*') if p.is_file()];paths += [LOG/'calls.jsonl'];files={p.relative_to(ROOT).as_posix():stat(p) for p in paths};once(previous,dict(at=now(),engineering_passed=True,conditions=frames,candidates=candidate_total,scientific_usable=fit['scientific_usable'],fitted=fit['fitted'],files=files));print('All P7 observations, labels, features and metrics reviewed',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cold',action='store_true');args=p.parse_args()
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run(args.cold)
