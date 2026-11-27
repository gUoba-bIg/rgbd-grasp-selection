"""One supervised baseline reuse pilot; never trains or chains another experiment."""
import os, sys, json, hashlib, time, threading, argparse
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='4'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.pipeline import project_lock
from scripts import evaluate_segmentation_baseline_v16 as old
import numpy as np
from PIL import Image
from scipy.io import loadmat
CONFIG=ROOT/'configs/baseline_usability_v20b.json'
FREEZE=ROOT/'logs/P4/baseline_usability_v20b/freeze.json'
def require(v,m):
    if not v: raise RuntimeError(m)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def state(p):
    p=Path(p);s=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return dict(sha256=h.hexdigest(),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f: json.dump(d,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def snapshots(c): return {p:state(ROOT/p) for p in c['protected_files']}
def pairs(pred,gt):
    pi=np.unique(pred[pred>0]);gi=np.unique(gt[gt>0]);adj=[]
    for a in pi:
        mask=pred==a;edges=[]
        for j,b in enumerate(gi):
            both=int(np.count_nonzero(mask&(gt==b)));union=int(np.count_nonzero(mask|(gt==b)))
            if 2*both>=union: edges.append(j)
        adj.append(edges)
    assigned={}
    def visit(a,seen):
        for b in adj[a]:
            if b in seen:continue
            seen.add(b)
            if b not in assigned or visit(assigned[b],seen):assigned[b]=a;return True
        return False
    for a in range(len(pi)):visit(a,set())
    return [[int(pi[a]),int(gi[b])] for b,a in sorted(assigned.items())]
def geometry(pred,rec,c):
    folder=ROOT/c['parent_protocol']['data_root'];meta=loadmat(folder/rec['files']['meta'],variable_names=['intrinsic_matrix','factor_depth'])
    K=np.asarray(meta['intrinsic_matrix'],np.float64);factor=float(np.asarray(meta['factor_depth']).reshape(-1)[0])
    require(K.shape==(3,3) and np.isfinite(K).all() and factor>0 and np.isfinite(factor),'Invalid camera')
    require(np.allclose(K[[0,1,2,2],[1,0,0,1]],0,atol=1e-12,rtol=0) and K[2,2]==1 and K[0,0]>0 and K[1,1]>0,'Unsupported intrinsic matrix')
    with Image.open(folder/rec['files']['depth']) as im: depth=np.asarray(im,np.float64)/factor
    require(depth.shape==(720,1280),'Original image shape changed')
    full=np.repeat(np.repeat(pred,3,axis=0),4,axis=1);uv=[];zs=[];ids=[];missing=[]
    for inst in np.unique(pred[pred>0]):
        flat=np.flatnonzero((full==inst)&np.isfinite(depth)&(depth>0))
        if not len(flat):missing.append(int(inst));continue
        chosen=flat[np.linspace(0,len(flat)-1,min(c['points_per_instance'],len(flat)),dtype=np.int64)]
        y,x=np.unravel_index(chosen,depth.shape);uv.extend(np.column_stack([x,y]).tolist());zs.extend(depth[y,x].tolist());ids.extend([int(inst)]*len(chosen))
    uv=np.asarray(uv,np.float64).reshape(-1,2);z=np.asarray(zs,np.float64);ids=np.asarray(ids,np.int32)
    xyz=np.column_stack([(uv[:,0]-K[0,2])*z/K[0,0],(uv[:,1]-K[1,2])*z/K[1,1],z])
    reference=(np.linalg.inv(K)@np.column_stack([uv,np.ones(len(z))]).T).T*z[:,None]
    projected=(K@xyz.T).T
    back=projected[:,:2]/projected[:,2,None] if len(z) else np.empty((0,2))
    err=float(np.max(np.abs(back-uv))) if len(z) else 0.;referr=float(np.max(np.abs(reference-xyz))) if len(z) else 0.
    return dict(uv=uv,depth_m=z,instance_ids=ids,xyz_m=xyz,intrinsic_matrix=K),dict(point_count=len(z),instances_without_valid_depth=missing,projection_error_pixels=err,inverse_formula_error_metres=referr,passed=bool(len(z)>0 and np.isfinite(xyz).all() and err<=c['gates']['projection_tolerance_pixels'] and referr<=c['gates']['inverse_formula_tolerance_metres']))
def inspect(pred,fg,rec,c):
    gt=old.load_labels(ROOT,c['parent_protocol'],rec);m=old.frame_metrics(pred,fg,gt);matched=pairs(pred,gt)
    require(len(matched)==m['instances']['tp'],'Independent matching disagrees')
    arrays,g=geometry(pred,rec,c);with_depth=set(arrays['instance_ids'].tolist())
    checks=dict(nonempty_instances=m['predicted_instances']>0,matched_instance=m['instances']['tp']>0,matched_instance_has_depth=any(a in with_depth for a,b in matched),geometry=g['passed'])
    return dict(metrics=m,matched_pairs=matched,geometry=g,checks=checks),arrays,gt

def main():
    arg=argparse.ArgumentParser();arg.add_argument('--verify-only',action='store_true');args=arg.parse_args();c=read(CONFIG)
    out=ROOT/c['output'];start=time.monotonic();torch=None
    limit=c['resources']['verify_seconds' if args.verify_only else 'run_seconds']
    frozen=read(FREEZE)
    require(state(CONFIG)['sha256']==frozen['protocol_sha256'] and state(__file__)['sha256']==frozen['runner_sha256'],'Freeze binding changed')
    require(c['status']=='frozen' and c['max_new_forward_calls']==30 and len(c['records'])==30,'Protocol scope changed')
    def guard():
        require(time.monotonic()-start<limit,'Time budget exceeded')
        require(old.peak_rss()<c['resources']['rss_bytes'],'RSS budget exceeded')
        require(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<c['resources']['output_bytes'],'Output budget exceeded')
        if torch is not None: require(torch.cuda.max_memory_allocated()<c['resources']['gpu_bytes'] and torch.cuda.max_memory_reserved()<c['resources']['gpu_bytes'],'GPU budget exceeded')
    done=threading.Event()
    def watchdog():
        if not done.wait(max(0,limit-(time.monotonic()-start))):
            try: save(out/('verify_timeout.json' if args.verify_only else 'timeout.json'),dict(status='timeout',counts='inspect durable intents; unreturned calls remain unknown'))
            finally:os._exit(124)
    with project_lock(ROOT):
        before=snapshots(c);require(all(before[p]['sha256']==v['sha256'] and before[p]['bytes']==v['bytes'] for p,v in c['protected_files'].items()),'Protected identity differs')
        require(__import__('shutil').disk_usage(ROOT).free>=c['resources']['disk_reserve_bytes'],'Disk reserve not met')
        if args.verify_only:
            require((out/'result.json').exists() and not (out/'readonly.json').exists(),'Require single completed trial, no prior verification')
        else:
            require(not out.exists(),'Prior run evidence exists; user confirmation required before any retry')
            out.mkdir(parents=True);save(out/'claim.json',dict(pid=os.getpid(),protocol_sha256=state(CONFIG)['sha256'],records=30,automatic_retry=False))
        threading.Thread(target=watchdog,daemon=True).start()
        try:
            if args.verify_only:
                result=read(out/'result.json');require(result['completed_frames']==30 and len(result['frames'])==30,'Incomplete trial')
                require(result['protocol_sha256']==state(CONFIG)['sha256'],'Result protocol changed')
                require(result['new_model_forward_calls']==30 and result['unknown_forward_calls']==0,'Unexpected call accounting')
                require(len(list((out/'intents').glob('*.json')))==30 and len(list((out/'outcomes').glob('*.json')))==30,'Durable call accounting differs')
                output_before={str(p.relative_to(ROOT)):state(p) for p in out.rglob('*') if p.is_file()};rows=[]
                for row,selection in zip(result['frames'],c['records']):
                    rec=selection['record'];stem=selection['partition']+'_scene_%04d_%s'%(rec['scene'],rec['frame'])
                    require(row['identity']==stem and row['partition']==selection['partition'] and row['scene']==rec['scene'] and row['frame']==rec['frame'],'Frame identity differs')
                    require(read(out/'outcomes'/(stem+'.json'))==dict(identity=stem,forward_returned=True,new_forward_calls=1),'Forward outcome differs')
                    fn=out/'frames'/(stem+'.npz');require(state(fn)['sha256']==row['prediction_sha256'],'New prediction changed')
                    with np.load(fn,allow_pickle=False) as a:pred=a['instances'];fg=a['foreground']
                    with np.load(ROOT/selection['cached_prediction'],allow_pickle=False) as a:
                        identical=np.array_equal(pred,a['instances']) and np.array_equal(fg,a['foreground'])
                    require(bool(identical)==row['reproduces_frozen_prediction'],'Frozen prediction comparison differs')
                    detail,arrays,_=inspect(pred,fg,rec,c);require(detail==row['assessment'],'Saved assessment differs on full readonly recount')
                    geo=out/'frames'/(stem+'.geometry.npz');require(state(geo)['sha256']==row['geometry_sha256'],'Geometry cache changed')
                    with np.load(geo,allow_pickle=False) as a:require(set(a.files)==set(arrays) and all(np.array_equal(a[k],v) for k,v in arrays.items()),'Geometry cache differs')
                    guard();rows.append(stem)
                usable=all(x['reproduces_frozen_prediction'] and all(x['assessment']['checks'].values()) for x in result['frames'])
                require(result['usable_for_limited_baseline_reuse']==usable,'Usability decision differs')
                require(snapshots(c)==before,'Protected evidence changed during readonly')
                require(all(state(ROOT/p)==v for p,v in output_before.items()),'Trial files changed during readonly')
                save(out/'readonly.json',dict(status='passed',verified_frames=len(rows),all_trial_files_unchanged=True,protected_files_unchanged=True,new_model_forward_calls=0,new_training_updates=0,elapsed_seconds=time.monotonic()-start))
                print(json.dumps(dict(status='readonly_passed',usable=result['usable_for_limited_baseline_reuse'],frames=len(rows))),flush=True);return
            import torch as t;torch=t
            from src.segmentation import PilotInstanceNet,decode_instances
            require(torch.cuda.is_available(),'CUDA unavailable');require(torch.__version__==c['parent_protocol']['runtime']['torch'],'Torch version changed')
            torch.manual_seed(c['parent_protocol']['seed']);torch.cuda.manual_seed_all(c['parent_protocol']['seed']);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            payload=torch.load(ROOT/c['checkpoint'],map_location='cpu',weights_only=False)
            require(payload['cursor']==dict(epoch=10,global_step=1600,next_batch=0),'Checkpoint cursor differs')
            require(payload['config_fingerprint']==c['checkpoint_config_fingerprint'] and payload['data_fingerprint']==c['checkpoint_data_fingerprint'],'Checkpoint binding differs')
            model=PilotInstanceNet(mode='rgbd');model.load_state_dict(payload['model'],strict=True);del payload
            require(all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()),'Invalid model weights')
            model=model.cuda().eval();torch.cuda.reset_peak_memory_stats();rows=[];returned=0
            for selection in c['records']:
                rec=selection['record'];stem=selection['partition']+'_scene_%04d_%s'%(rec['scene'],rec['frame']);guard()
                inputs=old.prepare_input(ROOT,c['parent_protocol'],rec,'rgbd')
                save(out/'intents'/(stem+'.json'),dict(identity=stem,forward_call_intended=1))
                with torch.inference_mode():
                    heads=model(torch.from_numpy(inputs)[None].cuda());torch.cuda.synchronize()
                    returned+=1;save(out/'outcomes'/(stem+'.json'),dict(identity=stem,forward_returned=True,new_forward_calls=1))
                    require(all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in heads.values()),'Invalid heads')
                    pred=decode_instances(heads,**c['parent_protocol']['decoder']);fg=heads['foreground'][0,0].sigmoid().cpu().numpy()>=.5
                target=out/'frames';target.mkdir(exist_ok=True);fn=target/(stem+'.npz');np.savez_compressed(fn,instances=pred,foreground=fg)
                with np.load(ROOT/selection['cached_prediction'],allow_pickle=False) as a:
                    identical=np.array_equal(pred,a['instances']) and np.array_equal(fg,a['foreground'])
                assessment,arrays,gt=inspect(pred,fg,rec,c);geo=target/(stem+'.geometry.npz');np.savez_compressed(geo,**arrays)
                row=dict(identity=stem,partition=selection['partition'],scene=rec['scene'],frame=rec['frame'],reproduces_frozen_prediction=bool(identical),prediction_sha256=state(fn)['sha256'],geometry_sha256=state(geo)['sha256'],assessment=assessment)
                save(target/(stem+'.json'),row);rows.append(row);del heads,inputs
                print(json.dumps(dict(frame=len(rows),total=30,identity=stem,instances=assessment['metrics']['predicted_instances'],tp=assessment['metrics']['instances']['tp'],passed=identical and all(assessment['checks'].values()))),flush=True);guard()
            after=snapshots(c);require(before==after,'Original sources/weights/data/cache changed')
            summary={}
            for partition in ['train','development']:
                subset=[x for x in rows if x['partition']==partition];summary[partition]=dict(frames=len(subset),tp=sum(x['assessment']['metrics']['instances']['tp'] for x in subset),fp=sum(x['assessment']['metrics']['instances']['fp'] for x in subset),fn=sum(x['assessment']['metrics']['instances']['fn'] for x in subset),macro_f1=sum(x['assessment']['metrics']['instances']['f1_iou50']['value'] for x in subset)/len(subset))
            usable=all(x['reproduces_frozen_prediction'] and all(x['assessment']['checks'].values()) for x in rows)
            save(out/'result.json',dict(schema='20b',status='completed_single_supervised_trial',usable_for_limited_baseline_reuse=usable,full_P4_complete=False,method_advantage_established=False,completed_frames=len(rows),new_model_forward_calls=returned,unknown_forward_calls=0,new_training_updates=0,new_official_AP_calls=0,protected_files_unchanged=True,protocol_sha256=state(CONFIG)['sha256'],summary=summary,frames=rows,resources=dict(elapsed_seconds=time.monotonic()-start,peak_rss_bytes=old.peak_rss(),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved()),stop_after_this_trial=True,requires_user_confirmation_before_next_experiment=True))
            print(json.dumps(dict(status='trial_complete_stop',usable=usable,new_model_forward_calls=returned)),flush=True)
        except BaseException as ex:
            name='verification_failure.json' if args.verify_only else 'failure.json'
            if not (out/name).exists():
                intents=len(list((out/'intents').glob('*.json')));outcomes=len(list((out/'outcomes').glob('*.json')))
                save(out/name,dict(status='stopped_no_automatic_retry',error_type=type(ex).__name__,error=str(ex),intents=intents,returned_forward_calls=outcomes,unknown_forward_calls=max(0,intents-outcomes),requires_user_confirmation_before_retry=True))
            raise
        finally:done.set()
if __name__=='__main__':main()
