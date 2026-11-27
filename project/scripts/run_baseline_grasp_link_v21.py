"""One supervised cached R16-to-R14 interface experiment. No model or scorer calls."""
import os,sys,json,hashlib,time,threading,argparse,shutil,ctypes
from pathlib import Path
from datetime import datetime
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.pipeline import project_lock
import numpy as np
from PIL import Image
from scipy.io import loadmat
CONFIG='configs/baseline_grasp_link_v21.json'
LOG='logs/P4/baseline_grasp_link_v21'
OUT='results/P4/baseline_grasp_link_v21'
FRAMES=['0016','0080','0160','0240']
ARRAYS=['valid_points','valid_pixel_indices','sampled_valid_indices','sampled_points','sampled_pixel_indices']
def need(v,m):
    if not v:raise RuntimeError(m)
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def identity(p):
    p=ROOT/p;s=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return dict(sha256=h.hexdigest(),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def save(p,v):
    p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def text(p,s):
    p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:f.write(s)
def archive(p):
    with np.load(ROOT/p,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def rss():
    class C(ctypes.Structure):_fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in ['peak','working','qpp','qp','qnp','qn','page','peakpage']]
    c=C();c.cb=ctypes.sizeof(c);ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
    need(ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(ctypes.windll.kernel32.GetCurrentProcess()),ctypes.byref(c),c.cb),'Cannot measure RSS')
    return c.peak
def now():return datetime.now().astimezone().isoformat()
def frozen_snapshot(c):return {p:identity(p) for p in c['protected']}
def maps(mask,pixels,independent=False):
    if independent:return np.asarray([int(mask[int(p)//1280//3,int(p)%1280//4]) for p in pixels],np.int32)
    return np.repeat(np.repeat(mask,3,axis=0),4,axis=1).reshape(-1)[pixels].astype(np.int32)

def associate(selection,independent=False):
    directory=selection['generation'];a={k:np.load(ROOT/directory/(k+'.npy'),allow_pickle=False) for k in ARRAYS}
    b=archive(directory+'/forward_bundle.npz');prov=archive(directory+'/seed_provenance.npz')
    candidates=np.load(ROOT/directory/'candidates.npy',allow_pickle=False)
    mask=archive(selection['prediction'])['instances'];count=selection['candidate_count']
    need(mask.shape==(240,320) and mask.dtype.kind in 'iu' and np.min(mask)>=0,'Invalid instance image')
    need(candidates.shape==(count,17) and candidates.dtype==np.float32 and np.isfinite(candidates).all(),'Candidate shape/type changed')
    need(np.array_equal(candidates,b['candidates']),'Cached candidate bundle changed')
    need(a['sampled_points'].shape==(20000,3),'Sample budget changed')
    for key in ['sampled_valid_indices','sampled_pixel_indices','valid_pixel_indices']:
        need(a[key].dtype==np.int64 and a[key].ndim==1,'Invalid pixel/sample indices')
    need(np.all((a['sampled_valid_indices']>=0)&(a['sampled_valid_indices']<len(a['valid_points']))),'Sample index out of bounds')
    need(np.array_equal(a['sampled_points'],a['valid_points'][a['sampled_valid_indices']]),'Sample-to-valid point chain differs')
    need(np.array_equal(a['sampled_pixel_indices'],a['valid_pixel_indices'][a['sampled_valid_indices']]),'Sample-to-pixel chain differs')
    sa1,sa2,selected=(b[k] for k in ['sa1_indices','sa2_indices','selected_seed_indices'])
    need(sa1.dtype==sa2.dtype==selected.dtype==np.int64 and sa1.shape==(2048,) and sa2.shape==(1024,) and selected.shape==(count,),'SA index shapes changed')
    need(np.all((sa1>=0)&(sa1<20000)) and np.all((sa2>=0)&(sa2<2048)) and np.all((selected>=0)&(selected<1024)) and np.all(np.diff(selected)>0),'SA index bounds/order changed')
    seeds=sa1[sa2][selected]
    need(np.array_equal(b['fp2_xyz'],a['sampled_points'][sa1[sa2]]) and np.array_equal(candidates[:,13:16],b['fp2_xyz'][selected]),'Candidate seed composition differs')
    pixels=a['sampled_pixel_indices'][seeds];valid=a['sampled_valid_indices'][seeds]
    need(np.array_equal(prov['candidate_sampled_indices'],seeds) and np.array_equal(prov['candidate_valid_indices'],valid) and np.array_equal(prov['candidate_pixel_indices'],pixels),'Committed provenance differs')
    need(np.all((pixels>=0)&(pixels<720*1280)),'Pixel index out of bounds')
    labels=maps(mask,pixels,independent)
    meta=loadmat(ROOT/selection['meta'],variable_names=['intrinsic_matrix','factor_depth'])
    K=np.asarray(meta['intrinsic_matrix'],np.float64);factor=float(np.asarray(meta['factor_depth']).item())
    with Image.open(ROOT/selection['depth']) as im:depth=np.asarray(im)
    need(depth.shape==(720,1280) and K.shape==(3,3) and np.isfinite(K).all() and factor>0,'Camera/depth invalid')
    y=pixels//1280;x=pixels%1280;z=depth.reshape(-1)[pixels].astype(np.float64)/factor
    if independent:
        flat=np.arange(depth.size,dtype=np.int64);zz=depth.reshape(-1).astype(np.float64)/factor
        cloud=np.column_stack([(flat%1280-K[0,2])*zz/K[0,0],(flat//1280-K[1,2])*zz/K[1,1],zz])
        good=np.isfinite(cloud).all(axis=1)&(zz>0)&(zz<=2.)
        need(np.array_equal(a['valid_pixel_indices'],flat[good]) and np.array_equal(a['valid_points'],cloud[good].astype(np.float32)),'Independent full raw cloud differs')
        xyz=np.asarray([(np.linalg.inv(K)@np.array([float(xx),float(yy),1.]))*float(zz) for xx,yy,zz in zip(x,y,z)])
    else:xyz=np.column_stack([(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z])
    need(np.all(z>0) and np.all(z<=2.) and np.isfinite(xyz).all(),'Candidate seed depth invalid')
    expected=np.column_stack([(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z]).astype(np.float32)
    need(np.array_equal(expected,candidates[:,13:16]),'Candidate coordinates differ from original pixel depth')
    need(np.max(np.abs(xyz-candidates[:,13:16]))<=2e-7,'Float32 coordinate difference exceeds contract')
    projected=(K@candidates[:,13:16].astype(np.float64).T).T
    error=np.max(np.abs(projected[:,:2]/projected[:,2,None]-np.column_stack([x,y])),axis=1)
    need(float(error.max())<=.001,'Seed reprojection differs')
    rotation=candidates[:,4:13].reshape(-1,3,3)
    need(np.max(np.abs(rotation.transpose(0,2,1)@rotation-np.eye(3)))<1e-4 and np.max(np.abs(np.linalg.det(rotation)-1))<1e-4,'Invalid rotations')
    need(np.all(candidates[:,16]==-1),'Original unknown object-class field changed')
    unique,counts=np.unique(labels,return_counts=True)
    arrays=dict(original_candidate_indices=np.arange(count,dtype=np.int64),seed_instance_ids=labels,seed_pixel_indices=pixels,seed_sampled_indices=seeds,seed_valid_indices=valid)
    report=dict(frame=selection['frame'],candidates=count,associated=int(np.count_nonzero(labels)),background=int(np.count_nonzero(labels==0)),unmatched=0,ambiguous=0,per_seed_instance_count={str(int(k)):int(v) for k,v in zip(unique,counts)},max_reprojection_error_pixels=float(error.max()),all_candidates_retained=True,all_original_pose_score_order_preserved=True,seed_label_is_not_object_class_or_gripper_ownership=True)
    return arrays,report,candidates

def diagnostics(c):
    rows=[];totals=dict(known=0,unknown=0,known_mu04_pass=0,known_mu04_pass_associated=0,known_mu04_pass_background=0)
    for sel in c['selections']:
        frame=sel['frame'];ann=archive(OUT+'/'+frame+'/association.npz');labels=ann['seed_instance_ids']
        official=read(sel['geometry_report'])['ordered_records'];seen=set();npass=0;nback=0
        for r in official:
            idx=r['original_index'];need(isinstance(idx,int) and 0<=idx<len(labels) and idx not in seen,'Official support identity invalid');seen.add(idx)
            score=float(r['score']);need(np.isfinite(score),'Nonfinite historical score')
            positive=not r['collision'] and 0<score<=.4;npass+=int(positive);nback+=int(positive and labels[idx]==0)
            rows.append(dict(frame=frame,original_candidate_index=idx,seed_instance_id=int(labels[idx]),historical_official_score=score,historical_collision=bool(r['collision']),historical_mu04_pass=bool(positive)))
        need(len(seen)==50,'Frozen support size changed')
        totals['known']+=len(seen);totals['unknown']+=len(labels)-len(seen);totals['known_mu04_pass']+=npass;totals['known_mu04_pass_background']+=nback;totals['known_mu04_pass_associated']+=npass-nback
    return dict(scope='Existing geometry-selected support only; descriptive linkage, not a new score/AP, probability, gain test, or held-out result. Unknown labels stay unknown. Background candidates are retained.',totals=totals,rows=rows)

def prepare():
    need(not (ROOT/CONFIG).exists() and not (ROOT/OUT).exists(),'Do not overwrite a protocol or previous run')
    parent=read('configs/grasp_multiview_v14r1.json');accept=read('results/P3/grasp_multiview_acceptance_v14r1.json');seg=read('configs/segmentation_baseline_v16.json')
    previous=read('results/P4/baseline_usability_v20b/closeout.json');need(previous['usable_for_limited_baseline_reuse'],'Previous usability gate not passed')
    protected={}
    def protect(p,expected=None):
        q=identity(p)
        if expected is not None:need(q['sha256']==expected,'Prior evidence hash differs: '+p)
        protected[p]=dict(sha256=q['sha256'],bytes=q['bytes'])
    for p in ['configs/grasp_multiview_v14r1.json','results/P3/grasp_multiview_acceptance_v14r1.json','configs/segmentation_baseline_v16.json','results/P4/segmentation_baseline_v16_acceptance.json','results/P4/baseline_usability_v20b/closeout.json','results/P4/baseline_usability_v20b/readonly.json','runtime/pipeline/state.json','configs/stages.json','scripts/pipeline.py','scripts/run_baseline_grasp_link_v21.py']:
        protect(p)
    need(identity('configs/grasp_multiview_v14r1.json')['sha256']==accept['protocol_sha256'],'R14 protocol differs')
    selections=[]
    for frame in FRAMES:
        directory=f'results/P3/grasp_multiview_v14r1/frames/{frame}/generation';g=f'results/P3/grasp_multiview_v14r1/frames/{frame}/geometry'
        protect(directory+'/completion.json',accept['frames'][frame]['generation_completion_sha256'])
        completion=read(directory+'/completion.json')
        for p,h in completion['outputs'].items():protect(directory+'/'+p,h)
        protect(g+'/completion.json',accept['frames'][frame]['geometry_completion_sha256'])
        geo=read(g+'/completion.json');need('report.json' in geo['outputs'],'Missing historical geometry report seal');protect(g+'/report.json',geo['outputs']['report.json'])
        prediction=f'results/P4/segmentation_baseline_v16_evaluation/rgbd/frames/train/scene_0003_{frame}/prediction.npz'
        record=read(str(Path(prediction).parent/'record.json'));receipt=read(prediction+'.sha256.json')
        need(record['identity']==dict(camera='realsense',frame=frame,partition='train',scene=3) and record['arm']=='rgbd' and record['checkpoint_sha256']=='1017aafd7f5f6a0663f2572e56a7373667db389fc4d57bf84bbb49742281e3c7','R16 frame/weight lineage differs')
        need(record['prediction_sha256']==receipt['sha256'],'R16 prediction receipts differ');protect(prediction,receipt['sha256']);protect(prediction+'.sha256.json');protect(str(Path(prediction).parent/'record.json'))
        for kind in ['rgb','depth','meta']:
            p=parent['frame_inputs'][frame][kind];need(parent['generation_inputs'][frame][p]==record['source_sha256'][kind],'Segmentation/grasp source image differs');protect(p,record['source_sha256'][kind])
        selections.append(dict(frame=frame,generation=directory,prediction=prediction,depth=parent['frame_inputs'][frame]['depth'],meta=parent['frame_inputs'][frame]['meta'],geometry_report=g+'/report.json',candidate_count=accept['frames'][frame]['candidate_count']))
    need(sum(s['candidate_count'] for s in selections)==719,'Frozen four-view candidate count changed')
    mask=np.zeros((240,320),np.int32);mask[1:3,1:3]=7;pix=np.array([0,3*1280+4,8*1280+11,9*1280+12],np.int64)
    need(maps(mask,pix).tolist()==[0,7,7,0] and np.array_equal(maps(mask,pix),maps(mask,pix,True)),'Synthetic boundary/background/ID checks fail')
    current=read('logs/progress_audit/supervised_v1_20261027/after_trial.json');need(current['earned_points_current']==25,'Progress baseline changed')
    # Refine one existing unfinished 2-point item before outcomes; total denominator is unchanged.
    for p in current['packages']:
        if p['id']=='P4':
            found=[s for s in p['subdeliverables'] if s['name']=='正式抓取链路接入验证'];need(len(found)==1 and found[0]['weight']==2 and found[0]['status']=='pending','Existing P4 remaining item differs')
            p['subdeliverables'].remove(found[0]);p['subdeliverables'].extend([dict(name='缓存候选的实例与原图像素追踪接口',weight=1,status='pending_v21'),dict(name='代表性范围的正式抓取链路验收',weight=1,status='pending')])
    current['v21_refinement']=dict(at=now(),scope='Subdivide existing pending P4 2 points into cached interface 1 and representative formal linkage 1. No extra scope/weight and no scientific acceptance implied.',credit_rule='Only after all four views/719 rows and independent zero-forward readonly acceptance, plus truthful use/reject report. Full P4 remains open.')
    save(LOG+'/progress_before_trial.json',current);protect(LOG+'/progress_before_trial.json')
    protocol=dict(schema=21,status='frozen',created_at=now(),scope='One supervised R16 cached-instance to R14 actual-seed/candidate interface experiment',selections=selections,protected=protected,
        resource=dict(run_seconds=120,verify_seconds=120,total_cap_seconds=240,cpu_threads=2,rss_bytes=1024**3,output_bytes=32*1024**2,disk_reserve_bytes=60*1024**3),
        gates=['All 719 candidates retain exact 17-column values and original order','Actual SA1/SA2 seed composition and original depth pixel source match exactly','Independent mask mapping agrees; 0 unmatched/ambiguous; backgrounds retained','Original-pixel metric coordinate and <=0.001 pixel reprojection verified','At least one associated candidate in every frame; no minimum foreground fraction','Full read-only independent recount passes and prior files unchanged'],
        decision='If passed, accept limited cached interface for development only. Never adopt a foreground-only hard filter or claim gain from this test.',
        labels='Frame-local predicted seed instance IDs are sidecar fields; original object class stays -1. Seed location does not prove full-gripper ownership.',
        diagnostics='After all label-free association outputs are sealed, join the existing 200 official labels; remaining 519 unknown, never negatives. No new scoring.',
        work_budget=dict(new_model_forward_calls=0,new_training_updates=0,new_official_scoring_calls=0,new_random_sampling_calls=0,frames=4,candidates=719),
        stop='One run plus one verification. Stop on completion or failure; no retries or next round without user confirmation. Automation stays deleted.',
        limitations=['Four related views of reused training scene 3, not independent/generalization evidence','No new candidates or scientific benefit comparison','Full P4/P6/P8 not completed by this trial'])
    save(CONFIG,protocol)
    text('docs/第二十一轮基线抓取接口单轮方案.txt','\n'.join(['第二十一轮：R16基线接入已有抓取链路，用户监督单轮',now(),'固定场景3四个既有训练视角0016/0080/0160/0240，共719候选。复用R16冻结掩膜及R14r1冻结候选/真实SA采样链，不重新训练、推理、采样或官方评分。','验收全部719项：候选姿态、分数和顺序逐值不变；从SA1/SA2/选中种子追到原图深度像素；整数3x4实例映射与独立算法相同；坐标源相同且重投影<=0.001像素；每帧至少一个候选具有预测实例归属。背景候选全部保留，不能作为丢弃依据。','仅侧表记录seed_instance_id，不覆盖原object_id=-1，不声称种子实例就是夹爪接触物体或完整夹爪归属。先封存无真值标签的部署关联结果，再连接已有200个官方支持项，其余519未知。','独立只读检查从原始深度重构完整点云，并逐候选重做种子与像素映射；不重算官方分数、不调用模型。','资源：运行120秒、只读120秒，累计240秒；CPU2线程、RSS抽样上限1GiB、输出32MiB、磁盘余量60GiB；超时强停。一次运行一次复验，失败或完成均停止，不自动修订重跑。','只判断有限接口可用性，不能作为方法提升、独立测试、正式P4或P8通过。','总进度分母保持100/实验80。将原P4剩余2点正式接入项在结果出现前细分为缓存接口1点、代表性正式验收1点；本轮完整试验及独立验收后才可记前1点，全部25->26%，实验31.25->32.5%。不是按719条数计总进度。','原协议、源码、数据、阈值、划分、预算、状态和旧结果全部保护不修改。','']))
    save(LOG+'/freeze.json',dict(created_at=now(),protocol_sha256=identity(CONFIG)['sha256'],runner_sha256=identity('scripts/run_baseline_grasp_link_v21.py')['sha256'],preflight='Passed synthetic mask-boundary/background/ID mapping and parent lineage checks; not formal trial passed',protected_files=len(protected)))
    handoff=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(handoff,ROOT/LOG/'handoff_before.txt')
    handoff.write_text('最新用户授权：'+now()+' 仅继续一轮R21有限实例到抓取候选接口验证，4视角719候选，0新模型前向/训练/评分，完成或失败后停止等待用户确认。自动任务仍删除。全部任务总进度25%，实验总进度31.25%（工作包估算），当前0/4视角。详见docs/第二十一轮基线抓取接口单轮方案.txt。\n\n以下是历史状态：\n'+handoff.read_text(encoding='utf-8-sig'),encoding='utf-8')
    print(json.dumps(dict(status='frozen',frames=4,candidates=719,protected=len(protected))))

def execute(verify=False):
    c=read(CONFIG);freeze=read(LOG+'/freeze.json');need(identity(CONFIG)['sha256']==freeze['protocol_sha256'] and identity('scripts/run_baseline_grasp_link_v21.py')['sha256']==freeze['runner_sha256'],'Frozen source/config changed')
    start=time.monotonic();limit=c['resource']['verify_seconds' if verify else 'run_seconds'];done=threading.Event()
    def guard():
        need(time.monotonic()-start<limit,'Time cap exceeded');need(rss()<c['resource']['rss_bytes'],'RSS cap exceeded')
        need(sum(p.stat().st_size for p in (ROOT/OUT).rglob('*') if p.is_file())<c['resource']['output_bytes'],'Output cap exceeded')
    def watchdog():
        if not done.wait(max(0,limit-(time.monotonic()-start))):
            try:save(OUT+('/verify_timeout.json' if verify else '/timeout.json'),dict(status='timeout_stopped',new_model_forward_calls=0,no_automatic_retry=True))
            finally:os._exit(124)
    before=frozen_snapshot(c);need(all(v['sha256']==c['protected'][p]['sha256'] and v['bytes']==c['protected'][p]['bytes'] for p,v in before.items()),'Parent identities changed')
    need(shutil.disk_usage(ROOT).free>=c['resource']['disk_reserve_bytes'],'Disk reserve not met')
    if verify:need((ROOT/OUT/'result.json').exists() and not (ROOT/OUT/'readonly.json').exists(),'Exactly one verification required')
    else:
        need(not (ROOT/OUT).exists(),'Prior output exists; no automatic retry');save(OUT+'/claim.json',dict(pid=os.getpid(),at=now(),work_budget=c['work_budget'],protocol_sha256=freeze['protocol_sha256'],automatic_retry=False))
    threading.Thread(target=watchdog,daemon=True).start()
    try:
        guard();reports=[]
        oldfiles={str(p.relative_to(ROOT)):identity(str(p.relative_to(ROOT))) for p in (ROOT/OUT).rglob('*') if p.is_file()} if verify else None
        for sel in c['selections']:
            arrays,report,candidates=associate(sel,independent=verify);directory=OUT+'/'+sel['frame']
            if verify:
                saved=archive(directory+'/association.npz');need(set(saved)==set(arrays) and all(np.array_equal(saved[k],a) and saved[k].dtype==a.dtype for k,a in arrays.items()),'Independent association differs')
                need(read(directory+'/report.json')==report,'Independent frame report differs')
                need(identity(directory+'/candidates.npy')['sha256']==identity(sel['generation']+'/candidates.npy')['sha256'],'Candidate values/order changed')
            else:
                (ROOT/directory).mkdir(parents=True);np.savez_compressed(ROOT/directory/'association.npz',**arrays)
                shutil.copyfile(ROOT/sel['generation']/'candidates.npy',ROOT/directory/'candidates.npy');save(directory+'/report.json',report)
            reports.append(report);guard();print(json.dumps(dict(mode='verify' if verify else 'run',frame=sel['frame'],candidates=report['candidates'],associated=report['associated'],background=report['background'])),flush=True)
        usable=all(r['associated']>0 for r in reports)
        if not verify:
            seals={str(p.relative_to(ROOT)):identity(str(p.relative_to(ROOT))) for p in (ROOT/OUT).rglob('*') if p.is_file()}
            save(OUT+'/deployment_seal.json',dict(at=now(),status='sealed_before_reading_official_labels',files=seals))
            diag=diagnostics(c);save(OUT+'/historical_support_diagnostic.json',diag)
            need(all(identity(p)==v for p,v in seals.items()),'Deployment files changed during diagnostics')
            need(frozen_snapshot(c)==before,'Protected originals changed')
            save(OUT+'/result.json',dict(schema=21,status='completed_stopped_awaiting_verification',usable_limited_interface=usable,frames=reports,completed_frames=4,candidate_count=719,associated=sum(r['associated'] for r in reports),background=sum(r['background'] for r in reports),retained_all_candidates=True,original_values_order_unchanged=True,new_model_forward_calls=0,new_training_updates=0,new_official_scoring_calls=0,new_random_sampling_calls=0,full_P4_complete=False,method_advantage_established=False,diagnostic_summary=diag['totals'],protected_files_unchanged=True,resources=dict(seconds=time.monotonic()-start,peak_rss_bytes=rss()),requires_user_confirmation_before_next=True))
        else:
            result=read(OUT+'/result.json');need(result['frames']==reports and result['completed_frames']==4 and result['candidate_count']==719 and result['usable_limited_interface']==usable,'Final result differs')
            seal=read(OUT+'/deployment_seal.json');need(all(identity(p)==v for p,v in seal['files'].items()),'Deployment seal differs')
            need(read(OUT+'/historical_support_diagnostic.json')==diagnostics(c),'Historical support/unknown linkage differs')
            need(frozen_snapshot(c)==before and all(identity(p)==v for p,v in oldfiles.items()),'Readonly changed source/result evidence')
            save(OUT+'/readonly.json',dict(status='passed',verified_frames=4,verified_candidates=719,independent_raw_cloud_and_pixel_mapping=True,all_sources_and_result_files_unchanged=True,new_model_forward_calls=0,new_official_scoring_calls=0,resources=dict(seconds=time.monotonic()-start,peak_rss_bytes=rss())))
        print(json.dumps(dict(status='readonly_passed_stop' if verify else 'trial_complete',usable=usable)),flush=True)
    except BaseException as e:
        p=OUT+('/verify_failure.json' if verify else '/failure.json')
        if not (ROOT/p).exists():save(p,dict(status='stopped_no_retry',error_type=type(e).__name__,error=str(e),new_model_forward_calls=0,new_training_updates=0,new_official_scoring_calls=0,requires_user_confirmation=True))
        raise
    finally:done.set()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','verify']);args=parser.parse_args()
    with project_lock(ROOT):
        if args.mode=='prepare':prepare()
        else:execute(args.mode=='verify')
if __name__=='__main__':main()
