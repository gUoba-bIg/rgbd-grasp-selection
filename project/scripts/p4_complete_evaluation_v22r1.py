"""Bounded CPU evaluation and independent cached-output audit for whole P4."""
import os,sys,json,hashlib,argparse,time,uuid,shutil
from pathlib import Path
from datetime import datetime
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
from scripts import evaluate_mask_ap_v17 as official
from scripts import accept_mask_ap_v17 as reference
from scripts.extract_mask_scores_v17 import decode_with_scores
from scripts.pipeline import project_lock
PARENT=ROOT/'configs/p4_full_baseline_v22r1.json'
CONFIG=ROOT/'configs/p4_complete_evaluation_v22r1.json'
BASE=ROOT/'results/P4/full_baseline_v22r1'
OUT=BASE/'evaluation'
LOG=ROOT/'logs/P4/full_baseline_v22r1/evaluation'
def need(v,m):
    if not v:raise RuntimeError(m)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stat(p):
    p=Path(p);s=p.stat();return dict(sha256=sha(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def now():return datetime.now().astimezone().isoformat()
def once(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():need(read(p)==x,'Immutable record mismatch: '+str(p));return
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def high_resolution(low):
    need(low.shape==(480,640),'Unexpected mask grid')
    return np.repeat(low,2,axis=1)[(2*np.arange(720)+1)//3].astype(np.int32)
def indexed_resize(low,height,width):
    # Pixel-center nearest sampling; odd/even geometry is tested before protocol freeze.
    yy=np.floor((np.arange(height)+.5)*low.shape[0]/height).astype(np.int64)
    xx=np.floor((np.arange(width)+.5)*low.shape[1]/width).astype(np.int64)
    return low[yy[:,None],xx[None,:]]
def boundaries(labels):
    padded=np.pad(labels,1,constant_values=0)
    return (labels>0)&((labels!=padded[:-2,1:-1])|(labels!=padded[2:,1:-1])|(labels!=padded[1:-1,:-2])|(labels!=padded[1:-1,2:]))
def boundary_counts(pred,gt):
    p=boundaries(pred);g=boundaries(gt);npix=int(p.sum());ng=int(g.sum())
    pm=int(np.count_nonzero(p&(distance_transform_edt(~g)<=2))) if ng else 0
    gm=int(np.count_nonzero(g&(distance_transform_edt(~p)<=2))) if npix else 0
    return dict(predicted=npix,reference=ng,matched_prediction=pm,matched_reference=gm)
def boundary_f1(rows):
    n={k:sum(r[k] for r in rows) for k in rows[0]};p=n['matched_prediction']/n['predicted'] if n['predicted'] else 0.;r=n['matched_reference']/n['reference'] if n['reference'] else 0.
    return dict(counts=n,precision=p,recall=r,f1=2*p*r/(p+r) if p+r else 0.)
def maximum_matching(overlap):
    # Independent augmenting-path maximum-cardinality matching (not Hungarian).
    assigned={}
    def visit(p,seen):
        for g in np.flatnonzero(overlap[p]>=.5):
            g=int(g)
            if g in seen:continue
            seen.add(g)
            if g not in assigned or visit(assigned[g],seen):assigned[g]=p;return True
        return False
    return sum(visit(p,set()) for p in range(len(overlap)))
def frame_check(pred,fg,gt,old):
    pids=[int(x) for x in np.unique(pred) if x>0];gids=[int(x) for x in np.unique(gt) if x>0]
    overlap=reference.dense_overlap(pred,gt,pids,gids)['iou'];tp=maximum_matching(overlap)
    expected=dict(tp=tp,fp=len(pids)-tp,fn=len(gids)-tp)
    need(all(old['instances'][k]==v for k,v in expected.items()),'Independent IoU50 matching differs')
    need(old['foreground_intersection']==int(np.count_nonzero(fg&(gt>0))) and old['foreground_union']==int(np.count_nonzero(fg|(gt>0))),'Foreground counts differ')
    need(old['predicted_instances']==len(pids) and old['actual_instances']==len(gids),'Instance counts differ')
    return expected
def prepare():
    need(not CONFIG.exists(),'Postprocessing protocol already exists')
    grid=np.arange(480*640,dtype=np.int32).reshape(480,640)
    need(np.array_equal(high_resolution(grid),indexed_resize(grid,720,1280)),'Nearest upsample geometry differs')
    raw=np.arange(720*1280,dtype=np.int32).reshape(720,1280)
    need(np.array_equal(np.asarray(Image.fromarray(raw).resize((640,480),Image.Resampling.NEAREST)),indexed_resize(raw,480,640)),'Nearest label downsample differs')
    need(maximum_matching(np.array([[.8,.6],[.7,0.]]))==2,'Independent assignment fixture failed')
    tiny=np.zeros((9,9),np.int32);tiny[2:7,2:7]=1
    need(boundary_f1([boundary_counts(tiny,tiny)])['f1']==1,'Boundary identity fixture failed')
    source=['scripts/p4_complete_evaluation_v22r1.py','scripts/evaluate_mask_ap_v17.py','scripts/accept_mask_ap_v17.py','scripts/extract_mask_scores_v17.py','scripts/evaluate_segmentation_baseline_v16.py','scripts/mask_ap_common_v17.py','scripts/pipeline.py','configs/p4_full_baseline_v22r1.json']
    c=dict(schema=22,created_at=now(),purpose='Complete paired 160-frame development evaluation and independent audit; no checkpoint selection, training, model inference, or held-out access',source_hashes={p:sha(ROOT/p) for p in source},
        mask_ap=dict(official='unmodified pycocotools via verified v17 evaluator',parameters=official.coco_parameters(),original_size=[720,1280],resize='exact rational nearest pixel centers: original row y selects (2*y+1)//3, column x selects x//2; ties to upper pixel; avoids Pillow floating accumulator tie drift',ground_truth='all original positive visible label IDs; one class; no area filtering; iscrowd=0',confidence='float32 sigmoid of retained selected center, no calibration',primary='macro of ten scene AP; pooled AP separately',blocks=22,independent_tolerance=1e-12),
        boundary=dict(grid=[480,640],distance_pixels=2,definition='positive pixels differing from any four-neighbor label; includes inter-instance and outer contours; Euclidean distance <=2',aggregation='pool precision/recall counts within scene; macro ten scenes',occlusion='true occlusion ratio unavailable; no fabricated occlusion metric'),
        resources=dict(max_official_calls=24,wall_seconds=3600,rss_bytes=6*1024**3,output_bytes=2*1024**3),
        final_readonly='rebuild every cached decoder using independent NumPy implementation, maximum matching, dense-overlap COCO ranking and all primary precision/recall/score arrays; verify immutable files and both full frame/scene summaries',
        figure_rule='fixed frame 0000 of each scene plus six lowest RGBD frame-F1 rows, ties scene/frame ascending; descriptive development error visualization, no tuning',new_model_forward_calls=0,new_training_updates=0)
    import importlib.metadata,platform
    environment=ROOT/'runtime/envs/p4_eval_v22'
    need(Path(sys.executable).resolve()==(environment/'Scripts/python.exe').resolve(),'Wrong evaluator environment')
    need('torch' not in sys.modules,'CPU evaluator imported torch')
    c['environment']=dict(python=platform.python_version(),packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},files={p.relative_to(ROOT).as_posix():sha(p) for p in environment.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    once(CONFIG,c);once(LOG/'freeze.json',dict(protocol_sha256=sha(CONFIG),synthetic_geometry_and_matching_checks_passed=True))
    print('P4 full CPU evaluation frozen; no research results read.',flush=True)
def inputs_snapshot(c):
    parent=read(PARENT);need((BASE/'training_and_export.json').exists(),'Training/export incomplete')
    paths={ROOT/p for p in parent['protected']}
    paths.update(ROOT/p for p in c['source_hashes']);paths.add(CONFIG)
    paths.update(p for p in BASE.rglob('*') if p.is_file() and OUT not in p.parents and p.name!='progress.json')
    selection=read(BASE/'training_and_export.json')['selected_epoch']
    for a in ['rgb','rgbd']:paths.add(ROOT/f'experiments/p4_full_baseline_v22r1/selected/epoch_{selection:02d}_{a}.pt')
    for a in ['rgb','rgbd']:paths.update((ROOT/f'experiments/p4_full_baseline_v22r1/fit/{a}').glob('*.json'))
    return {p.relative_to(ROOT).as_posix():stat(p) for p in sorted(paths)}
def bind(c):
    need(sha(CONFIG)==read(LOG/'freeze.json')['protocol_sha256'],'Postprocessing protocol modified')
    need(all(sha(ROOT/p)==h for p,h in c['source_hashes'].items()),'Postprocessing source modified')
    need(all(sha(ROOT/p)==h for p,h in c['environment']['files'].items()),'Evaluator environment modified')
    need('torch' not in sys.modules,'Unexpected torch import')
    current=inputs_snapshot(c)
    binding=LOG/'input_binding.json'
    if binding.exists():need(read(binding)==current,'Training/input bindings changed')
    else:once(binding,current)
    return current
def scan(arm,independent=False,make_inputs=False):
    parent=read(PARENT);images=[];annotations=[];detections=[];matching=[];rows=[]
    for rec in sorted(parent['development_records'],key=lambda r:(r['scene'],r['frame'])):
        stem=f"scene_{rec['scene']:04d}_{rec['frame']}";cache=BASE/'exports'/arm/(stem+'.npz');old=read(cache.with_suffix('.json'))
        need(sha(cache)==old['cache_sha256'],'Export digest changed')
        with np.load(cache,allow_pickle=False) as v:
            heads={k:v[k] for k in ['foreground_probability','center_probability','offsets']}
            if independent:
                rebuilt=reference.independent_decode(**heads,decoder=parent['decoder']);pred=rebuilt['instances'];fg=rebuilt['foreground'];scores=rebuilt['retained_instances']
            else:pred,fg,scores,_=decode_with_scores(heads,**parent['decoder'])
            need(pred.shape==(480,640) and np.array_equal(pred,v['instances']) and np.array_equal(fg,v['foreground']) and scores==old['scores'],'Decoder/confidence provenance differs')
        label=ROOT/parent['data_root']/rec['files']['label'];need(sha(label)==rec['sha256']['label'],'Ground truth changed')
        with Image.open(label) as im:raw=np.asarray(im);gt=np.asarray(im.resize((640,480),Image.Resampling.NEAREST),np.int64)
        need(raw.shape==(720,1280),'Unexpected original mask dimensions')
        if independent:need(np.array_equal(gt,indexed_resize(raw,480,640)),'Reference label coordinates differ')
        counts=frame_check(pred,fg,gt,old['metrics']);high=indexed_resize(pred,720,1280) if independent else high_resolution(pred)
        pids=[int(x) for x in np.unique(high) if x>0];gids=[int(x) for x in np.unique(raw) if x>0];image_id=rec['scene']*1000+int(rec['frame'])
        images.append(dict(id=image_id,height=720,width=1280));gtids=[];dtids=[]
        for gid in gids:
            annid=len(annotations)+1;gtids.append(annid);mask=raw==gid;yy,xx=np.nonzero(mask)
            row=dict(id=annid,image_id=image_id,category_id=1,area=int(mask.sum()),iscrowd=0)
            if make_inputs:row.update(segmentation=official.encode_mask(mask),bbox=[int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)])
            annotations.append(row)
        for score in scores:
            need(score['instance_id'] in pids,'Scored mask has no original-resolution support')
            row=dict(image_id=image_id,category_id=1,score=score['score'])
            if make_inputs:row['segmentation']=official.encode_mask(high==score['instance_id'])
            detections.append(row);dtids.append(len(detections))
        overlap=reference.dense_overlap(high,raw,pids,gids)['iou']
        matching.append(reference.independent_image_matching(image_id,gtids,dtids,[s['score'] for s in scores],overlap))
        rows.append(dict(scene=rec['scene'],frame=rec['frame'],counts=counts,foreground_intersection=old['metrics']['foreground_intersection'],foreground_union=old['metrics']['foreground_union'],boundary=boundary_counts(pred,gt),f1=2*counts['tp']/(2*counts['tp']+counts['fp']+counts['fn']) if 2*counts['tp']+counts['fp']+counts['fn'] else 0.))
    need(len(rows)==160 and len({r['scene'] for r in rows})==10,'Development coverage differs')
    summary=read(BASE/(arm+'_full_development.json'));f1=[];iou=[];bound=[]
    for oldscene in summary['scenes']:
        subset=[r for r in rows if r['scene']==oldscene['scene']];n={k:sum(r['counts'][k] for r in subset) for k in ['tp','fp','fn']};v=2*n['tp']/(2*n['tp']+n['fp']+n['fn']) if 2*n['tp']+n['fp']+n['fn'] else 0.
        u=sum(r['foreground_union'] for r in subset);iv=sum(r['foreground_intersection'] for r in subset)/u if u else 1.
        need(len(subset)==16 and v==oldscene['f1'] and iv==oldscene['foreground_iou'],'Scene F1/IoU summary differs')
        f1.append(v);iou.append(iv);bound.append(dict(scene=oldscene['scene'],**boundary_f1([r['boundary'] for r in subset])))
    need(float(np.mean(f1))==summary['scene_macro_f1'] and float(np.mean(iou))==summary['scene_macro_foreground_iou'],'Full development macro differs')
    return dict(gt=dict(info={},images=images,categories=[dict(id=1,name='object')],annotations=annotations),detections=detections),matching,dict(rows=rows,boundary_scenes=bound,boundary_scene_macro_f1=float(np.mean([b['f1'] for b in bound])))
def primary(value):
    p=np.asarray(value['precision'])[:,:,0,0,2]
    return dict(AP=float(p[p>=0].mean()),AP50=float(p[0][p[0]>=0].mean()),AP75=float(p[5][p[5]>=0].mean()))
def audit_ap(value,matching):
    ref=reference.independent_ap(matching)
    for key,expected in [('precision',np.asarray(value['precision'])[:,:,0,0,2]),('recall',np.asarray(value['recall'])[:,0,0,2]),('scores',np.asarray(value['scores'])[:,:,0,0,2])]:
        reference.close_arrays(expected,ref[key],'Independent '+key+' differs')
    return ref['summary']
def scoped(inputs,matching,scene):
    ids={v['id'] for v in inputs['gt']['images'] if scene is None or v['id']//1000==scene}
    gt=dict(inputs['gt'],images=[v for v in inputs['gt']['images'] if v['id'] in ids],annotations=[v for v in inputs['gt']['annotations'] if v['image_id'] in ids])
    return dict(gt=gt,detections=[v for v in inputs['detections'] if v['image_id'] in ids]),[v for v in matching if v and v['image_id'] in ids]
def event(kind,**x):
    LOG.mkdir(parents=True,exist_ok=True)
    with (LOG/'official_calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(kind=kind,at=now(),**x))+'\n');f.flush();os.fsync(f.fileno())
def colorize(labels):
    palette=np.array([[0,0,0]]+[[int((i*71)%200+55),int((i*131)%200+55),int((i*47)%200+55)] for i in range(1,89)],np.uint8)
    return palette[labels]
def figures(diagnostics):
    parent=read(PARENT);records={(r['scene'],r['frame']):r for r in parent['development_records']}
    selected=sorted({(r['scene'],r['frame']) for r in diagnostics['rgbd']['rows'] if r['frame']=='0000'}|{(r['scene'],r['frame']) for r in sorted(diagnostics['rgbd']['rows'],key=lambda r:(r['f1'],r['scene'],r['frame']))[:6]})
    inventory=[]
    for scene,frame in selected:
        rec=records[(scene,frame)];panels=[]
        with Image.open(ROOT/parent['data_root']/rec['files']['rgb']) as im:rgb=np.array(im.convert('RGB').resize((640,480)))
        panels.append(rgb)
        with Image.open(ROOT/parent['data_root']/rec['files']['label']) as im:gt=np.array(im.resize((640,480),Image.Resampling.NEAREST))
        panels.append(colorize(gt))
        for arm in ['rgb','rgbd']:
            with np.load(BASE/f'exports/{arm}/scene_{scene:04d}_{frame}.npz') as a:mask=a['instances'];panel=rgb.copy();valid=mask>0;panel[valid]=(.45*rgb[valid]+.55*colorize(mask)[valid]).astype(np.uint8);panels.append(panel)
        canvas=Image.new('RGB',(1280,1020),'white');draw=ImageDraw.Draw(canvas)
        for i,(label,panel) in enumerate(zip(['RGB input','Visible ground truth (colors are local IDs)','RGB prediction overlay','RGBD prediction overlay'],panels)):
            x=(i%2)*640;y=(i//2)*510;draw.text((x+6,y+6),f'Scene {scene:04d}, frame {frame}: {label}',fill='black');canvas.paste(Image.fromarray(panel),(x,y+30))
        file=OUT/f'figures/scene_{scene:04d}_{frame}.png';file.parent.mkdir(parents=True,exist_ok=True);canvas.save(file);inventory.append(dict(path=file.relative_to(ROOT).as_posix(),sha256=sha(file)))
    once(OUT/'figures/index.json',dict(rule=read(CONFIG)['figure_rule'],files=inventory))
def run(mode):
    c=read(CONFIG);baseline=bind(c);start=time.monotonic();uid=uuid.uuid4().hex;diagnostics={};metrics={};readonly=mode=='verify'
    def guard():
        need(time.monotonic()-start<c['resources']['wall_seconds'],'Evaluation wall cap reached')
        prior=sum(read(p).get('seconds',0) for folder in ['logs/P4/full_baseline_v22','logs/P4/full_baseline_v22r1','logs/P4/full_baseline_v22r1/evaluation'] for p in (ROOT/folder).glob('invocation_*.json'))
        need(prior+time.monotonic()-start<14400,'Whole P4 cumulative wall budget reached')
        need(sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<c['resources']['output_bytes'],'Evaluation output cap reached')
        need(shutil.disk_usage(ROOT).free>60*1024**3,'Disk reserve reached')
        need(official.common.storage.peak_memory_bytes()<c['resources']['rss_bytes'],'Evaluation RSS cap reached')
    try:
        for arm in ['rgb','rgbd']:
            inputs,matches,diag=scan(arm,independent=readonly,make_inputs=not readonly);diagnostics[arm]=diag;metrics[arm]={}
            for scene in [r['scene'] for r in read(BASE/(arm+'_full_development.json'))['scenes']]+[None]:
                guard();scope='pooled' if scene is None else f'scene_{scene:04d}';path=OUT/f'official/{arm}_{scope}.json';subset,matching=scoped(inputs,matches,scene)
                if path.exists():value=read(path)
                else:
                    need(not readonly,'Missing official AP artifact')
                    intents=sum(json.loads(line)['kind']=='intent' for line in (LOG/'official_calls.jsonl').read_text().splitlines()) if (LOG/'official_calls.jsonl').exists() else 0
                    need(intents<c['resources']['max_official_calls'],'Official call budget reached');callid=uuid.uuid4().hex;event('intent',id=callid,arm=arm,scope=scope)
                    value=official.official_evaluation(subset);once(path,value);event('returned',id=callid,sha256=sha(path));print(json.dumps(dict(arm=arm,scope=scope,metrics=primary(value))),flush=True)
                audit_ap(value,matching);metrics[arm][scope]=primary(value)
            once(OUT/(arm+'_boundary_and_counts.json'),diag);guard()
        report=dict(status='complete_paired_development_evaluation',frames_per_arm=160,scenes=10,arms={a:dict(scene_macro={m:float(np.mean([v[m] for k,v in metrics[a].items() if k!='pooled'])) for m in ['AP','AP50','AP75']},pooled=metrics[a]['pooled'],boundary_scene_macro_f1=diagnostics[a]['boundary_scene_macro_f1']) for a in ['rgb','rgbd']},by_scene=metrics,new_model_forward_calls=0,new_training_updates=0,independent_primary_precision_recall_scores_match=True)
        once(OUT/'metrics.json',report)
        if not readonly:figures(diagnostics)
        need(all(stat(ROOT/p)==v for p,v in baseline.items()),'Original/input file changed during evaluation')
        if readonly:
            once(OUT/'readonly.json',dict(status='passed',at=now(),source_count=len(baseline),decoded_frames=320,full_scene_summaries_verified=True,official_primary_arrays_independently_verified=True,originals_unchanged=True,new_model_forward_calls=0,new_training_updates=0,new_official_calls=0,metrics_sha256=sha(OUT/'metrics.json')))
        else:once(LOG/'production_outputs.json',{p.relative_to(ROOT).as_posix():stat(p) for p in OUT.rglob('*') if p.is_file()})
        if readonly:need(all(stat(ROOT/p)==v for p,v in read(LOG/'production_outputs.json').items()),'Evaluation outputs changed during readonly')
        once(LOG/f'invocation_{uid}.json',dict(status='passed',mode=mode,seconds=time.monotonic()-start));print(json.dumps(dict(status='passed',mode=mode,arms=report['arms'])),flush=True)
    except BaseException as e:
        once(LOG/f'invocation_{uid}.json',dict(status='failed_preserved',mode=mode,seconds=time.monotonic()-start,error=repr(e)));raise
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','compute','verify']);a=p.parse_args()
    if a.mode=='prepare':prepare()
    else:
        with project_lock(ROOT):run(a.mode)
if __name__=='__main__':main()
