"""Frozen P5 mask evaluation; independent cached reconstruction, no GPU imports."""
import os,sys,time,argparse,shutil
from pathlib import Path
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p5_common_v23 import *
from scripts.pipeline import project_lock
from scripts import p4_complete_evaluation_v22r2 as inherited
from scripts import evaluate_mask_ap_v17 as official
from scripts import accept_mask_ap_v17 as reference
from scripts.extract_mask_scores_v17 import decode_with_scores
import numpy as np
from PIL import Image

OUT=BASE/'evaluation';EVALCONFIG=ROOT/'configs/p5_evaluation_v23.json'

def prepare():
    # Scientific thresholds were frozen before any research; this freezes the CPU implementation.
    c=read(CONFIG);need(not EVALCONFIG.exists(),'Evaluation already frozen')
    paths=['scripts/p5_evaluate_v23.py','scripts/p4_complete_evaluation_v22r2.py','scripts/evaluate_mask_ap_v17.py','scripts/accept_mask_ap_v17.py','scripts/mask_ap_common_v17.py','scripts/extract_mask_scores_v17.py','scripts/p5_common_v23.py','configs/p5_quality_gate_v23.json']
    old=read(ROOT/'configs/p4_complete_evaluation_v22r2.json')
    grid=np.arange(480*640,dtype=np.int32).reshape(480,640);need(np.array_equal(inherited.high_resolution(grid),inherited.indexed_resize(grid,720,1280)),'Resize fixture differs')
    need(inherited.maximum_matching(np.array([[.8,.6],[.7,0.]]))==2,'Matching fixture failed')
    once(EVALCONFIG,dict(protocol_sha256=sha(CONFIG),protected={p:stat(ROOT/p) for p in paths},environment=old['environment'],mask_ap=old['mask_ap'],scope='14 complete160-frame AP sets;14 graded20-frame F1/IoU sets; all previously exposed development data; paired one seed',official_calls=154,max_official_calls=160))
    once(LOG/'evaluation_freeze.json',dict(config_sha256=sha(EVALCONFIG),new_model_calls=0));print('P5 CPU evaluation frozen',flush=True)

def bind():
    c=verify();ec=read(EVALCONFIG);need(sha(EVALCONFIG)==read(LOG/'evaluation_freeze.json')['config_sha256'],'Evaluation config changed')
    need('torch' not in sys.modules,'CPU evaluation must not import torch')
    for p,v in ec['protected'].items():need(stat(ROOT/p)==v,'Evaluation source changed '+p)
    for p,h in ec['environment']['files'].items():need(sha(ROOT/p)==h,'Evaluator environment changed '+p)
    frozen=read(LOG/'segmentation_freeze.json')
    for p,v in frozen['protected'].items():need(stat(ROOT/p)==v,'Segmentation dependencies changed '+p)
    need(read(BASE/'exports_complete.json')['total_frame_entries']==2520,'Full P5 export missing')
    paths=[p for p in (BASE/'exports').rglob('*') if p.is_file()]+[BASE/'exports_complete.json',BASE/'selection.json',BASE/'first_five_gate.json',BASE/'paired_unit_training.json']
    snap={p.relative_to(ROOT).as_posix():stat(p) for p in paths};once(LOG/'evaluation_input_binding.json',snap);return c,snap

def scan(c,item,independent=False,make_inputs=False):
    records=c['development_records'] if item['official_AP'] else c['monitor_records'];name=item['name'];report=read(BASE/'exports'/(name+'.json'));need(sha(BASE/'exports'/(name+'.json'))==item['report_sha256'],'Report hash changed');images=[];annotations=[];detections=[];matching=[];rows=[]
    for rec in records:
        stem=f"scene_{rec['scene']:04d}_{rec['frame']}";cache=BASE/'exports'/name/(stem+'.npz');old=read(cache.with_suffix('.json'));need(sha(cache)==old['cache_sha256'],'Cache changed')
        with np.load(cache,allow_pickle=False) as v:
            h={k:v[k] for k in ['foreground_probability','center_probability','offsets']}
            if independent:
                rebuilt=reference.independent_decode(**h,decoder=c['decoder']);pred,fg,scores=rebuilt['instances'],rebuilt['foreground'],rebuilt['retained_instances']
            else:pred,fg,scores,_=decode_with_scores(h,**c['decoder'])
            need(np.array_equal(pred,v['instances']) and np.array_equal(fg,v['foreground']) and scores==old['scores'],'Independent decoder mismatch')
            gate=v['gate'];need(np.isfinite(gate).all() and gate.min()>=.5 and gate.max()<=1,'Invalid bounded gate')
            if item['arm']=='fixed' or item['ablation']=='unit':need(np.all(gate==1),'Unit gate changed')
            if item['ablation']=='mean':need(np.ptp(gate)==0,'Spatial mean ablation is not constant')
        with Image.open(ROOT/c['data_root']/rec['files']['label']) as im:raw=np.asarray(im);gt=np.asarray(im.resize((640,480),Image.Resampling.NEAREST),np.int64)
        if independent:need(np.array_equal(gt,inherited.indexed_resize(raw,480,640)),'Reference resize mismatch')
        counts=inherited.frame_check(pred,fg,gt,old['metrics']);rows.append(dict(scene=rec['scene'],frame=rec['frame'],counts=counts,foreground_intersection=old['metrics']['foreground_intersection'],foreground_union=old['metrics']['foreground_union'],boundary=inherited.boundary_counts(pred,gt)))
        if not item['official_AP']:continue
        high=inherited.indexed_resize(pred,720,1280) if independent else inherited.high_resolution(pred);pids=[int(x) for x in np.unique(high) if x>0];gids=[int(x) for x in np.unique(raw) if x>0];image_id=rec['scene']*1000+int(rec['frame']);images.append(dict(id=image_id,height=720,width=1280));gtids=[];dtids=[]
        for gid in gids:
            mask=raw==gid;annid=len(annotations)+1;gtids.append(annid);row=dict(id=annid,image_id=image_id,category_id=1,area=int(mask.sum()),iscrowd=0)
            if make_inputs:
                yy,xx=np.nonzero(mask);row.update(segmentation=official.encode_mask(mask),bbox=[int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)])
            annotations.append(row)
        for score in scores:
            need(score['instance_id'] in pids,'Mask lost original-resolution support');row=dict(image_id=image_id,category_id=1,score=score['score'])
            if make_inputs:row['segmentation']=official.encode_mask(high==score['instance_id'])
            detections.append(row);dtids.append(len(detections))
        overlap=reference.dense_overlap(high,raw,pids,gids)['iou'];matching.append(reference.independent_image_matching(image_id,gtids,dtids,[x['score'] for x in scores],overlap))
    scenes=[]
    for sc in report['scenes']:
        sub=[r for r in rows if r['scene']==sc['scene']];n={k:sum(r['counts'][k] for r in sub) for k in ['tp','fp','fn']};f1=2*n['tp']/(2*n['tp']+n['fp']+n['fn']) if 2*n['tp']+n['fp']+n['fn'] else 0.;u=sum(r['foreground_union'] for r in sub);iou=sum(r['foreground_intersection'] for r in sub)/u if u else 1.;need(f1==sc['f1'] and iou==sc['foreground_iou'],'Scene aggregate mismatch');scenes.append(dict(scene=sc['scene'],f1=f1,foreground_iou=iou,predicted_instances=n['tp']+n['fp'],boundary=inherited.boundary_f1([r['boundary'] for r in sub])))
    need(len(rows)==item['frames'],'Frame scope mismatch');need(float(np.mean([s['f1'] for s in scenes]))==report['scene_macro_f1'],'Macro F1 mismatch')
    return dict(gt=dict(info={},images=images,categories=[dict(id=1,name='object')],annotations=annotations),detections=detections),matching,dict(frames=len(rows),scene_macro_f1=report['scene_macro_f1'],scene_macro_foreground_iou=report['scene_macro_foreground_iou'],scenes=scenes)

def adoption(c,metrics):
    def key(arm,cond,kind='core',ablation='learned'):return f'{kind}_{arm}_{ablation}_{cond}'
    degraded=[x for x in c['core_conditions'] if x!='clean'];gain=float(np.mean([metrics[key('quality',d)]['scene_macro_AP']-metrics[key('fixed',d)]['scene_macro_AP'] for d in degraded]));clean_loss=metrics[key('fixed','clean')]['scene_macro_AP']-metrics[key('quality','clean')]['scene_macro_AP'];scene_gains={}
    for sc in sorted({r['scene'] for r in c['development_records']}):
        scope=f'scene_{sc:04d}';scene_gains[str(sc)]=float(np.mean([metrics[key('quality',d)]['official'][scope]['AP']-metrics[key('fixed',d)]['official'][scope]['AP'] for d in degraded]))
    joint=metrics[key('quality','joint')]['scene_macro_AP'];ablation_gain={a:joint-metrics[key('quality','joint','ablation',a)]['scene_macro_AP'] for a in ['unit','mean']}
    gates=dict(quality=read(BASE/'quality_screen.json')['quality_usable'],clean_f1=all(metrics[key(a,'clean')]['scene_macro_f1']>=.5 for a in ['fixed','quality']),stability=read(BASE/'selection.json')['stability_passed'],noncollapsed_all_clean_scenes=all(s['predicted_instances']>0 for a in ['fixed','quality'] for s in metrics[key(a,'clean')]['scenes']),degraded_AP_gain=gain>=.01,clean_AP_preserved=clean_loss<=.01,scene_improvement=sum(v>0 for v in scene_gains.values())>=6,spatial_ablation=all(v>0 for v in ablation_gain.values()))
    return dict(gates=gates,scientifically_adopt=all(gates.values()),mean_degraded_AP_gain=gain,clean_AP_loss=clean_loss,by_scene_degraded_gain=scene_gains,joint_ablation_gain=ablation_gain,engineering_completion_separate_from_efficacy=True,scope='P5 exposed-development segmentation; no independent generalization, no grasp AP, no P8 completion')

def run(mode):
    c,snap=bind();readonly=mode=='verify';start=time.monotonic();uid=uuid.uuid4().hex;usage=LOG/('usage_'+uid+'.json');previous=sum(read(p)['seconds'] for p in LOG.glob('usage_*.json'));metrics={};items=read(BASE/'exports_complete.json')['sets'];outcome='failed';error=None
    def usage_save(state):
        row=dict(at=now(),seconds=time.monotonic()-start,state=state,phase='evaluation_'+mode,error=error);tmp=usage.with_suffix('.tmp');tmp.write_text(json.dumps(row),encoding='utf-8');os.replace(tmp,usage)
    def guard():
        usage_save('running');need(previous+time.monotonic()-start<c['resources']['wall_seconds'],'Cumulative P5 time budget reached');need(official.common.storage.peak_memory_bytes()<c['resources']['rss_bytes'],'RSS limit');need(shutil.disk_usage(ROOT).free>=c['resources']['disk_reserve_bytes'],'Disk reserve');need(sum(p.stat().st_size for d in [RUN,BASE,LOG] for p in d.rglob('*') if p.is_file())<c['resources']['output_bytes'],'Output budget')
    try:
        for item in items:
            guard();name=item['name'];receipt=OUT/'readonly_sets'/(name+'.json')
            if readonly and receipt.exists():
                saved=read(receipt);need(all(stat(ROOT/p)==v for p,v in saved['outputs'].items()),'Previously reviewed output changed');metrics[name]=read(OUT/'sets'/(name+'.json'));continue
            inputs,matches,diag=scan(c,item,readonly,not readonly);value=dict(diag,official={})
            if item['official_AP']:
                for scene in [x['scene'] for x in diag['scenes']]+[None]:
                    guard();scope='pooled' if scene is None else f'scene_{scene:04d}';path=OUT/'official'/f'{name}_{scope}.json';subset,matching=inherited.scoped(inputs,matches,scene)
                    if path.exists():ap=read(path)
                    else:
                        need(not readonly,'Missing official AP artifact');calls=[x for x in events() if x['kind']=='official_AP_intent'];need(len(calls)<160,'Official AP budget exhausted');cid=uuid.uuid4().hex;event('official_AP_intent',call_id=cid,set=name,scope=scope);ap=official.official_evaluation(subset);once(path,ap);event('official_AP_returned',call_id=cid,set=name,scope=scope,path=path.relative_to(ROOT).as_posix(),sha256=sha(path))
                    inherited.audit_ap(ap,matching);value['official'][scope]=inherited.primary(ap)
                value['scene_macro_AP']=float(np.mean([v['AP'] for k,v in value['official'].items() if k!='pooled']))
            report=OUT/'sets'/(name+'.json');once(report,value);metrics[name]=value
            if readonly:
                outputs=[report]+list((OUT/'official').glob(name+'_*.json'));once(receipt,dict(passed=True,frames=item['frames'],outputs={p.relative_to(ROOT).as_posix():stat(p) for p in outputs},independent_decoding_matching_and_primary_AP_arrays=True,new_model_calls=0,new_official_calls=0))
            status('P5_evaluation_'+mode,completed_sets=len(metrics),total_sets=28);print(json.dumps(dict(mode=mode,set=name,f1=value['scene_macro_f1'],AP=value.get('scene_macro_AP'))),flush=True)
        decision=adoption(c,metrics);once(OUT/'decision.json',decision)
        for p,v in snap.items():need(stat(ROOT/p)==v,'Evaluation modified research inputs')
        if readonly:once(OUT/'readonly.json',dict(passed=True,verified_frame_entries=2520,verified_official_blocks=154,all_sources_inputs_unchanged=True,new_model_calls=0,new_training=0,new_official_calls=0,decision_sha256=sha(OUT/'decision.json')))
        else:once(LOG/'evaluation_output_binding.json',{p.relative_to(ROOT).as_posix():stat(p) for p in OUT.rglob('*') if p.is_file()})
        if readonly:need(all(stat(ROOT/p)==v for p,v in read(LOG/'evaluation_output_binding.json').items()),'Readonly altered outputs')
        outcome='passed'
    except BaseException as e:error=repr(e);raise
    finally:usage_save(outcome)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','compute','verify']);a=ap.parse_args()
    with project_lock(ROOT):
        if a.mode=='prepare':prepare()
        else:run(a.mode)
if __name__=='__main__':main()
