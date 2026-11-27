"""R1: immutable v51 cache audit and a new raw-score control, no new physics.

The raw control re-runs the pinned NMS and official object/global quotas using
complete singleton labels, after reproducing all 350 existing result supports.
It is a cached-label metric computation, not a new eval_grasp experiment.
"""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,hashlib,uuid,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runtime/build/graspnetAPI_eval_v7'))
import numpy as np
from src import p56_metrics_v51 as math51

OUT=ROOT/'results/simulation_v52/audit'
DOC=ROOT/'docs/P5-P6模块与Raw基线审计_v52.txt'
LIMIT_SECONDS=300
LIMIT_OUTPUT_BYTES=100*1024**2
GROUPS=('Raw','A','C0')

def need(value,message):
    if not bool(value):raise RuntimeError(message)

def read(path):return json.loads(Path(path).read_text('utf-8-sig'))

def identity(path):
    path=Path(path);s=path.stat()
    with path.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(sha256=h,bytes=s.st_size,mtime_ns=s.st_mtime_ns)

def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()

def write_once(path,value):
    path=Path(path);need(path.resolve().is_relative_to(ROOT),'Output escapes project')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def sealed(path,value):
    write_once(path,value);write_once(Path(str(path)+'.receipt.json'),identity(path))

def load_sealed(path):
    need(identity(path)==read(Path(str(path)+'.receipt.json')),'Sealed source changed: '+rel(path))
    return read(path)

def append_event(row):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'events.jsonl').open('a',encoding='utf8') as f:
        f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def now():
    from datetime import datetime
    return datetime.now().astimezone().isoformat()

def exact_indices(source,rows):
    used=set();result=[]
    for row in rows:
        matches=[int(i) for i in np.flatnonzero(np.all(source==row,axis=1)) if int(i) not in used]
        need(matches,'NMS returned geometry absent from fixed candidates')
        result.append(matches[0]);used.add(matches[0])
    return result

def select_cached(candidates,scores,pool_indices,physics,Group):
    """Same official NMS/quota algorithm; invariant GT slots come from full20 labels.

    Slots/physical labels are evaluation-only. No label affects method scores.
    Unique ordinal confidences preserve stable raw-score ties as in frozen v51.
    """
    normalized,order=math51.normalize_ranking(candidates,scores,pool_indices)
    nms=Group(normalized.copy()).nms(.03,30./180*np.pi)
    positions=exact_indices(normalized,nms.grasp_group_array)
    slots=[]
    for p,row in zip(positions,nms.grasp_group_array):
        key=int(pool_indices[int(order[p])]);label=physics[key]
        need(label['grasp'][1:]==row[1:].tolist(),'Cached singleton geometry differs after normalization')
        slots.append(int(label['object_slot']))
    slots=np.asarray(slots,dtype=np.int32);pre=[]
    for slot in sorted(set(slots.tolist())):
        group=nms[slots==slot];group.sort_by_score()
        pre.append((slot,group[:10].grasp_group_array))
    all_rows=np.vstack([rows for _,rows in pre]);need(len(all_rows),'Empty official support')
    rank=np.argsort(all_rows[:,0])[::-1]
    threshold=all_rows[rank[min(49,len(rank)-1)],0]
    selected=[]
    for slot,rows in pre:
        rows=rows[rows[:,0]>=threshold]
        for row,p in zip(rows,exact_indices(normalized,rows)):
            key=int(pool_indices[int(order[p])]);label=physics[key]
            selected.append(dict(index=key,object_slot=slot,confidence=float(row[0]),
                                 grasp=row.tolist(),score=float(label['score']),collision=bool(label['collision'])))
    selected.sort(key=lambda row:-row['confidence'])
    need(len(selected)<=20 and len({r['index'] for r in selected})==len(selected),'Invalid selected support')
    return selected,dict(normalized=normalized.tolist(),local_order=order.tolist(),
                         nms_count=len(nms),object_slots_from_complete_singleton_labels=True,
                         selected_pool_indices=[r['index'] for r in selected])

def independent_metrics(selected):
    prefix=[[sum(0<r['score']<=mu and not r['collision'] for r in selected[:k])/k
             for mu in math51.FRICTIONS] for k in range(1,51)]
    return dict(AP=float(np.mean(prefix)),AP_04=float(np.mean(np.asarray(prefix)[:,1])),
        prefix=prefix,selected=len(selected),topk={str(k):dict(mu04=prefix[k-1][1],
        six_average=float(np.mean(prefix[k-1])),available=min(k,len(selected)),denominator=k,
        any_mu04=any(0<r['score']<=.4 and not r['collision'] for r in selected[:k])) for k in math51.TOP_K})

def summarize(rows):
    result={}
    for group in GROUPS:
        subset=[r for r in rows if r['group']==group]
        need(len(subset)==50,'Incomplete fifty-case scope')
        condition={c:float(np.mean([r['metrics']['AP'] for r in subset if r['condition']==c])) for c in math51.CONDITIONS}
        degraded=[r for r in subset if r['condition']!='clean']
        result[group]=dict(condition_AP=condition,clean_AP=condition['clean'],
            degraded_AP=float(np.mean([condition[c] for c in math51.CONDITIONS if c!='clean'])),
            degraded_TopK={str(k):dict(mu04=float(np.mean([r['metrics']['topk'][str(k)]['mu04'] for r in degraded])),
                six_average=float(np.mean([r['metrics']['topk'][str(k)]['six_average'] for r in degraded]))) for k in math51.TOP_K},
            selected_count=sum(r['metrics']['selected'] for r in subset),complete_pool_denominator=1000)
    return result

def main():
    need(Path.cwd().resolve()==ROOT,'Explicit project working directory required')
    need(not (OUT/'events.jsonl').exists() and not (OUT/'summary.json').exists() and not DOC.exists(),
         'Prior R1 attempt exists; no implicit overwrite or retry')
    started=time.monotonic();call=uuid.uuid4().hex;old={};nms_calls=0;replayed=0
    def guard():
        need(time.monotonic()-started<LIMIT_SECONDS,'R1 300-second cap exceeded')
        need(not (ROOT/'results/simulation_v52/STOP').exists(),'STOP requested')
        if OUT.exists():need(sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<LIMIT_OUTPUT_BYTES,'R1 output cap exceeded')
    def bind(path):
        path=Path(path);expected=identity(path);name=rel(path)
        if name in old:need(old[name]==expected,'Source changed during audit: '+name)
        else:old[name]=expected
        receipt=Path(str(path)+'.receipt.json')
        if receipt.exists():
            receipt_name=rel(receipt);receipt_identity=identity(receipt)
            if receipt_name in old:need(old[receipt_name]==receipt_identity,'Historical receipt changed '+receipt_name)
            else:old[receipt_name]=receipt_identity
        return path
    append_event(dict(at=now(),event='intent',id=call,kind='R1_cached_raw_control',
                      scientific_model_calls=0,official_physics_calls=0,wall_cap_seconds=LIMIT_SECONDS))
    try:
        config_path=bind(ROOT/'configs/p56_joint_v51.json');config=read(config_path)
        need(identity(config_path)==read(bind(ROOT/'logs/joint_validation_v51/freeze.json'))['config'],'v51 freeze differs')
        close=read(bind(ROOT/'results/joint_validation_v51r2/closeout.json'))
        need(close['engineering_recovery_complete'] and close['required_scientific_outputs_complete'],'v51r2 is not closed')
        review=load_sealed(bind(ROOT/'results/joint_validation_v51/official/review.json'))
        need(review['passed'] and review['official_units']==350,'Complete old350 review missing')
        need(len(config['cases'])==50 and config['groups']==list(math51.GROUPS),'Unexpected frozen scope')
        # Imported solely for the pinned GraspGroup NMS; prohibit full physics paths.
        from graspnetAPI import GraspGroup
        from graspnetAPI.utils import eval_utils as official
        def prohibited(*args,**kwargs):raise RuntimeError('R1 forbids new official physical scoring/model loading')
        originals={name:getattr(official,name) for name in ('eval_grasp','get_grasp_score','load_dexnet_model')}
        for name in originals:setattr(official,name,prohibited)
        for name in ['src/p56_metrics_v51.py','src/p56_geometry_v51.py','src/p56_segmentation_v51.py',
                     'scripts/p56_joint_v51.py','scripts/p56_official_v51.py','scripts/verify_grasp_geometry_v10.py',
                     'configs/p6_local_correspondence_v50.json','results/P6/local_correspondence_v50/model.json',
                     'configs/p5_quality_gate_v23.json','configs/p4_full_baseline_v22r2.json','configs/candidate_v30.json',
                     'configs/risk_calibration_v31.json','runtime/pipeline/state.json',
                     'docs/P5-P6联合候选选择验证协议_v51.txt','docs/P5-P6联合候选选择验证结果_v51r2.txt']:
            bind(ROOT/name)
        for name in ('graspnetAPI/grasp.py','graspnetAPI/utils/eval_utils.py'):
            bind(ROOT/'runtime/build/graspnetAPI_eval_v7'/name)
        # Stage1 reproduces all350 from complete singleton labels before any new Raw metric.
        units=[]
        for case in config['cases']:
            guard();source=ROOT/case['source_dir'];key=f"scene_{case['scene']:04d}_{case['frame']:04d}_{case['condition']}"
            pool=load_sealed(bind(source/'pool.json'));generation=load_sealed(bind(source/'generation.json'))
            label_path=bind(source/'labels.json');labels=load_sealed(label_path)
            need(identity(label_path)==review['checked'][rel(label_path)],'Labels changed after v51 review')
            obs=bind(source/'observations.npz');need(identity(obs)==generation['artifact'],'Observed cache differs')
            ranking_path=bind(ROOT/case['ranking_file']);ranking=read(ranking_path)
            need(pool['all_methods_same_pool'] and not pool['GT_used'] and len(pool['indices'])==20,'Fixed20 observation-only pool missing')
            need(ranking['pool_indices']==pool['indices'],'Ranking pool differs')
            with np.load(obs,allow_pickle=False) as values:candidates=values['candidates'][pool['indices']].copy()
            physics={int(r['index']):r for r in labels['records']}
            need(len(physics)==20 and set(physics)==set(pool['indices']),'Complete singleton labels missing; never reuse a truncated group')
            need(candidates.shape==(20,17) and candidates.dtype==np.float32,'Candidate array schema differs')
            for local,key_id in enumerate(pool['indices']):
                clipped=candidates[local].copy();clipped[1]=np.clip(clipped[1],0,.1)
                need(physics[key_id]['grasp'][1:]==clipped[1:].tolist(),'Singleton candidate geometry mismatch')
            controls={}
            for group in math51.GROUPS:
                guard();p=bind(ROOT/'results/joint_validation_v51/official/results'/key/(group+'.json'))
                need(identity(p)==review['checked'][rel(p)],'Old scored result changed')
                saved=load_sealed(p)
                selected,details=select_cached(candidates,ranking['scores'][group],pool['indices'],physics,GraspGroup);nms_calls+=1
                need(selected==saved['selected'] and details['normalized']==saved['normalized'] and details['local_order']==saved['local_order'],
                     'Cached selection does not reproduce existing official result '+key+'/'+group)
                rebuilt=independent_metrics(selected)
                need(rebuilt==saved['metrics'],'Cached metric reconstruction differs '+key+'/'+group)
                replayed+=1
                if group in GROUPS:controls[group]=dict(selected=selected,metrics=rebuilt,selection=details,reused_official_result=rel(p))
            units.append(dict(case=case,key=key,pool=pool,candidates=candidates,physics=physics,ranking=ranking,controls=controls))
        need(replayed==350,'Raw computation prohibited before full selection replay')
        sealed(OUT/'selection_replay.json',dict(at=now(),passed=True,cases=50,groups=7,
             complete_support_replays=replayed,new_official_physics_calls=0,new_model_calls=0,
             method='Pinned NMS plus complete singleton object-slot assignment, top10/object and global50 threshold',
             all_selected_geometry_physics_AP_TopK_exact=True))
        rows=[];policy_files=[];case_files=[]
        for unit in units:
            guard();case,key,candidates=unit['case'],unit['key'],unit['candidates'];pool=unit['pool']['indices']
            raw,details=select_cached(candidates,candidates[:,0],pool,unit['physics'],GraspGroup);nms_calls+=1
            metrics=independent_metrics(raw)
            need(metrics==math51.grasp_metrics([r['score'] for r in raw],np.asarray([r['collision'] for r in raw],bool)),
                 'Independent Raw metric loops disagree with frozen implementation')
            result=dict(case={k:case[k] for k in ('scene','frame','condition')},key=key,
                        metric_scope='fixed20 pool GraspNet-protocol cached-singleton-label AP/TopK; not a new actual official physics call',
                        groups=dict(Raw=dict(selected=raw,metrics=metrics,selection=details,new_official_physics_calls=0),**unit['controls']))
            destination=OUT/'cases'/(key+'.json');sealed(destination,result);case_files.append(rel(destination))
            # This separate interface deliberately contains no GT physical label/slot/success information.
            scores=dict(Raw=candidates[:,0].astype(float).tolist(),A=unit['ranking']['scores']['A'],C0=unit['ranking']['scores']['C0'])
            policy_candidates=candidates.copy();policy_candidates[:,1]=np.clip(policy_candidates[:,1],0,.1)
            policy=dict(case=result['case'],key=key,candidate_rows=policy_candidates.tolist(),pool_indices=pool,
                score_by_group=scores,ranked_pool_indices={g:[int(pool[i]) for i in np.argsort(-np.asarray(v),kind='stable')] for g,v in scores.items()},
                source_observations=case['source_dir']+'/observations.npz',
                source_ranking=case['ranking_file'],GT_in_policy_inputs=False,
                inherited_official_width_clip_m=[0.,.1],candidate_positions_and_rotations_unchanged=True,
                coordinate_frame='original GraspNet RealSense camera coordinates; no simulation transform applied',
                robot_feasibility_checked=False)
            policy_path=OUT/'policy_inputs'/(key+'.json');sealed(policy_path,policy);policy_files.append(rel(policy_path))
            for group,value in result['groups'].items():rows.append(dict(scene=case['scene'],condition=case['condition'],group=group,metrics=value['metrics']))
        aggregates=summarize(rows)
        contrasts={name:aggregates[a]['degraded_AP']-aggregates[b]['degraded_AP'] for name,a,b in
                   [('A_minus_Raw','A','Raw'),('C0_minus_A','C0','A'),('C0_minus_Raw','C0','Raw')]}
        scene_contrasts={name:{str(scene):float(np.mean([r['metrics']['AP'] for r in rows if r['scene']==scene and r['condition']!='clean' and r['group']==a]))-
                    float(np.mean([r['metrics']['AP'] for r in rows if r['scene']==scene and r['condition']!='clean' and r['group']==b])) for scene in math51.SCENES}
                    for name,a,b in [('A_minus_Raw','A','Raw'),('C0_minus_A','C0','A'),('C0_minus_Raw','C0','Raw')]}
        for name,expected in old.items():guard();need(identity(ROOT/name)==expected,'Historical file changed '+name)
        lines=['P5-P6模块与Raw原始分数基线审计 v52',now(),'',
          '范围：R1缓存审计；不是新训练、官方物理调用、机器人执行或独立确认。既有v51r2失败/负结果及全部历史文件保留。',
          '完整20候选singleton物理标签覆盖已核；先逐项复现旧350组的NMS、对象配额、选取、AP及TopK，再计算50例Raw。没有使用旧A/C0截断支持冒充Raw。',
          '所有方法固定同20候选、原场景观察过滤及NMS；Raw为原候选分数排序，A为实例接触/确定性占用代理，C0为固定几何U。Raw仍是经过既有观察过滤的固定池控制，不是完整候选生成系统的全量官方测试AP。',
          '主结果：四退化AP均值；10场景等权，每场景4退化等权。数值为百分比；差为百分点。',
          '方法 | clean AP(%) | 四退化AP(%) | 四退化Top1/Top5/Top10(mu=.4,%)']
        for group,value in aggregates.items():lines.append(f"{group} | {value['clean_AP']*100:.6f} | {value['degraded_AP']*100:.6f} | "+'/'.join(f"{value['degraded_TopK'][str(k)]['mu04']*100:.4f}" for k in (1,5,10)))
        lines+=['差值：'+json.dumps({k:v*100 for k,v in contrasts.items()},ensure_ascii=False),'',
          '参数与来源：固定U是v50八训练场景864人工warp标签案例的Dirichlet(alpha=1/81)频率先验，81共享整数warp；深度为t3尺度5mm，非标准差5mm，该参照在v43/v45已存在。v51于2026-11-02T17:35:08冻结后未调参。它不是已校准真实传感器误差或安全概率。',
          '统一公式：raw_score * sum_g p(g)*association(g)*min(left(g),right(g)) * (1-clip(expected_occupied_ratio/.01,0,1))。背景/其他物体全部保留，原观察域0<z<2m，体素1cm，指长6cm、厚1cm、approach5cm，每侧最多32固定几何锚点。',
          '直接GT隔离：排序来自观测深度、预测实例/前景及固定训练先验；原先验不按本例真warp选择；GT对象归属/物理标签只用于本离线评价。policy_inputs独立文件不含物理标签/对象slot/GT成功信息。',
          '泄漏边界：v51十场景与分割/候选网络训练场景无交叠，但已经用于上游模型选择/准入及长期研究开发；本轮不得称盲测。固定5mm量级与已知noise5人工退化一致，不证明真实域标定。',
          'P5原科学收益门槛仍失败；P6学习式后验正式校准仍失败；C0/D0为参照模型，不能反向改变历史采用判定。D0的质量融合也不能仅凭个别TopK改善认为优越。',
          '仍缺独立贡献：C0去实例关联、warp-only和depth-only，以及新封存场景确认；目前不启动这些新实验。条件关联配对不是独立统计样本，不能直接声称因果/显著性。',
          '机器人适配：现有代理夹爪与Panda URDF尚需TCP/轴向/开口核对；原官方width上限10cm可能超Panda实际开口，不得静默修改候选位姿或宽度再声称相同候选。IK/碰撞/不可达失败必须保留。',
          '只允许已授权R2及最多48回合工程准入，完成暂停；本审计不授权R4/72或720/240正式实验。',
          '全部任务总进度45%；实验总进度56.25%；本R1完成不代表完整研究或科学门槛通过。']
        with DOC.open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
        guard();summary=dict(at=now(),completed=True,stage='R1',source_sha256=identity(Path(__file__)),
            all_bound_historical_files_unchanged=True,source_bindings=old,method_audit=dict(freeze_at=config['at'],frozen_method=config['method'],
            no_new_parameters_selected=True,P5_scientific_passed=False,P6_scientific_passed=False,
            direct_GT_model_input_seen=False,independent_confirmation=False,learned_U_not_adopted=True),
            full_previous_support_replays=350,new_cached_raw_cases=50,actual_NMS_calls=nms_calls,
            actual_official_eval_grasp_calls=0,actual_get_grasp_score_calls=0,actual_model_forward_calls=0,training_updates=0,
            aggregates=aggregates,contrasts_AP=contrasts,scene_contrasts_AP=scene_contrasts,
            cases=case_files,policy_inputs=policy_files,report=identity(DOC),
            elapsed_seconds=time.monotonic()-started,resources=dict(wall_seconds_limit=LIMIT_SECONDS,cpu_threads=2,output_bytes_limit=LIMIT_OUTPUT_BYTES),
            expanded_simulation_started=False,scientific_gate_changed=False)
        sealed(OUT/'summary.json',summary)
        append_event(dict(at=now(),event='returned',id=call,kind='R1_cached_raw_control',passed=True,summary=identity(OUT/'summary.json')))
        print(json.dumps(dict(completed=True,seconds=summary['elapsed_seconds'],full_previous_replays=350,new_raw_cases=50,
             official_physics_calls=0,aggregates=aggregates,contrasts_AP=contrasts,report=rel(DOC)),ensure_ascii=False))
    except BaseException as exc:
        append_event(dict(at=now(),event='failure',id=call,error=repr(exc),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started))
        raise

if __name__=='__main__':main()
