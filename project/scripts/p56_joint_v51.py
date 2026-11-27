"""One bounded, pre-registered P5/P6 downstream joint-use study, then stop."""
import os,sys,time,json,uuid,shutil,subprocess,traceback,argparse
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p56_common_v51 import *
from scripts.pipeline import project_lock

SOURCES=['scripts/p56_common_v51.py','scripts/p56_joint_v51.py','scripts/p56_official_v51.py',
         'src/p56_segmentation_v51.py','src/p56_geometry_v51.py','src/p56_metrics_v51.py',
         'tests/test_p56_segmentation_v51.py','tests/test_p56_geometry_v51.py','tests/test_p56_metrics_v51.py','tests/test_p56_controller_v51.py']
from src.p56_metrics_v51 import GATE_THRESHOLDS as GATES

def freeze():
    from src.p56_geometry_v51 import FROZEN_METHOD
    with project_lock(ROOT),own_lock():
        need(not CONFIG.exists(),'v51 is already frozen')
        tests=read(LOG/'internal_checks.json');review=read(LOG/'independent_review.json')
        for report in (tests,review):
            need(report['passed'],'Internal check or independent review failed')
            for name in SOURCES:need(report['source_sha256'][name]==identity(ROOT/name)['sha256'],'Reviewed source changed '+name)
        checks=['p5_asset_audit.json','p5_asset_audit_full_protected.json','p6_asset_audit.json','evaluation_asset_audit.json']
        need(review['protocol']==identity(PROTOCOL),'Reviewed protocol changed')
        for name in checks:
            audit=read(LOG/name)
            need(audit.get('passed',audit.get('integrity_passed',False)),'Asset audit failed '+name)
        protected={}
        for name in ['configs/p5_completion_v39.json','configs/p6_local_correspondence_v50.json','configs/risk_calibration_v31.json','configs/p8_ablation_v32.json']:
            old=read(ROOT/name)
            for path,expected in {**old['protected'],**old.get('reference_inputs',{})}.items():
                actual=identity(ROOT/path)
                need(all(actual[k]==v for k,v in expected.items() if k in actual),'Historical binding differs '+path)
                protected[path]=actual
            protected[name]=identity(ROOT/name)
        from src.p56_segmentation_v51 import dependency_paths
        cache_audit=read(LOG/'p5_asset_audit.json')
        for name in dependency_paths(ROOT):
            actual=identity(ROOT/name)
            if name in cache_audit['files_checked']:
                need(actual==cache_audit['files_checked'][name],'Audited P5 dependency differs '+name)
            protected[name]=actual
        for directory in ['results/P5/completion_v39','logs/P5/completion_v39','results/P6/local_correspondence_v50','logs/P6/local_correspondence_v50']:
            for path in (ROOT/directory).rglob('*'):
                if path.is_file():protected[rel(path)]=identity(path)
        for case in case_definitions():
            folder=ROOT/case['source_dir']
            for path in folder.rglob('*'):
                if path.is_file():protected[rel(path)]=identity(path)
        for name in SOURCES+[rel(PROTOCOL),'runtime/pipeline/state.json',
                  'scripts/p6_local_data_v50.py','scripts/research_cache_recovery_v44r1.py',
                  'src/p6_local_features_v50.py','src/p6_local_correspondence_v50.py',
                  'scripts/p8_geometry_v32.py','scripts/verify_grasp_geometry_v10.py','src/p8_math_v32.py']:
            protected[name]=identity(ROOT/name)
        for name in checks+['internal_checks.json','independent_review.json']:
            protected[rel(LOG/name)]=identity(LOG/name)
        # Entire local pinned official implementation is part of the immutable evaluator.
        for path in (ROOT/'runtime/build/graspnetAPI_eval_v7/graspnetAPI').rglob('*.py'):
            protected[rel(path)]=identity(path)
        preparation=read(LOG/'preparation_budget.json')
        need(preparation['finite_scope_closed'] is True,'Preparation scope not closed')
        protected[rel(LOG/'preparation_budget.json')]=identity(LOG/'preparation_budget.json')
        cfg=dict(version=51,at=now(),scope='Previously exposed development fixed-20-candidate conditional selection; not full GraspNet test or robot validation',
            cases=case_definitions(),groups=list(GROUPS),scenes=list(SCENES),conditions=list(CONDITIONS),
            protected=protected,method=FROZEN_METHOD,gates=GATES,
            all_tasks_percent=45,experiment_percent=56.25,
            preparation_accounting=preparation,
            progress_policy='No automatic main-delivery credit for this bounded screening; original P5/P6 scientific failures remain',
            limits=dict(controller_wall_seconds=7200,cpu_threads=2,tree_rss_bytes=6*1024**3,output_bytes=2*1024**3,
                free_disk_min=20*1024**3,p6_head_forward=100,feature_cases=100,official_eval_grasp=357,
                official_grasp_score_upper_bound=7140,new_segmentation_forward=0,new_grasp_forward=0,optimizer_updates=0),
            result_policy=dict(no_threshold_changes=True,all_fifty_cases_in_denominators=True,
               new_training=False,auto_expansion=False,require_user_approval_for_expansion=True),
            evaluation=dict(cases=case_definitions(),groups=list(GROUPS),reference_config='configs/p8_ablation_v32.json',
                out_dir='results/joint_validation_v51/official',log_dir='logs/joint_validation_v51/official',
                controller_state='runtime/pipeline/p56_joint_v51.json',call_cap=357,wall_seconds=7200,
                gates=GATES,concordance_scope='all'))
        need(shutil.disk_usage(ROOT).free>=cfg['limits']['free_disk_min'],'Free disk reserve insufficient')
        once(CONFIG,cfg);once(LOG/'freeze.json',dict(at=now(),config=identity(CONFIG),protected_count=len(protected)))
        verify(full=False);status('frozen_ready',case_count=50,groups=list(GROUPS),progress=0)
        handoff('P5-P6联合候选选择v51已冻结：50开发案例、同一20候选池、A/B/C/D加RGB与固定U参照；0新训练，7200秒运行硬帽、2CPU、6GiB RSS。先完整特征及复验，再统一官方评价/全量只读/cold/真实汇总后停止。扩大须用户确认，旧P5/P6失败不改。全部任务总进度45%、实验总进度56.25%。状态runtime/pipeline/p56_joint_v51.json；入口scripts/p56_joint_v51.py deliver；禁止重复启动。')
        print(json.dumps(dict(frozen=True,protected=len(protected),cases=50,groups=7)))

def neural_event(scope,row):
    event=dict(at=now(),event=row['event'],id=scope+':'+str(row['id']),kind='p6_head_forward',scope=scope,payload=row)
    if row['event']=='intent':
        previous=events(LOG/'neural_calls.jsonl')
        need(sum(e['event']=='intent' for e in previous)<100,'Actual head forward cap')
        need(not any(e['event']=='intent' and e['id']==event['id'] for e in previous),'Duplicate head call')
    append(LOG/'neural_calls.jsonl',event)

def save_arrays(path,**arrays):
    import numpy as np
    path.parent.mkdir(parents=True,exist_ok=True);need(not path.exists(),'Existing array payload retained')
    with path.open('xb') as f:np.savez_compressed(f,**arrays)

def features(replay=False):
    import numpy as np
    from src.p56_segmentation_v51 import load_case
    from src.p56_geometry_v51 import build_features,depth_scale_map
    from src.p6_local_features_v50 import local_features,FEATURE_NAMES
    from src.p6_local_correspondence_v50 import predict_warp
    c=verify();child_guard();model=read(ROOT/'results/P6/local_correspondence_v50/model.json')
    scope='replay' if replay else 'production';records=[];started=time.monotonic()
    for i,row in enumerate(c['cases']):
        child_guard();key=case_key(row['scene'],row['condition']);call_id=scope+':'+key
        previous=events(LOG/'feature_calls.jsonl')
        need(not any(r['event']=='intent' and r['id']==call_id for r in previous),'Duplicate feature call; preserve failed or unknown history')
        append(LOG/'feature_calls.jsonl',dict(at=now(),event='intent',id=call_id,kind='feature_case',case=row,scope=scope))
        try:
            case=load_case(row['scene'],row['condition'],root=ROOT)
            x=local_features(case['rgb'],case['observed_depth'],case['observed_valid'],case['pure_rgb_maps'],case['K'],case['original_size'])
            posterior=predict_warp(model['warp'],x[None],FEATURE_NAMES,guard=child_guard,on_event=lambda e:neural_event(call_id,e))[0]
            scale=depth_scale_map(case['observed_depth'],case['observed_valid'],model['depth'],guard=child_guard)
            value=build_features(case,learned_posterior=posterior,fixed_prior=np.asarray(model['empirical_prior']),depth_scale_map=scale,shape_model=model['shape'],guard=child_guard)
            idx=np.asarray(case['pool_indices'],int);cand=np.asarray(case['candidates'])[idx]
            need(len(idx)==20 and cand.shape==(20,17),'Frozen20 candidate scope differs')
            need(set(value['scores'])==set(GROUPS),'Missing group')
            for name,scores in value['scores'].items():
                a=np.asarray(scores);need(a.shape==(20,) and np.isfinite(a).all(),'Invalid ranking values '+name)
            need(value['pool_indices']==idx.tolist(),'Geometry pool identity differs')
            result=dict(value)
            result['geometry_subroutine_neural_calls']=result.pop('actual_neural_forward_calls')
            result.update(scene=row['scene'],frame=64,condition=row['condition'],
                source_observations=dict(path=row['source_dir']+'/observations.npz',identity=identity(ROOT/row['source_dir']/'observations.npz')))
            # Actual NN calls are separately evidenced, not inferred from nominal frame count.
            actual_events=[e for e in events(LOG/'neural_calls.jsonl') if e['scope']==call_id]
            need(len(actual_events)==2 and [e['event'] for e in actual_events]==['intent','returned'],'Head did not return once')
            destination=ROOT/row['ranking_file'];arrays=destination.with_name('uncertainty_inputs.npz')
            if replay:
                saved=read(destination)
                # Same frozen deterministic math must reproduce every score and failure marker, not only AP.
                actual=json.loads(json.dumps(result,allow_nan=False,default=encode))
                need(saved==actual,'Full joint feature replay differs '+key)
                with np.load(arrays,allow_pickle=False) as a:
                    for k,v in dict(features=x,posterior=posterior,scale=scale).items():need(np.array_equal(a[k],v),'Feature input replay differs '+key+'/'+k)
            else:
                save_arrays(arrays,features=x,posterior=posterior,scale=scale)
                once(destination,result)
            records.append(dict(key=key,ranking=identity(destination),uncertainty=identity(arrays)))
            append(LOG/'feature_calls.jsonl',dict(at=now(),event='returned',id=call_id,kind='feature_case',scope=scope))
            atomic(LOG/('feature_'+scope+'_progress.json'),dict(at=now(),completed=i+1,total=50,current=key))
        except BaseException as exc:
            append(LOG/'failure_cases.jsonl',dict(at=now(),phase=scope,case=row,error=repr(exc),traceback=traceback.format_exc(),candidate_denominator_preserved=True))
            raise
    once(LOG/('feature_review.json' if replay else 'features_complete.json'),dict(at=now(),passed=True,cases=len(records),records=records,seconds=time.monotonic()-started,complete_numeric_replay=replay))

def run():
    from scripts.research_cache_recovery_v44r1 import process_tree_rss
    with project_lock(ROOT),own_lock():
        start=time.monotonic();launch_epoch=time.time()
        c=verify(full=True)
        need(not (OUT/'closeout.json').exists(),'v51 already completed; do not restart')
        need(not events(LOG/'children.jsonl'),'Prior run exists; no silent retry or budget reset')
        token=uuid.uuid4().hex;deadline=launch_epoch+c['limits']['controller_wall_seconds'];peak=0;last_resource_check=-1e30;orphan_stop_unconfirmed=False
        def guard(force=False):
            nonlocal peak,last_resource_check
            need(time.time()<deadline,'Cumulative controller wall cap reached')
            need(not (LOG/'STOP').exists(),'STOP marker received')
            if not force and time.monotonic()-last_resource_check<2:return
            last_resource_check=time.monotonic()
            rss,_=process_tree_rss(os.getpid());peak=max(peak,rss)
            need(rss<c['limits']['tree_rss_bytes'],'Process tree RSS cap reached')
            need(output_bytes()<c['limits']['output_bytes'],'Output cap reached')
            need(shutil.disk_usage(ROOT).free>=c['limits']['free_disk_min'],'Disk reserve reached')
            atomic(LOG/'usage.json',dict(at=now(),elapsed_seconds=time.monotonic()-start,peak_tree_rss_bytes=peak,deadline_epoch=deadline))
        def execute(name,environment,script,args):
            nonlocal orphan_stop_unconfirmed
            guard();verify(full=False);cid=uuid.uuid4().hex
            cmd=[str(ROOT/f'runtime/envs/{environment}/Scripts/python.exe'),'-B',script,*args]
            append(LOG/'children.jsonl',dict(at=now(),event='intent',id=cid,kind=name,command=cmd))
            env=os.environ.copy();env['P56_V51_TOKEN']=token;env['P56_V51_CHILD_TOKEN']=token
            state=dict(token=token,run_token=token,deadline_epoch=deadline,controller_pid=os.getpid(),child_pid=None,call_id=cid)
            status(name,**state)
            with (LOG/(name+'.stdout.log')).open('xb') as out,(LOG/(name+'.stderr.log')).open('xb') as err:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=err,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
                state['child_pid']=child.pid;status(name,**state)
                try:
                    while child.poll() is None:guard();time.sleep(2)
                except BaseException:
                    from scripts.pipeline import pid_alive
                    try:
                        _,tree_pids=process_tree_rss(child.pid)
                        stopped=subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW,check=False)
                        child.wait(timeout=30)
                        survivors=[pid for pid in tree_pids if pid_alive(pid)]
                        orphan_stop_unconfirmed=bool(survivors) or stopped.returncode!=0
                        append(LOG/'termination_evidence.jsonl',dict(at=now(),child_pid=child.pid,taskkill_returncode=stopped.returncode,survivors=survivors,confirmed=not orphan_stop_unconfirmed))
                    except BaseException as termination_error:
                        orphan_stop_unconfirmed=True
                        append(LOG/'termination_evidence.jsonl',dict(at=now(),child_pid=child.pid,error=repr(termination_error),confirmed=False))
                    raise
                code=child.returncode
            append(LOG/'children.jsonl',dict(at=now(),event='returned',id=cid,kind=name,exit_code=code))
            need(code==0,'Stage failure preserved '+name)
        try:
            status('starting',token=token,run_token=token,deadline_epoch=deadline,controller_pid=os.getpid())
            handoff('P5-P6联合候选选择v51单次有界任务已启动，50案例×7组；全量特征复验与统一官方评价后停止，不自动扩大。全部任务总进度45%、实验总进度56.25%。实际状态runtime/pipeline/p56_joint_v51.json；无周期任务。')
            execute('features','vision','scripts/p56_joint_v51.py',['features'])
            execute('feature_review','vision','scripts/p56_joint_v51.py',['feature-review'])
            for mode in ('score','review','cold','metrics'):
                execute('official_'+mode,'grasp_eval','scripts/p56_official_v51.py',['--config',rel(CONFIG),'--mode',mode])
            guard();verify(full=True);guard()
            finish(guard)
        except BaseException as exc:
            append(LOG/'controller_failures.jsonl',dict(at=now(),error=repr(exc),traceback=traceback.format_exc()))
            status('orphan_stop_unconfirmed' if orphan_stop_unconfirmed else 'failure_preserved_stopped',error=repr(exc),elapsed_seconds=time.monotonic()-start,child_calls=ledger(LOG/'children.jsonl'),feature_calls=ledger(LOG/'feature_calls.jsonl'),neural_calls=ledger(LOG/'neural_calls.jsonl'))
            handoff('P5-P6联合候选选择v51发生需要处理的失败，证据及未知调用保留；'+('子树退出尚未确认，须先核孤儿进程。' if orphan_stop_unconfirmed else '本次运行已停止。')+'不自动重跑或扩预算。全部任务总进度45%、实验总进度56.25%。详见runtime/pipeline/p56_joint_v51.json及logs/joint_validation_v51。')
            raise

def finish(guard):
    # The official component owns metric calculation; this stage only reports immutable results.
    from scripts.p56_official_v51 import load_sealed,event_ledger
    guard();metrics=load_sealed(OUT/'official/metrics.json')
    review=load_sealed(OUT/'official/review.json')
    need(review['passed'] and metrics['full_review']==identity(OUT/'official/review.json'),'Missing final official proof')
    need(metrics['config']==identity(CONFIG),'Metrics config binding differs')
    official_ledger=event_ledger(LOG/'official/calls.jsonl')
    need(metrics['ledger']==official_ledger and not official_ledger['unknown'] and official_ledger['returned_by_kind']['eval_grasp']==357,'Actual official calls not complete')
    for name,expected in review['checked'].items():
        guard();need(identity(ROOT/name)==expected,'Final reviewed artifact changed '+name)
    for group in GROUPS:
        cold=load_sealed(OUT/'official/cold/scene_0002_0064_clean'/(group+'.json'))
        production=load_sealed(OUT/'official/results/scene_0002_0064_clean'/(group+'.json'))
        need(cold['official']==production['official'] and cold['selected']==production['selected'],'Cold output mismatch')
    fr=read(LOG/'feature_review.json');need(fr['passed'] and fr['cases']==50,'Feature replay incomplete')
    for record in fr['records']:
        folder=OUT/'cases'/record['key']
        need(identity(folder/'ranking.json')==record['ranking'] and identity(folder/'uncertainty_inputs.npz')==record['uncertainty'],'Replayed feature mutated')
    children=ledger(LOG/'children.jsonl')
    need(children['intents']==6 and children['known_returns']==6 and not children['unknown'],'Incomplete controller stage returns')
    proofs={}
    for p in OUT.rglob('*'):
        if p.is_file():guard();proofs[rel(p)]=identity(p)
    neural=ledger(LOG/'neural_calls.jsonl');features_done=ledger(LOG/'feature_calls.jsonl')
    need(neural['intents']==100 and neural['known_returns']==100 and not neural['unknown'],'Incomplete actual head-call ledger')
    need(features_done['intents']==100 and features_done['known_returns']==100 and not features_done['unknown'],'Incomplete feature ledger')
    result=dict(at=now(),engineering_passed=True,all_tasks_percent=45,experiment_percent=56.25,
        P5_original_scientific_passed=False,P6_original_scientific_passed=False,
        bounded_joint_validation=metrics,artifacts=proofs,neural_calls=neural,feature_calls=features_done,
        children=ledger(LOG/'children.jsonl'),usage=read(LOG/'usage.json'),expansion_started=False,
        expansion_requires_user_confirmation=True,scope='Fixed candidate exposed development screening only')
    guard(True);result['usage']=read(LOG/'usage.json')
    lines=['P5-P6下游抓取候选选择联合验证 v51',now(),'',
        '全部任务总进度45%；实验总进度56.25%。50开发案例、同一20候选池、7组实验已停止；不自动扩大。',
        '旧P5与P6科学采用未通过结论保持。本轮是新用途的有限开发验证，不是独立测试或机器人结果。',
        '新用途扩大建议条件通过：'+str(metrics['proposal_expand'])+'；即使通过仍须用户确认。','',
        '四退化场景等权AP（百分点）']
    for name,ap in metrics['mean_degraded_AP'].items():lines.append(f'{name}: {ap*100:.6f}')
    lines.extend(['','各模块贡献及交互（百分点）'])
    for name,difference in metrics['contrasts'].items():lines.append(f'{name}: {difference*100:+.6f}')
    lines.extend(['','已冻结科学条件：'+json.dumps(metrics['gates'],ensure_ascii=False),
        '条件几何关联：'+json.dumps(metrics['conditional_geometry_association'],ensure_ascii=False),
        '实际eval_grasp返回：'+str(official_ledger['returned_by_kind']['eval_grasp']),
        '实际get_grasp_score返回：'+str(official_ledger['returned_by_kind']['get_grasp_score']),
        '实际P6神经前向返回：'+str(neural['known_returns'])+'（生产与逐案例重放分列于账本）；新训练/RGB分割前向/抓取生成均0。',
        '详细AP、Top-K、分场景/条件、候选失败/未知及统计口径见results/joint_validation_v51/official/metrics.json与各case文件。',
        '不把AP交互项当机制因果证明，不把未校准风险代理当安全概率。',
        '实际调用、完整复验与资源记录见results/joint_validation_v51/closeout.json。'])
    text='\n'.join(lines)+'\n'
    need(not REPORT.exists(),'Existing report retained');REPORT.write_text(text,encoding='utf-8');guard(True)
    result['usage']=read(LOG/'usage.json');result['report']=identity(REPORT);once(OUT/'closeout.json',result)
    status('completed_stopped',report=rel(REPORT),closeout=rel(OUT/'closeout.json'))
    handoff('P5-P6联合候选选择v51已完成完整实际范围、官方评价、独立只读与冷重入并停止。旧P5/P6科学失败不改，扩大须用户确认。全部任务总进度45%、实验总进度56.25%。结果docs/P5-P6联合候选选择验证结果_v51.txt；状态runtime/pipeline/p56_joint_v51.json。')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['freeze','verify','deliver','features','feature-review']);args=a.parse_args()
    if args.command=='freeze':freeze()
    elif args.command=='verify':verify(full=True);print('Frozen reentry passed')
    elif args.command=='deliver':run()
    elif args.command=='features':features(False)
    elif args.command=='feature-review':features(True)
