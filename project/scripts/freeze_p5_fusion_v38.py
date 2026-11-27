"""Root-only one-shot v38 freeze after independent review and bounded preflight."""
import sys, json, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts.p5_fusion_delivery_v38 import LOG,OUT,CONF,read,sha,stat,atomic,need,now,status

def main():
    from scripts.pipeline import project_lock
    from scripts.restore_direction_v37 import verify as verify_previous
    with project_lock(ROOT):
        need(not (LOG/'freeze.json').exists() and not (LOG/'authorization.json').exists() and not CONF.exists(),'Freeze already exists; verify/recover without replacing it')
        previous=verify_previous()
        proposal=read(LOG/'proposal.json');preflight=read(OUT/'preflight.json')
        need(preflight['passed'] and preflight['model_sha256']==sha(ROOT/'src/modal_fusion_v38.py'),'Model preflight not current')
        review=read(LOG/'independent_review.json');need(review['passed'],'Independent pre-freeze review missing')
        for path,digest in review['source_sha256'].items():need(sha(ROOT/path)==digest,'Code changed after independent review '+path)
        need(proposal['training']['total_optimizer_intents']==1600,'Unexpected training scope')
        need(proposal['resources']['max_forward_intents']==20000 and proposal['resources']['max_official_AP_intents']==290,'Unexpected call budget')
        calls=preflight['accounting']
        need(calls['forward']['intents']==5 and calls['forward']['known_returns']==5 and not calls['forward']['unknown_ids'],'Preparation calls not fully observed')
        need(calls['optimizer']['intents']==0 and calls['official_AP']['intents']==0,'Preparation performed unauthorized research steps')
        p4_sources=['logs/P4/full_baseline_v22r2/evaluation/input_binding.json','logs/P4/full_baseline_v22r2/final_readonly/before.json']
        p4_bound={}
        for name in p4_sources:
            for path,identity in read(ROOT/name).items():
                if path in p4_bound:need(p4_bound[path]==identity,'Old P4 manifests disagree '+path)
                p4_bound[path]=identity
        for path in proposal['parent_reference_files']:
            if path in p4_bound:need(stat(ROOT/path)==p4_bound[path],'Accepted P4 parent/reference was modified '+path)
            elif path==proposal['parent'] or '/exports/rgb/' in path or '/evaluation/official/' in path:
                raise RuntimeError('RGB reference lacks historical accepted identity: '+path)
        protected=dict(previous['protected'])
        extra=proposal['inherited_sources']+proposal['parent_reference_files']+p4_sources+[
            'configs/restore_direction_v37.json','results/Restore/v37/closeout.json','logs/Restore/v37/root_final_review.json',
            'scripts/freeze_p5_fusion_v38.py','scripts/p5_fusion_delivery_v38.py','scripts/restore_p5_v38.py','src/modal_fusion_v38.py',
            'docs/P5模态融合整体修复协议_v38.txt','docs/P5整体恢复进度口径与剩余交付_v38.txt',
            'logs/P5/fusion_v38/proposal.json','logs/P5/fusion_v38/independent_review.json',
            'logs/P5/fusion_v38/progress_scope_audit.json','logs/P5/fusion_v38/kickoff_audit.json',
            'logs/P5/fusion_v38/controller_check.json','results/P5/fusion_v38/preflight.json',
            'logs/P5/fusion_v38/preflight_calls.jsonl']
        extra += [p.relative_to(ROOT).as_posix() for p in LOG.glob('preflight_usage_*.json')]
        for path in sorted(set(extra)):
            current=stat(ROOT/path)
            if path in protected:need(current==protected[path],'Parent identity mismatch '+path)
            protected[path]=current
        # All train/screen/full raw identities must already be protected and match each manifest.
        data_root=read(ROOT/proposal['inherited_config'])['data_root']
        for rec in proposal['train_records']+proposal['full_records']:
            for kind,path in rec['files'].items():
                rel=(Path(data_root)/path).as_posix()
                need(rel in protected and protected[rel]['sha256']==rec['sha256'][kind],'Unbound raw input '+rel)
        auth=dict(frozen=True,at=now(),part='P5_v38',proposal_sha256=sha(LOG/'proposal.json'),protected=protected)
        atomic(LOG/'authorization.json',auth)
        protected_for_controller=dict(protected)
        protected_for_controller['logs/P5/fusion_v38/authorization.json']=stat(LOG/'authorization.json')
        vision='runtime/envs/vision/Scripts/python.exe';cpu='runtime/envs/p4_eval_v22/Scripts/python.exe'
        stages=[]
        for name,mode,python,done,conditional in [
            ('train','train',vision,'results/P5/fusion_v38/training_complete.json',False),
            ('export_screen','export-screen',vision,'results/P5/fusion_v38/screen/exports_complete.json',False),
            ('evaluate_screen','evaluate-screen',cpu,'results/P5/fusion_v38/screen/decision.json',False),
            ('export_full','export-full',vision,'results/P5/fusion_v38/full/exports_complete.json',True),
            ('evaluate_full','evaluate-full',cpu,'results/P5/fusion_v38/full/decision.json',True),
            ('review','review',cpu,'logs/P5/fusion_v38/review.json',False),
            ('cold','cold',cpu,'logs/P5/fusion_v38/cold.json',False)]:
            stage=dict(name=name,script='scripts/restore_p5_v38.py',python=python,args=[mode],done=done)
            if conditional:stage['condition']=dict(path='results/P5/fusion_v38/screen/decision.json',keys=['eligible_full_evaluation'],equals=True)
            stages.append(stage)
        c=dict(version='v38',at=now(),controller_wall_seconds=10800,
            preflight_compute_seconds=sum(read(p)['seconds'] for p in LOG.glob('preflight_usage_*.json')),
            protected=protected_for_controller,stages=stages,reports=dict(screen='results/P5/fusion_v38/screen/decision.json',
                full='results/P5/fusion_v38/full/decision.json',pass_field='scientific_adoption_supported',
                reviews=dict(independent='logs/P5/fusion_v38/review.json',cold='logs/P5/fusion_v38/cold.json')),
            current_progress=dict(all_tasks_percent=41,experiment_percent=51.25),stop_after='P5_v38')
        atomic(CONF,c);atomic(LOG/'freeze.json',dict(at=now(),config_sha256=sha(CONF),protected_files=len(protected_for_controller),independent_review_sha256=sha(LOG/'independent_review.json')))
        from scripts.p5_fusion_delivery_v38 import verify
        verify()
        handoff=ROOT/'docs/实验自动续接说明.txt'
        shutil.copy2(handoff,LOG/'handoff_before_freeze.txt')
        handoff.write_text(f'最新：{now()} 用户授权的P5模态融合整体修复v38已通过5次准备前向、独立设计/代码审查并冻结。两臂各800更新，20帧有效性准入通过则自动完整160帧五条件/消融评价，独立只读和真实冷重入后停止；不进P6/P10或周期任务。入口scripts/p5_fusion_delivery_v38.py deliver，状态runtime/pipeline/p5_fusion_delivery_v38.json。先核实际进程和项目锁，禁止重复启动。全部任务总进度41%，实验总进度51.25%；固定恢复范围权重，尚未预记本次完成分。\n\n历史：\n'+handoff.read_text(encoding='utf-8-sig'),encoding='utf-8')
        metadata=ROOT/'runtime/pipeline/part_level_supervision.json';v=read(metadata)
        v.update(recorded_at=now(),current_execution_state='P5_v38_frozen_ready',current_runner='scripts/p5_fusion_delivery_v38.py',remaining_research_updates_cap=1600,waiting_for_user_next_part_confirmation=False)
        atomic(metadata,v);status('frozen_ready',protected_files=len(protected_for_controller))
        print(json.dumps(dict(frozen=True,protected_files=len(protected_for_controller),stages=[s['name'] for s in stages])))

if __name__=='__main__':main()
