"""Freeze one supplementary complete evaluation; never rewrites v38 eligibility."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts.p5_completion_delivery_v39 import LOG,OUT,CONF,read,sha,stat,atomic,need,now,status

def main():
    from scripts.pipeline import project_lock
    from scripts.p5_fusion_delivery_v38 import verify as verify_old,deliver as verify_old_completion
    with project_lock(ROOT):
        need(not CONF.exists() and not (LOG/'authorization.json').exists() and not (LOG/'freeze.json').exists(),'Freeze exists; never overwrite')
        old=verify_old();verify_old_completion()
        prepare=read(LOG/'prepare.json');need(prepare['passed'],'Preparation failed')
        need(prepare['new_forward']==prepare['new_optimizer']==prepare['new_official_AP']==0,'Preparation scope differs')
        need(read(LOG/'adapter_check.json')['passed'],'Namespace adaptation audit failed')
        independent=read(LOG/'independent_review.json');need(independent['passed'],'Independent review required')
        for path,digest in independent['source_sha256'].items():need(sha(ROOT/path)==digest,'Changed after independent review '+path)
        prior=read(ROOT/'logs/P5/fusion_v38/root_final_review.json')
        need(prior['passed'] and not prior['full160_evaluation_completed'],'Unexpected old completed range')
        need(read(ROOT/'results/P5/fusion_v38/screen/decision.json')['eligible_full_evaluation'] is False,'Old screen state must stay false')
        protected=dict(old['protected'])
        for path,value in read(ROOT/'logs/P5/fusion_v38/review.json')['reviewed_outputs'].items():
            need(stat(ROOT/path)==value,'Accepted v38 artifact changed '+path);protected[path]=value
        extras=['configs/p5_fusion_v38.json','runtime/pipeline/p5_fusion_delivery_v38.json',
            'docs/P5模态融合整体修复结果_v38.txt','docs/P5模态融合有限准入结果解释_v38.txt',
            'scripts/complete_p5_v39.py','scripts/p5_completion_delivery_v39.py','scripts/freeze_p5_completion_v39.py',
            'scripts/summarize_p5_completion_v39.py',
            'docs/P5完整评价协议_v39.txt','docs/P5完整评价协议审核意见_v39.txt','docs/原方案可行性独立审查_v39.txt',
            'logs/P5/completion_v39/proposal.json','logs/P5/completion_v39/prepare.json',
            'logs/P5/completion_v39/adapter_check.json','logs/P5/completion_v39/implementation_ready.json',
            'logs/P5/completion_v39/kickoff_audit.json','logs/P5/completion_v39/independent_review.json']
        extras += [p.relative_to(ROOT).as_posix() for directory in [ROOT/'logs/P5/fusion_v38',ROOT/'results/P5/fusion_v38',ROOT/'experiments/p5_fusion_v38'] for p in directory.rglob('*') if p.is_file()]
        extras += [p.relative_to(ROOT).as_posix() for p in LOG.glob('preflight_usage_*.json')]
        for path in sorted(set(extras)):
            value=stat(ROOT/path)
            if path in protected:need(value==protected[path],'Parent changed '+path)
            protected[path]=value
        auth=dict(frozen=True,at=now(),new_full_evaluation_explicitly_authorized=True,
            parent_screen_eligible_unchanged=False,proposal_sha256=sha(LOG/'proposal.json'),protected=protected)
        atomic(LOG/'authorization.json',auth)
        protected_for_controller=dict(protected)
        protected_for_controller['logs/P5/completion_v39/authorization.json']=stat(LOG/'authorization.json')
        vision='runtime/envs/vision/Scripts/python.exe';cpu='runtime/envs/p4_eval_v22/Scripts/python.exe'
        c=dict(version='v39',at=now(),controller_wall_seconds=3600,
            preflight_compute_seconds=sum(read(p)['seconds'] for p in LOG.glob('preflight_usage_*.json')),
            max_new_optimizer_intents=0,max_new_forward_intents=3920,max_new_AP_intents=140,
            parent_recorded_compute_seconds=prior['recorded_compute_seconds'],protected=protected_for_controller,
            current_progress=dict(all_tasks_percent=42,experiment_percent=52.5),stop_after='P5_v39',
            stages=[dict(name=mode,python=python,done=done) for mode,python,done in [
                ('export',vision,'results/P5/completion_v39/full/exports_complete.json'),
                ('evaluate',cpu,'results/P5/completion_v39/full/decision.json'),
                ('review',cpu,'logs/P5/completion_v39/review.json'),
                ('cold',cpu,'logs/P5/completion_v39/cold.json')]])
        atomic(CONF,c);atomic(LOG/'freeze.json',dict(at=now(),config_sha256=sha(CONF),protected_files=len(protected_for_controller)))
        from scripts.p5_completion_delivery_v39 import verify
        verify()
        handoff=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(handoff,LOG/'handoff_before_freeze.txt')
        handoff.write_text(f'最新：{now()}（用户客户端授权日期2026-11-01）P5 v39完整评价新协议已审查冻结：只补v38已有final权重完整160帧，0重训，新增最多3920模型入口/140官方AP，控制链累计上限3600秒。旧screen失败不改写，完整科学门槛不变。入口scripts/p5_completion_delivery_v39.py deliver；先核实际进程和项目锁禁止重复启动。完整只读、真实冷重入及实际取舍后停止，不自动新训练、P6、P10或周期任务。全部任务总进度42%、实验总进度52.5%；原1点不重复，本次完整交付后根核实B+C才计分。\n\n历史：\n'+handoff.read_text(encoding='utf-8-sig'),encoding='utf-8')
        metadata=ROOT/'runtime/pipeline/part_level_supervision.json';shutil.copy2(metadata,LOG/'supervision_before_freeze.json')
        v=read(metadata);v.update(recorded_at=now(),current_authorized_part='P5_complete_existing_fusion_v39',
            current_execution_state='P5_v39_frozen_ready',current_runner='scripts/p5_completion_delivery_v39.py',
            remaining_research_updates_cap=0,waiting_for_user_next_part_confirmation=False,
            current_execution_evidence='logs/P5/completion_v39/freeze.json',stop_after_part='P5_v39',stop_after='P5_v39',
            user_authorization='2026-11-01用户同意先独立审原方案可行性并采取有限行动；根决定补既有完整范围再取舍，0新训',
            active_internal_repair='complete_160_without_retraining',automatic_resume_authorized=False)
        atomic(metadata,v);status('frozen_ready',protected_files=len(protected_for_controller))
        print(json.dumps(dict(frozen=True,protected_files=len(protected_for_controller))))
if __name__=='__main__':main()
