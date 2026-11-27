"""Freeze one authorized selective correspondence candidate; no formal computation."""
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts.p6_selective_delivery_v42 import LOG,OUT,CONF,read,stat,sha,atomic,need,now,status,supervision,handoff

def main():
    started=time.monotonic()
    from scripts.pipeline import project_lock
    with project_lock(ROOT):
        need(not CONF.exists() and not (LOG/'authorization.json').exists(),'Existing freeze must not be replaced')
        from scripts.p6_background_v40 import verify as old_verify,completed_verify as old_completed
        from scripts.p6_preprocessing_v41r1 import check_frozen as geometry_verify
        old=old_verify();old_completed(old);geometry_verify()
        tests=read(LOG/'component_tests.json');review=read(LOG/'independent_review.json');proposal=read(LOG/'proposal.json')
        need(tests['passed'] and review['passed'],'Preparation tests and independent review must pass')
        for record in (tests,review):
            need(bool(record['source_sha256']),'Missing review/test source binding')
            for p,h in record['source_sha256'].items():need(sha(ROOT/p)==h,'Source changed after review/test '+p)
        for p,value in proposal['protected'].items():need(stat(ROOT/p)==value,'Proposal dependency changed '+p)
        protected=dict(old['protected'])
        protected.update(proposal['protected'])
        extras=['scripts/p6_selective_delivery_v42.py','scripts/freeze_p6_selective_v42.py','scripts/p6_selective_v42.py','src/correspondence_v42.py',
            'docs/P6匹配歧义有限修复协议_v42.txt','logs/P6/selective_v42/proposal.json','logs/P6/selective_v42/component_tests.json',
            'logs/P6/selective_v42/preparation_usage.json','logs/P6/selective_v42/independent_review.json',
            'configs/p6_correspondence_v40.json','logs/P6/correspondence_v40/controller_freeze.json','logs/P6/correspondence_v40/data_audit.json',
            'results/P6/correspondence_v40/generated.json','results/P6/correspondence_v40/model.json','results/P6/correspondence_v40/result.json',
            'results/P6/correspondence_v40/closeout.json','logs/P6/correspondence_v40/review.json','logs/P6/correspondence_v40/cold.json',
            'runtime/pipeline/p6_correspondence_v40.json','docs/P6对应可识别性结果_v40.txt',
            'scripts/p6_preprocessing_v41r1.py','configs/p6_preprocessing_v41r1.json','docs/P6输入几何一致性诊断协议_v41r1.txt',
            'results/P6/preprocessing_v41r1/result.json','results/P6/preprocessing_v41r1/closeout.json',
            'logs/P6/preprocessing_v41r1/independent_review.json','logs/P6/preprocessing_v41r1/cold_readonly.json',
            'docs/P6输入几何一致性诊断结果_v41r1.txt','runtime/pipeline/p6_preprocessing_v41r1.json','runtime/pipeline/state.json']
        for p in sorted(set(extras+proposal.get('inherited_sources',[]))):
            value=stat(ROOT/p)
            if p in protected:need(protected[p]==value,'Historical identity differs '+p)
            protected[p]=value
        need(protected['runtime/pipeline/state.json']['sha256']=='b48eadde3c1f7ea400c48dd206f707ecf804d54f6f018a013e9d662dcc4b2b76','Original state changed')
        # Reserve a labelled conservative allowance for root verification/freeze;
        # measured component preparation remains separate, no old v40 run is charged twice.
        preparation_measured=float(read(LOG/'preparation_usage.json')['seconds'])
        need(time.monotonic()-started<10,'Freeze verification exceeded reserved10s')
        preparation_seconds=preparation_measured+10.0
        need(preparation_seconds<1800,'Preparation exhausted cumulative budget')
        atomic(LOG/'authorization.json',dict(at=now(),frozen=True,proposal_sha256=sha(LOG/'proposal.json'),protected=protected,
            scope='One selective local-support correspondence candidate only',user_authorization='2026-11-01: continue after step1 geometry diagnostic; one bounded P6 matching-ambiguity repair',
            P5_failure_unchanged=True,old_P6_failure_unchanged=True,P6_full_passed=False,uvz_calibrated=False))
        control=dict(protected);control['logs/P6/selective_v42/authorization.json']=stat(LOG/'authorization.json')
        c=dict(at=now(),version='v42',controller_wall_seconds=1800,preparation_seconds=preparation_seconds,
            preparation_measured_component_seconds=preparation_measured,root_preflight_conservative_charge_seconds=10.0,
            python='runtime/envs/vision/Scripts/python.exe',protected=control,
            stages=[dict(name=n,done=p) for n,p in [('generate','results/P6/selective_v42/generated.json'),
                ('evaluate','results/P6/selective_v42/result.json'),('review','logs/P6/selective_v42/review.json'),('cold','logs/P6/selective_v42/cold.json')]],
            all_tasks_percent=45,experiment_percent=56.25,stop_after='P6_selective_correspondence_v42')
        atomic(CONF,c);atomic(LOG/'controller_freeze.json',dict(at=now(),config_sha256=sha(CONF),protected_files=len(control)))
        from scripts.p6_selective_delivery_v42 import verify
        verify()
        measured=time.monotonic()-started
        need(measured<=10,'Complete freeze verification exceeded conservative10s charge; do not launch')
        atomic(LOG/'root_freeze_usage.json',dict(at=now(),measured_seconds=measured,conservative_charged_seconds=10.0))
        status('frozen_ready',protected_files=len(control),preparation_seconds_with_conservative_charge=preparation_seconds)
        supervision('P6_v42_frozen_ready',current_runner='scripts/p6_selective_delivery_v42.py',waiting_for_user_next_part_confirmation=False,
            active_internal_repair='single_selective_correspondence_candidate',current_execution_evidence='logs/P6/selective_v42/controller_freeze.json',
            user_authorization='2026-11-01用户继续授权P6匹配歧义一次有限修复，旧门槛不变；复验/冷重入/报告后停止，后台本地控制，无周期任务或LLM实时监测')
        handoff('P6 v42匹配歧义有限修复已冻结：六格局部支持、精确零票/非零票互斥，少于规定空间支持回退原位置但记unknown拒绝。80旧暴露帧×9warp720案例，原global结果与ridge复用0拟合；720局部生成+720独立复算，原全部科学门槛不变。累计1800秒含准备及根预检保守扣款10秒，0新神经/AP。入口scripts/p6_selective_delivery_v42.py deliver；先核进程及锁禁止重复启动。后台验收/冷重入/报告后停止，不调参追通过、不进P7/P10或周期任务。全部任务总进度45%、实验总进度56.25%；完整P6/uvz尚未通过。','freeze')
        print(json.dumps(dict(frozen=True,protected_files=len(control),preparation_seconds=preparation_seconds)))

if __name__=='__main__':main()
