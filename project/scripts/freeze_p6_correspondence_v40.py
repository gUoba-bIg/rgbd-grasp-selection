"""One-shot root freeze for the explicitly authorized finite P6 background job."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts.p6_background_v40 import LOG,OUT,CONF,read,stat,sha,atomic,need,now,status,supervision,handoff
def main():
    from scripts.pipeline import project_lock
    with project_lock(ROOT):
        need(not CONF.exists() and not (LOG/'authorization.json').exists() and not (LOG/'controller_freeze.json').exists(),'Existing freeze must be verified, never replaced')
        d=read(LOG/'data_audit.json');need(d['passed'] and d['raw_file_count']==320 and d['planned_case_count']==720,'Selected input audit differs')
        tests=read(LOG/'component_tests.json');need(tests['passed'],'Component self-tests required')
        need(bool(tests['source_sha256']),'Component test source identities required')
        for path,value in tests['source_sha256'].items():need(sha(ROOT/path)==value,'Source changed after component tests '+path)
        review=read(LOG/'independent_review.json');need(review['passed'],'Independent review required')
        for path,value in review['source_sha256'].items():need(sha(ROOT/path)==value,'Source changed after independent review '+path)
        protected=dict(d['protected']);proposal=read(LOG/'proposal.json')
        for path,value in protected.items():need(stat(ROOT/path)==value,'Historical accepted source/input changed '+path)
        extras=['scripts/p6_background_v40.py','scripts/freeze_p6_correspondence_v40.py','scripts/p6_correspondence_v40.py',
            'scripts/p5_fusion_delivery_v38.py','scripts/pipeline.py','docs/P6对应可识别性后台协议_v40.txt',
            'docs/原方向恢复误差模型失败分析_v37.txt','docs/P5完整评价与原方案取舍结果_v39.txt',
            'logs/P5/completion_v39/root_final_review.json','logs/P5/completion_v39/final_delivery_index.json',
            'logs/P6/correspondence_v40/data_audit.json','logs/P6/correspondence_v40/data_audit_usage.json',
            'logs/P6/correspondence_v40/proposal.json','logs/P6/correspondence_v40/component_tests.json',
            'logs/P6/correspondence_v40/preparation_usage.json','logs/P6/correspondence_v40/independent_review.json',
            'runtime/pipeline/state.json']+proposal.get('inherited_sources',[])
        extras += [p.relative_to(ROOT).as_posix() for p in (ROOT/'src').glob('*correspondence*v40*.py')]
        for path in sorted(set(extras)):
            identity=stat(ROOT/path)
            if path in protected:need(protected[path]==identity,'Accepted dependency changed '+path)
            protected[path]=identity
        need(protected['runtime/pipeline/state.json']['sha256']=='b48eadde3c1f7ea400c48dd206f707ecf804d54f6f018a013e9d662dcc4b2b76','Original pipeline state differs')
        preparation_seconds=float(read(LOG/'data_audit_usage.json')['seconds'])+float(read(LOG/'preparation_usage.json')['seconds'])
        need(0<=preparation_seconds<1800,'Preparation budget exhausted')
        atomic(LOG/'authorization.json',dict(at=now(),frozen=True,proposal_sha256=sha(LOG/'proposal.json'),protected=protected,
            scope='P6 global integer correspondence identifiability only',user_authorization='2026-11-01: start P6 if feasible; finite local background execution without repeated status polling',
            P5_failure_unchanged=True,P6_full_passed=False,uvz_calibrated=False))
        control_protected=dict(protected);control_protected['logs/P6/correspondence_v40/authorization.json']=stat(LOG/'authorization.json')
        c=dict(at=now(),version='v40',controller_wall_seconds=1800,preparation_seconds=preparation_seconds,
            python='runtime/envs/vision/Scripts/python.exe',protected=control_protected,
            stages=[dict(name=name,done=done) for name,done in [
                ('generate','results/P6/correspondence_v40/generated.json'),
                ('fit','results/P6/correspondence_v40/model.json'),
                ('evaluate','results/P6/correspondence_v40/result.json'),
                ('review','logs/P6/correspondence_v40/review.json'),
                ('cold','logs/P6/correspondence_v40/cold.json')]],
            all_tasks_percent=45,experiment_percent=56.25,stop_after='P6_global_correspondence_v40')
        atomic(CONF,c);atomic(LOG/'controller_freeze.json',dict(at=now(),config_sha256=sha(CONF),protected_files=len(control_protected)))
        from scripts.p6_background_v40 import verify
        verify()
        status('frozen_ready',protected_files=len(control_protected),preparation_seconds=preparation_seconds)
        supervision('P6_v40_frozen_ready',current_runner='scripts/p6_background_v40.py',
            waiting_for_user_next_part_confirmation=False,active_internal_repair='global_correspondence_identifiability',
            current_execution_evidence='logs/P6/correspondence_v40/controller_freeze.json',
            user_authorization='2026-11-01用户授权满足前置后直接后台运行P6，使用本地有限控制器自动验收/报告/停机，不使用LLM实时轮询')
        handoff('P6 v40全局对应可识别性协议已冻结：80旧暴露帧×9平移720案例，CPU观测对应估计与一次低容量拟合，独立复验/真实冷重入/本地报告后停止。入口scripts/p6_background_v40.py deliver，状态runtime/pipeline/p6_correspondence_v40.json；先核实际进程/锁，禁止重复启动。累计1800秒含准备，0神经/抓取/AP调用。全部任务总进度45%、实验总进度56.25%；完整P6与uvz校准未完成，不预记剩余2点。无LLM实时轮询、无周期任务、不进P7/P10。','freeze')
        print(json.dumps(dict(frozen=True,protected_files=len(control_protected),preparation_seconds=preparation_seconds)))
if __name__=='__main__':main()
