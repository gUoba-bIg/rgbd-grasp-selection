"""Synthetic operational checks only; no production images, metrics or training."""
import copy
import json
import sys
import time
import traceback
import uuid
import subprocess
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts import p6_local_correspondence_v50 as runner


def check_gate_formulas():
    import numpy as np
    from scripts.p6_joint_evaluate_v45 import aggregate_warp
    from scripts.p6_mixture_v43 import close_numeric
    from src.p6_fixed_mixture_v49 import candidate_metrics
    scenes=np.repeat([3,8,12,17],108);y=np.arange(432)%9
    prior=np.full(81,.001/72);prior[:9]=.999/9
    priors=np.broadcast_to(prior,(432,81)).copy()
    p=priors.copy()
    for i,target in enumerate(y):
        p[i]*=.02/(1-prior[target]);p[i,target]=.98
    for values in (priors,p):
        primary=candidate_metrics(values,priors,y,scenes)
        reference=runner.reference_candidate(values,priors,y,scenes)
        close_numeric(primary,reference)
        rows=[dict(key='synthetic/'+str(i),split='calibration',scene=int(scene),
                   warp_method=runner.warp_metric(v,t),warp_prior=runner.warp_metric(prior,t))
              for i,(v,t,scene) in enumerate(zip(values,y,np.repeat([30,31,34,38],108)))]
        _,gates=aggregate_warp(rows,split='calibration')
        assert gates==primary['gates']
    assert not candidate_metrics(priors,priors,y,scenes)['gates']['warp_nll_gain']
    assert all(candidate_metrics(p,priors,y,scenes)['gates'].values())
    return p,y,scenes,priors


def check_auxiliary_interface(folder):
    import numpy as np
    from scripts.p6_mixture_v43 import close_numeric
    p,y,scenes,priors=check_gate_formulas()
    root=folder/('aux_'+uuid.uuid4().hex);out=root/'out';out.mkdir(parents=True)
    rows=[dict(key='toy/'+str(i),scene=int(s),split='fit') for i,s in enumerate(scenes)]
    allrows=rows+[dict(key='toy/train/'+str(i),scene=4,split='fit') for i in range(864)]
    ally=np.concatenate([y,np.arange(864)%9]);allp=np.concatenate([p,np.tile(priors[0],(864,1))])
    head=dict(empirical_prior=priors[0].tolist())
    selection=(np.zeros((432,81)),priors,y,scenes,rows,head,{},dict(scope='synthetic only'))
    def read_fixture(path):
        path=Path(path)
        if path==runner.PARENT/'fold_0.json':return {'holdout_indices':list(range(432))}
        if path==out/'head.json':return head
        raise AssertionError('Synthetic auxiliary attempted unlisted read '+str(path))
    with patch.object(runner,'OUT',out),patch.object(runner,'read',side_effect=read_fixture), \
         patch.object(runner,'identity',return_value={'synthetic':True}), \
         patch.object(runner,'selection_data',return_value=selection):
        result=runner.auxiliary_result(allp,ally,allrows)
        reference=runner.auxiliary_result(allp,ally,allrows,reference=True)
        close_numeric(result,reference)
        assert result['supported'] and len(result['gates'])==8
        same=np.tile(priors[0],(1296,1))
        negative=runner.auxiliary_result(same,ally,allrows)
        assert not negative['supported'] and not negative['gates']['nll_gain_over_frozen_same_fold_head']


def check_plan_branches(folder):
    """Scheduler/closeout mocks only, never represented as training validation."""
    first=['fit_data','fit_data_review','train','train_review','auxiliary_review']
    cal6=['calibration_data','calibration_data_review','warp_calibration','warp_review']
    cal17=['calibration','calibration_review']
    evaluation=['evaluation_data','evaluation_data_review','evaluation','evaluation_review']
    for branch in ('aux_failure','cal6_failure','cal17_failure','eval_failure','full_pass'):
        auxiliary=branch!='aux_failure';six=branch not in ('aux_failure','cal6_failure')
        fullcal=branch in ('eval_failure','full_pass');success=branch=='full_pass'
        expected=first+(cal6 if auxiliary else [])+(cal17 if six else [])+(evaluation if fullcal else [])
        root=folder/(branch+'_'+uuid.uuid4().hex);out=root/'out';log=root/'logs'
        out.mkdir(parents=True);log.mkdir();report=root/'report.txt';state=root/'state.json'
        runner.atomic(state,dict(synthetic_fixture=True))
        fixture={out/'auxiliary.json':dict(supported=auxiliary,gates={'synthetic_gate':auxiliary})}
        if auxiliary:fixture[out/'warp_calibration.json']=dict(passed=six,gates={'synthetic_gate':six})
        if six:fixture[out/'calibration.json']=dict(passed=fullcal,gates={'synthetic_gate':fullcal})
        if fullcal:fixture[out/'evaluation.json']=dict(passed=success,gates={'synthetic_gate':success})
        proof=dict(stages=expected,auxiliary_supported=auxiliary,all17_executed=six,evaluation_executed=fullcal,
                   development_scientific_passed=success,calls={},neural_calls={},
                   gates_by_split={s:{g:None for g in runner.GATES} for s in ('calibration','evaluation')})
        original_read=runner.read
        def fixture_read(path):
            path=Path(path)
            if path in fixture:return copy.deepcopy(fixture[path])
            if path==log/'cold.json':return dict(passed=True,proof=copy.deepcopy(proof))
            assert path.resolve().is_relative_to(root.resolve()),'Unlisted synthetic file read '+str(path)
            if path.name in ('model.json','head.json','warp_calibration.json','calibration.json','evaluation.json'):
                raise AssertionError('Unexecuted scientific artifact accessed '+path.name)
            return original_read(path)
        calls=[]
        with patch.object(runner,'OUT',out),patch.object(runner,'LOG',log),patch.object(runner,'STATE',state), \
             patch.object(runner,'REPORT',report),patch.object(runner,'read',side_effect=fixture_read), \
             patch.object(runner,'verify_executed',return_value=proof),patch.object(runner,'protected_check'), \
             patch.object(runner,'status'),patch.object(runner,'supervise'),patch.object(runner,'handoff'):
            runner.run_plan(calls.append)
            assert calls==expected+['cold']
            assert runner.executed_stages()==expected
            runner.atomic(log/'cold.json',dict(passed=True,proof=proof))
            runner.finish(dict(controller_wall_seconds=7200,preparation_seconds=0,output_limit=2*1024**3,
                               budget_scope='synthetic scheduler mock only'),lambda:None)
        done=original_read(out/'closeout.json')
        assert done['stopped'] and done['P6_development_scope_passed']==success
        assert [done['all_tasks_percent'],done['experiment_percent']]==([47,58.75] if success else [45,56.25])
        assert report.exists()


def check_unknown_calls(folder):
    root=folder/('unknown_'+uuid.uuid4().hex);root.mkdir()
    def failed():raise RuntimeError('intent has no return')
    with patch.object(runner,'LOG',root),patch.object(runner,'LIMITS',{'synthetic_only':1}),patch.object(runner,'_seen',None):
        try:runner.call('synthetic_only','one',failed)
        except RuntimeError:pass
        else:raise AssertionError('Exception lost')
        before=(root/'calls.jsonl').read_bytes();account=runner.ledger()
        assert account['intents']==1 and account['known_returns']==0 and len(account['unknown'])==1
        try:runner.call('synthetic_only','one',lambda:True)
        except (ValueError,RuntimeError):pass
        else:raise AssertionError('Duplicate/unknown call repeated')
        assert (root/'calls.jsonl').read_bytes()==before
        callback=runner.neural_callback('synthetic_prediction')
        callback(dict(stage='forward',event='intent',id=0,cases=1,start=0))
        assert len(runner.ledger(root/'neural_calls.jsonl')['unknown'])==1
        callback(dict(stage='forward',event='returned',id=0,cases=1,start=0))
        assert runner.ledger(root/'neural_calls.jsonl')['counts']=={'predict_forward':1}


def main():
    started=time.monotonic();checks=[];passed=False;failure_trace=None
    record=runner.LOG/'preparation'/('integration_'+uuid.uuid4().hex+'.json');record.parent.mkdir(parents=True,exist_ok=True)
    try:
        for name in runner.SOURCES:compile((ROOT/name).read_text(encoding='utf-8-sig'),name,'exec')
        checks.append('All final new sources compile')
        check_unknown_calls(record.parent)
        checks.append('Unreturned scientific call remains unknown, cannot be repeated; actual neural intent/return separate')
        with runner.own_lock():
            blocked=False
            try:
                with runner.own_lock():pass
            except OSError:blocked=True
            assert blocked
        with runner.project_lock(ROOT):pass
        checks.append('Duplicate controller lock rejects and project lock available')
        child=subprocess.Popen([str(ROOT/'runtime/envs/vision/Scripts/python.exe'),'-X','utf8','-B','-c','import time;time.sleep(3)'],cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            time.sleep(.75);rss,members=runner.process_tree_rss(child.pid)
            assert child.pid in members and len(members)>=2 and rss>0
        finally:child.wait(timeout=10)
        assert runner.process_tree_rss(child.pid)==(0,[])
        checks.append('Actual venv-owned child tree RSS observed and exited processes excluded')
        assert runner.sha(ROOT/'runtime/pipeline/state.json')==runner.ORIGINAL_STATE
        checks.append('Original historical pipeline state unchanged')
        check_auxiliary_interface(record.parent)
        checks.append('Actual auxiliary interface, independent formula and original cal6 aggregate agree; unchanged baseline cannot pass gain')
        check_plan_branches(record.parent)
        checks.append('All five scheduler/closeout branches mock-tested; no future artifact access; null gates and progress stop boundaries preserved')
        modules=['tests.test_p6_local_correspondence_v50','tests.test_p6_local_features_v50',
                 'tests.test_p6_local_training_review_v50','tests.test_p6_local_correspondence_independent_v50',
                 'tests.test_p6_local_runner_data_v50']
        command=[str(ROOT/'runtime/envs/vision/Scripts/python.exe'),'-X','utf8','-B','-m','unittest',*modules,'-v']
        result=subprocess.run(command,cwd=ROOT,capture_output=True,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
        record.with_suffix('.stdout.log').write_bytes(result.stdout);record.with_suffix('.stderr.log').write_bytes(result.stderr)
        assert result.returncode==0,'Synthetic suite failed; stdout/stderr retained'
        checks.append('All five core/data/review/independent/runner-data synthetic suites pass; no production fit or images')
        passed=True
    except BaseException:
        failure_trace=traceback.format_exc();raise
    finally:
        proof=dict(at=runner.now(),passed=passed,seconds=time.monotonic()-started,checks=checks,failure_trace=failure_trace,
            scope='Synthetic numerical, real empty-lock/process, and mocked scheduler/closeout checks; not production training acceptance',
            real_image_decodes=0,real_neural_forwards=0,production_optimizer_updates=0,real_scientific_metric_calls=0,
            source_sha256={name:runner.sha(ROOT/name) for name in runner.SOURCES if (ROOT/name).exists()})
        runner.once(record,proof)
        if passed:
            target=runner.LOG/'internal_checks.json';runner.backup(target,'pre_freeze_checks');runner.atomic(target,proof)
        print(json.dumps(proof))

if __name__=='__main__':main()
