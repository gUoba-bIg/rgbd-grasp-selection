"""Operational preflight and complete synthetic tests; no real-data fit/evaluation."""
import sys, time, uuid, json, subprocess, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from scripts import p6_fixed_mixture_v49 as runner

def check_fixed_branches(folder):
    """Actual selection/replay and closeout for all three auxiliary outcomes."""
    import copy
    import numpy as np
    from unittest.mock import patch
    from tests.test_p6_fixed_mixture_v49 import fixture
    from tests.test_p6_probability_calibration_v46 import base_model
    for branch in ('no_feasible','gap_failure','supported'):
        root=folder/(branch+'_'+uuid.uuid4().hex);out=root/'results';log=root/'logs';out.mkdir(parents=True);log.mkdir()
        report=root/'report.txt';scores,priors,y,scenes=fixture()
        if branch=='no_feasible':scores=np.zeros_like(scores)
        elif branch=='gap_failure':
            p=np.full(81,.001/72);p[:9]=.199/8;p[0]=.8
            scores=np.broadcast_to(np.log(p)-np.log(priors[0]),(432,81)).copy();y=np.zeros(432,dtype=np.int64)
        head=base_model();head.update(fitted_cases=864,empirical_prior=priors[0].tolist(),center=[0.]*16,scale=[1.]*16,coefficients=[1.]+[0.]*15)
        parent=dict(shape={'synthetic_shape':True},depth={'synthetic_depth':True},empirical_prior=(np.ones(81)/81).tolist())
        rows=[dict(key=f'aux/{i}',split='fit',scene=int(scene)) for i,scene in enumerate(scenes)]
        provenance=dict(inputs={},inherited_complete_oof_review={'synthetic_input':True},fixed_fold=0,
            training_scenes=[4,5,9,10,14,16,19,20],auxiliary_scenes=[3,8,12,17],
            auxiliary_unseen_by_head=True,auxiliary_unseen_by_depth_H_or_whole_research=False)
        x=np.zeros((432,81,16));x[:,:,0]=scores
        calrows=[dict(key=f'cal/{i}',split='calibration',scene=int(scene)) for i,scene in enumerate(np.repeat([30,31,34,38],108))]
        arrays=dict(X=x,y=np.full(432,80,dtype=np.int64))
        original_read,original_identity=runner.read,runner.identity
        def scoped_identity(path):
            return original_identity(path) if Path(path).resolve().is_relative_to(root.resolve()) else {'synthetic_input':True}
        def scoped_read(path):
            path=Path(path)
            if path==runner.PARENT/'fold_0.json':return {'head':head}
            if path==ROOT/'results/P6/joint_repair_v45/model.json':return parent
            assert path.resolve().is_relative_to(root.resolve()),'Synthetic fixture attempted real-data read'
            if branch!='supported' and path.name in ('model.json','warp_calibration.json','calibration.json','evaluation.json'):
                raise AssertionError('Rejected selection read an unrun artifact')
            return original_read(path)
        with patch.object(runner,'OUT',out),patch.object(runner,'LOG',log),patch.object(runner,'REPORT',report), \
             patch.object(runner,'selection_data',return_value=(scores,priors,y,scenes,rows,head,parent,provenance)), \
             patch.object(runner,'cache',return_value=(arrays,calrows)),patch.object(runner,'identity',side_effect=scoped_identity), \
             patch.object(runner,'read',side_effect=scoped_read),patch.object(runner,'child_guard'),patch.object(runner,'protected_check'), \
             patch.object(runner,'status'),patch.object(runner,'supervise'),patch.object(runner,'handoff'),patch.object(runner,'_seen',None), \
             patch.object(runner,'rel',side_effect=lambda p:Path(p).name):
            runner.selection();runner.selection_review();selection=original_read(out/'selection.json')
            assert selection['supported']==(branch=='supported')
            if branch=='no_feasible':assert selection['selected'] is None
            elif branch=='gap_failure':assert selection['selected'] is not None and selection['selected']['max_reliability_gap']>.05
            if branch=='supported':
                model=original_read(out/'model.json')
                assert model['empirical_prior']==head['empirical_prior'] and model['empirical_prior']!=parent['empirical_prior']
                assert model['warp']['base_model']==head and model['shape']==parent['shape'] and model['depth']==parent['depth']
                runner.warp_calibration();runner.warp_review();assert not original_read(out/'warp_calibration.json')['passed']
            proof=runner.verify_executed()
            assert not proof['development_scientific_passed'] and not proof['all17_executed']
            assert proof['production_auxiliary_candidate_metrics']==proof['reference_auxiliary_candidate_metrics']==21
            assert proof['supervised_calibration_selections']==1 and proof['new_head_fits']==0
            runner.cold();runner.finish({'scope':'Synthetic fixed-head branch integration only'},lambda:None)
        done=original_read(out/'closeout.json');assert done['engineering_passed'] and not done['P6_development_scope_passed']
        assert done['all_tasks_percent']==45 and done['experiment_percent']==56.25 and report.exists()
        if branch=='no_feasible':assert 'no_feasible_candidate' in report.read_text(encoding='utf-8')
        if branch=='gap_failure':assert 'selected_multilevel_reliability' in report.read_text(encoding='utf-8')


def main():
    started = time.monotonic(); checks = []; passed = False; failure_trace = None
    record = runner.LOG/'preparation'/('integration_'+uuid.uuid4().hex+'.json')
    record.parent.mkdir(parents=True, exist_ok=True)
    work = runner.LOG/'preparation'/('ledger_'+uuid.uuid4().hex+'.jsonl')
    try:
        for name in runner.SOURCES:
            compile((ROOT/name).read_text(encoding='utf-8-sig'), name, 'exec')
        checks.append('All new sources compile')
        runner.append(work, dict(event='intent', id='missing-return', kind='test', key='only'))
        assert runner.ledger(work)['unknown'] == ['missing-return']
        runner.append(work, dict(event='returned', id='missing-return', kind='test', key='only'))
        assert runner.ledger(work)['known_returns'] == 1 and not runner.ledger(work)['unknown']
        checks.append('Missing return remains unknown until explicit corresponding return')
        with runner.own_lock():
            blocked = False
            try:
                with runner.own_lock(): pass
            except OSError: blocked = True
            assert blocked
        with runner.project_lock(ROOT): pass
        checks.append('Duplicate controller lock rejected and project lock available')
        command = [str(ROOT/'runtime/envs/vision/Scripts/python.exe'), '-X', 'utf8', '-B', '-c', 'import time;time.sleep(3)']
        child = subprocess.Popen(command, cwd=ROOT, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            time.sleep(.75); rss, members = runner.process_tree_rss(child.pid)
            assert child.pid in members and len(members) >= 2 and rss > 0
        finally: child.wait(timeout=10)
        assert runner.process_tree_rss(child.pid) == (0, [])
        checks.append('Actual venv child tree is monitored, exited handles are not false failures')
        assert runner.sha(ROOT/'runtime/pipeline/state.json') == runner.ORIGINAL_STATE
        checks.append('Original pipeline state hash remains unchanged')
        import numpy as np
        from scripts.p6_joint_evaluate_v45 import aggregate_warp
        p = np.ones(81)/81
        metric = runner.warp_metric(p, 0)
        rows = [dict(key=f'toy/{scene}/{i}', split='calibration', scene=scene,
                     warp_method=metric, warp_prior=metric)
                for scene in (30, 31, 34, 38) for i in range(108)]
        summary, gates = aggregate_warp(rows, split='calibration')
        assert len(gates) == 6 and summary['case_count'] == 432
        assert not gates['warp_nll_gain'] and not gates['warp_set_sharper']
        checks.append('Actual runner warp-stat schema enters aggregate432; equal prior cannot pass gain/sharpness')
        check_fixed_branches(record.parent)
        checks.append('Actual no-feasible and feasible-but-gap-failed selections replay all21 candidates and close cold with no model/cal access')
        checks.append('Supported selection assembles same8scene head/prior, passes full21 independent reference, then actual cal6/replay and negative closeout; parent12scene prior deliberately differs')
        command = [str(ROOT/'runtime/envs/vision/Scripts/python.exe'), '-X', 'utf8', '-B', '-m', 'unittest',
                   'tests.test_p6_fixed_mixture_v49', 'tests.test_p6_fixed_mixture_independent_v49', '-v']
        run = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=120, creationflags=subprocess.CREATE_NO_WINDOW)
        (record.with_suffix('.stdout.log')).write_bytes(run.stdout); (record.with_suffix('.stderr.log')).write_bytes(run.stderr)
        assert run.returncode == 0, 'Synthetic tests failed; logs preserved'
        checks.append('Both complete synthetic calibration and independent reference test modules pass')
        passed = True
    except BaseException:
        failure_trace=traceback.format_exc();raise
    finally:
        proof = dict(at=runner.now(), passed=passed, seconds=time.monotonic()-started, checks=checks, failure_trace=failure_trace,
            real_data_fit_calls=0, real_data_metric_calls=0, neural_forwards=0, official_AP=0,
            source_sha256={name:runner.sha(ROOT/name) for name in runner.SOURCES if (ROOT/name).exists()})
        runner.once(record, proof)
        if passed:
            target = runner.LOG/'internal_checks.json'; runner.backup(target, 'pre_freeze_checks')
            runner.atomic(target, proof)
        print(json.dumps(proof))

if __name__ == '__main__': main()
