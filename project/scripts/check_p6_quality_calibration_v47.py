"""Operational preflight and complete synthetic tests; no real-data fit/evaluation."""
import sys, time, uuid, json, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from scripts import p6_quality_calibration_v47 as runner

def check_unsupported_closeout(folder):
    """Exercise actual negative branch with synthetic receipts, no real-data read."""
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory(prefix='unsupported_',dir=folder) as temp:
        root = Path(temp); output = root/'results'; output.mkdir(); report = root/'report.txt'
        evidence = dict(supported=False,passed=False,features={'v':{'passed':False},'h':{'passed':False}},
                        inputs={},parent_oof_review={'synthetic':True})
        (output/'evidence.json').write_text(json.dumps(evidence),encoding='utf-8')
        proof = None
        def read_synthetic(path):
            path = Path(path)
            if path == output/'evidence.json': return evidence
            if path == runner.LOG/'evidence_review.json': return dict(passed=True,supported=False,result={'synthetic':True})
            if path == runner.LOG/'cold.json': return dict(passed=True,proof=proof)
            raise AssertionError('Unsupported branch read unexpected artifact '+str(path))
        calls = dict(intents=2,known_returns=2,unknown=[],counts={'evidence_compute':1,'evidence_replay':1})
        with patch.object(runner,'OUT',output), patch.object(runner,'REPORT',report), \
             patch.object(runner,'read',side_effect=read_synthetic), patch.object(runner,'identity',return_value={'synthetic':True}), \
             patch.object(runner,'ledger',return_value=calls), patch.object(runner,'protected_check'), \
             patch.object(runner,'status'), patch.object(runner,'supervise'), patch.object(runner,'handoff'), \
             patch.object(runner,'rel',side_effect=lambda p:Path(p).name):
            proof = runner.verify_executed()
            assert proof['stages'] == ['evidence','evidence_review']
            assert not proof['conditional_fit_executed'] and not proof['development_scientific_passed']
            assert all(v is None for gates in proof['gates_by_split'].values() for v in gates.values())
            runner.finish({'scope':'Synthetic branch integration only'},lambda:None)
        done = json.loads((output/'closeout.json').read_text(encoding='utf-8'))
        assert done['engineering_passed'] and not done['P6_development_scope_passed']
        assert done['all_tasks_percent'] == 45 and done['experiment_percent'] == 56.25
        assert report.exists() and {p.name for p in output.iterdir()} == {'evidence.json','closeout.json'}


def main():
    started = time.monotonic(); checks = []; passed = False
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
        from src.p6_quality_calibration_v47 import fit_conditional_temperature
        toy_scores = np.zeros((100,81)); toy_scores[:,0] = 3.
        toy_prior = np.full((100,81),1/81)
        toy_targets = np.r_[np.zeros(10,dtype=np.int64),np.arange(90,dtype=np.int64)%80+1]
        toy_x = np.zeros((100,81,16)); toy_x[:,:,0] = 1.; toy_x[:,:,14] = .5
        toy_x[:5,:,0] = 0.; toy_x[:5,:,14] = 0.
        trace = []
        toy_calibration = fit_conditional_temperature(toy_x,toy_scores,toy_prior,toy_targets,.5,on_evaluation=trace.append)
        certificate = runner.conditional_certificate(toy_x,toy_scores,toy_prior,toy_targets,toy_calibration)
        assert certificate['passed'] and certificate['certificate_objective_evaluations'] <= 120
        assert trace == toy_calibration['objective_trace'] and certificate['unknown_rows'] == 5
        assert runner.observable_counts(toy_x)['unknown_rows'] == 5
        checks.append('Actual independent four-corner certificate verifies saved objectives, gradients, KKT and unknown full denominator')
        check_unsupported_closeout(record.parent)
        checks.append('Unsupported evidence verifies and closes cold scope with no model/warp artifact access and all17 null')
        command = [str(ROOT/'runtime/envs/vision/Scripts/python.exe'), '-X', 'utf8', '-B', '-m', 'unittest',
                   'tests.test_p6_quality_calibration_v47', 'tests.test_p6_quality_independent_v47', '-v']
        run = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=120, creationflags=subprocess.CREATE_NO_WINDOW)
        (record.with_suffix('.stdout.log')).write_bytes(run.stdout); (record.with_suffix('.stderr.log')).write_bytes(run.stderr)
        assert run.returncode == 0, 'Synthetic tests failed; logs preserved'
        checks.append('Both complete synthetic calibration and independent reference test modules pass')
        passed = True
    finally:
        proof = dict(at=runner.now(), passed=passed, seconds=time.monotonic()-started, checks=checks,
            real_data_fit_calls=0, real_data_metric_calls=0, neural_forwards=0, official_AP=0,
            source_sha256={name:runner.sha(ROOT/name) for name in runner.SOURCES if (ROOT/name).exists()})
        runner.once(record, proof)
        if passed:
            target = runner.LOG/'internal_checks.json'; runner.backup(target, 'pre_freeze_checks')
            runner.atomic(target, proof)
        print(json.dumps(proof))

if __name__ == '__main__': main()
