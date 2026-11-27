"""Operational preflight and complete synthetic tests; no real-data fit/evaluation."""
import sys, time, uuid, json, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from scripts import p6_probability_calibration_v46 as runner

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
        from src.p6_probability_calibration_v46 import calibrate_temperature
        toy_scores = np.zeros((100,81)); toy_scores[:,0] = 3.
        toy_prior = np.full((100,81),1/81)
        toy_targets = np.r_[np.zeros(10,dtype=np.int64),np.arange(90,dtype=np.int64)%80+1]
        toy_calibration = calibrate_temperature(toy_scores,toy_prior,toy_targets)
        certificate = runner.temperature_certificate(toy_scores,toy_prior,toy_targets,toy_calibration)
        assert certificate['passed'] and 0 < certificate['tau'] < 1
        assert certificate['certificate_objective_evaluations'] <= 64
        checks.append('Actual runner independent convex certificate validates internal optimum and bounded replay')
        command = [str(ROOT/'runtime/envs/vision/Scripts/python.exe'), '-X', 'utf8', '-B', '-m', 'unittest',
                   'tests.test_p6_probability_calibration_v46', 'tests.test_p6_calibration_independent_v46', '-v']
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
