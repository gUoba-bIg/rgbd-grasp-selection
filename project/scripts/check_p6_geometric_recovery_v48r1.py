"""Operational preflight and complete synthetic tests; no real-data fit/evaluation."""
import os, sys, time, uuid, json, subprocess, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from scripts import p6_geometric_recovery_v48r1 as runner

def check_nested_negative_branch(folder):
    """A synthetic old-failure fixture exercises actual recovery without real data.

    Constructed historical counters are test inputs, never claimed as executed
    production work. Actual test compute is charged only by elapsed preparation.
    """
    import copy
    import numpy as np
    from unittest.mock import patch
    from src.p6_probability_calibration_v46 import FOLDS
    from src.p6_geometric_logits_v48 import fit_base_head,score_warp
    from scripts.p6_mixture_v43 import arrays_read,arrays_save
    root=folder/('recovery_fixture_'+uuid.uuid4().hex);output=root/'results';log=root/'logs'
    oldout=root/'old_results';oldlog=root/'old_logs'
    for path in (output,log,oldout,oldlog):path.mkdir(parents=True)
    report=root/'report.txt';scenes=np.concatenate([np.repeat(group,108) for group in FOLDS]);assigned=np.repeat(np.arange(3),432)
    x=np.random.RandomState(48).normal(scale=.02,size=(1296,81,16));y=np.full(1296,40,dtype=np.int64)
    scores=np.zeros((1296,81));scores[:,40]=10.;priors=np.full((1296,81),1/81)
    oof=dict(scores=scores,priors=priors,targets=y,scenes=scenes,assigned_folds=assigned,visits=np.ones(1296,dtype=np.int64))
    rows=[dict(key=f'fixture/{i}',scene=int(scene),split='fit') for i,scene in enumerate(scenes)]
    original_read,original_identity,original_sha=runner.read,runner.identity,runner.sha
    # Fabricate only the historical-control metadata to test fixed62/67 counters.
    head0=fit_base_head(x[:432],y[:432]);assert head0['optimizer_converged']
    records=head0['objective_evaluation_records'];assert len(records)<=62
    head0['objective_evaluation_records']=records+[copy.deepcopy(records[-1]) for _ in range(62-len(records))]
    head0['objective_trace']=[r['value'] for r in head0['objective_evaluation_records']]
    head0.update(objective_evaluations=62,optimizer_iterations=57)
    seed=fit_base_head(x[432:864],y[432:864]);first=copy.deepcopy(seed['objective_evaluation_records'][0])
    seed.update(coefficients=first['theta'],optimizer_converged=False,optimizer_iterations=60,objective_evaluations=67,
        objective_evaluation_records=[copy.deepcopy(first) for _ in range(67)],objective_trace=[first['value']]*67,
        optimizer_message='SYNTHETIC iteration-cap fixture; not actual historical run')
    runner.once(oldout/'head_0.json',head0);runner.once(oldout/'head_1.json',seed);runner.once(oldout/'nested_rows.json',rows)
    runner.once(oldout/'head_0_provenance.json',dict(source_fold=0,training_indices=list(range(432)),
        training_scenes=sorted(FOLDS[0]),input_sha256='synthetic_input'))
    for i,head in ((0,head0),(1,seed)):
        for row in head['objective_evaluation_records']:runner.append(oldlog/f'head_{i}_objective.jsonl',dict(at='synthetic',**row))
    for target in (1,2):
        held=np.flatnonzero(assigned==target);score=score_warp(head0,x[held])
        arrays_save(oldout/f'cross_0_to_{target}.npz',scores=score,priors=np.broadcast_to(head0['empirical_prior'],score.shape).copy(),indices=held)
    old_account=dict(old_production_objective_evaluations=129,old_head_attempts=2,head0_objectives=62,
        head1_failed_objectives=67,old_accounted_seconds=1.,synthetic_fixture=True)
    def scoped_identity(path):
        return original_identity(path) if Path(path).resolve().is_relative_to(root.resolve()) else {'synthetic_input':True}
    def scoped_sha(path):
        return original_sha(path) if Path(path).resolve().is_relative_to(root.resolve()) else 'synthetic_input'
    def scoped_read(path):
        if Path(path).name in ('model.json','geometric_fit.json','warp_calibration.json','calibration.json','evaluation.json'):
            raise AssertionError('Rejected nested branch accessed unrun final artifact '+str(path))
        assert Path(path).resolve().is_relative_to(root.resolve()),'Fixture attempted real-data read'
        return original_read(path)
    with patch.object(runner,'OUT',output),patch.object(runner,'LOG',log),patch.object(runner,'REPORT',report), \
         patch.object(runner,'FAILED',oldout),patch.object(runner,'FAILED_LOG',oldlog), \
         patch.object(runner,'oof_inputs',return_value=(x,y,rows,oof,{})),patch.object(runner,'identity',side_effect=scoped_identity), \
         patch.object(runner,'read',side_effect=scoped_read),patch.object(runner,'sha',side_effect=scoped_sha), \
         patch.object(runner,'old_failure_audit',return_value=old_account),patch.object(runner,'child_guard'), \
         patch.object(runner,'protected_check'),patch.object(runner,'status'),patch.object(runner,'supervise'), \
         patch.object(runner,'handoff'),patch.object(runner,'_seen',None),patch.object(runner,'rel',side_effect=lambda p:Path(p).name):
        runner.nested_fit()
        assert not original_read(output/'nested_gate.json')['supported']
        cross=output/'cross_0_to_1.npz';prior_bytes=cross.read_bytes();prior_stat=cross.stat();block=arrays_read(cross);block['indices']=block['indices'][::-1]
        with cross.open('wb') as stream:np.savez_compressed(stream,**block)
        rejected=False
        try:runner.meta_training(2,oof)
        except (AssertionError,ValueError,RuntimeError):rejected=True
        finally:
            cross.write_bytes(prior_bytes);os.utime(cross,ns=(prior_stat.st_atime_ns,prior_stat.st_mtime_ns))
        assert rejected,'Misaligned reused cross-score rows must be rejected'
        runner.nested_review();proof=runner.verify_executed()
        assert proof['stages']==['nested_fit','nested_review'] and not proof['final_fit_executed']
        assert not proof['development_scientific_passed'] and all(v is None for gates in proof['gates_by_split'].values() for v in gates.values())
        assert proof['production_accounting']['total_fit_attempts_including_old_failure']==7
        assert proof['production_accounting']['total_head_attempts']==4
        assert proof['production_accounting']['head1_including_failed_objectives']<=120
        for source in (1,2):
            head=original_read(output/f'head_{source}.json')
            assert head['optimizer_converged'] and head['final_gradient_inf']<=1e-5 and head['hessian_evaluations']>0
        assert original_read(oldout/'head_1.json')['optimizer_converged'] is False
        runner.cold();runner.finish({'scope':'Synthetic numerical-recovery integration only'},lambda:None)
    done=original_read(output/'closeout.json')
    assert done['engineering_passed'] and not done['P6_development_scope_passed']
    assert done['all_tasks_percent']==45 and done['experiment_percent']==56.25
    assert report.exists() and {p.name for p in output.iterdir()}==runner.nested_names()|{'closeout.json'}
    assert done['proof']['calls']['counts']['head1_continuation']==1 and done['proof']['calls']['counts']['head2_initial']==1
    assert done['proof']['calls']['counts']['nested_cross_prediction']==4
    assert all(original_sha(oldout/name)==original_sha(output/name) for name in runner.REUSE)


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
        check_nested_negative_branch(record.parent)
        checks.append('Actual synthetic recovery reuses old head0/two cross blocks, continues failed head1 once and cold-fits head2; Newton Hessian/Armijo certificates, all nested replays, cold and negative closeout pass')
        checks.append('Corrupted cross-score row IDs are rejected and exact prior bytes restored before replay')
        command = [str(ROOT/'runtime/envs/vision/Scripts/python.exe'), '-X', 'utf8', '-B', '-m', 'unittest',
                   'tests.test_p6_geometric_solver_v48r1', 'tests.test_p6_geometric_recovery_independent_v48r1', '-v']
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
