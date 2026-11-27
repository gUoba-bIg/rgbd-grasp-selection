"""Bounded actual official GraspNet evaluation of the fixed v51 candidate pool.

Parent owns the project lock. This child owns only its evaluation lock and
records actual eval_grasp and get_grasp_score invocation boundaries separately.
No prediction features are computed here; official references are evaluation-only.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, json, time, uuid, hashlib, argparse, traceback
from pathlib import Path
from contextlib import contextmanager
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'runtime/build/graspnetAPI_eval_v7'))
sys.dont_write_bytecode = True
import numpy as np
from src import p56_metrics_v51 as metrics


def need(value, message):
    if not bool(value):
        raise RuntimeError(message)


def path(name):
    value = (ROOT/name).resolve()
    need(value.is_relative_to(ROOT.resolve()), 'Path escapes project')
    return value


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def identity(p):
    p = Path(p); st = p.stat()
    with p.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(sha256=digest, bytes=st.st_size, mtime_ns=st.st_mtime_ns)


def write_once(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def sealed(p, value):
    write_once(p, value); write_once(p.with_suffix(p.suffix+'.receipt.json'), identity(p))


def load_sealed(p):
    need(identity(p) == read(p.with_suffix(p.suffix+'.receipt.json')), 'Sealed artifact differs '+str(p))
    return read(p)


def append(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())


def event_ledger(p):
    starts, ends = {}, {}
    for row in (json.loads(line) for line in p.read_text('utf8').splitlines()) if p.exists() else []:
        need(row['event'] in ('intent', 'returned'), 'Unknown evaluation event')
        need(row['kind'] in ('eval_grasp', 'get_grasp_score'), 'Unknown evaluation kind')
        need(isinstance(row['key'], str) and bool(row['key']), 'Invalid evaluation key')
        table = starts if row['event'] == 'intent' else ends
        need(row['id'] not in table, 'Duplicate evaluation event')
        table[row['id']] = row
    need(set(ends) <= set(starts), 'Evaluation return without intent')
    for key, row in ends.items():
        need(all(row[k] == starts[key][k] for k in ('kind', 'key', 'parent')), 'Evaluation event identity differs')
    for row in starts.values():
        if row['kind'] == 'get_grasp_score':
            parent = starts.get(row['parent'])
            need(parent is not None and parent['kind'] == 'eval_grasp', 'Getter parent is not an actual eval_grasp intent')
            need(row['key'].startswith(parent['key']+'/') and len(row['key']) > len(parent['key'])+1,
                 'Getter key does not belong to its eval_grasp parent')
        else:
            need(row['parent'] is None, 'Outer eval_grasp unexpectedly has a parent')
    return dict(intents=len(starts), returned=len(ends), unknown=sorted(set(starts)-set(ends)),
        started_by_kind={kind:sum(r['kind'] == kind for r in starts.values()) for kind in ('eval_grasp', 'get_grasp_score')},
        returned_by_kind={kind:sum(r['kind'] == kind for r in ends.values()) for kind in ('eval_grasp', 'get_grasp_score')})


def case_key(case):
    return f"scene_{case['scene']:04d}_{case['frame']:04d}_{case['condition']}"


@contextmanager
def exclusive(log):
    import msvcrt
    log.mkdir(parents=True, exist_ok=True)
    with (log/'.execution.lock').open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0: stream.write(b'0'); stream.flush()
        stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try: yield
        finally: stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


class Evaluation:
    def __init__(self, config_path, mode):
        self.config_path = path(config_path); self.config_identity = identity(self.config_path)
        self.config = read(self.config_path); self.spec = self.config['evaluation']; self.mode = mode
        need(self.config['gates'] == self.spec['gates'] == metrics.GATE_THRESHOLDS, 'Displayed and executed scientific thresholds differ')
        need(self.spec['concordance_scope'] == 'all', 'Conditional-association scope differs')
        self.output = path(self.spec.get('out_dir', 'results/joint_validation_v51/official'))
        self.log = path(self.spec.get('log_dir', 'logs/joint_validation_v51/official'))
        need(self.output.is_relative_to(path('results/joint_validation_v51'))
             and self.log.is_relative_to(path('logs/joint_validation_v51')), 'Wrong v51 output namespace')
        self.cases = self.spec['cases']; self.groups = tuple(self.spec['groups'])
        expected = {(s, 64, c) for s in metrics.SCENES for c in metrics.CONDITIONS}
        need(self.groups == metrics.GROUPS and len(self.cases) == 50
             and {(r['scene'], r['frame'], r['condition']) for r in self.cases} == expected, 'Changed fixed evaluation scope')
        need(self.spec['call_cap'] == 357, 'Official invocation cap differs')
        self.calls = self.log/'calls.jsonl'; self.start = time.monotonic()
        self.prior_seconds = sum(float(read(p)['seconds']) for p in self.log.glob('usage_*.json')) if self.log.exists() else 0.
        self.current_ledger = event_ledger(self.calls)
        need(not self.current_ledger['unknown'], 'Unknown prior official call: preserve, audit before any recovery')
        events = [json.loads(line) for line in self.calls.read_text('utf8').splitlines()] if self.calls.exists() else []
        self.started_keys = {(e['kind'], e['key']) for e in events if e['event'] == 'intent'}
        need(len(self.started_keys) == self.current_ledger['intents'], 'Duplicate historical official invocation key')
        self.geometry = None

    def guard(self):
        need(Path.cwd().resolve() == ROOT.resolve(), 'Explicit project working directory required')
        need(identity(self.config_path) == self.config_identity, 'Evaluation config changed')
        need(not path('logs/joint_validation_v51/STOP').exists(), 'User STOP marker')
        need(self.prior_seconds+time.monotonic()-self.start <= self.spec.get('wall_seconds', 7200), 'Official evaluation cumulative wall cap')
        if self.spec.get('controller_state'):
            state = read(path(self.spec['controller_state']))
            need(bool(state.get('run_token')) and os.environ.get('P56_V51_CHILD_TOKEN') == state['run_token'], 'Official child token mismatch')
            need(time.time() < state['deadline_epoch'], 'Parent cumulative deadline exceeded')

    def geometry_instance(self):
        if self.geometry is None:
            from scripts.p8_geometry_v32 import Geometry
            self.geometry = Geometry(read(path(self.spec['reference_config'])))
        return self.geometry

    def load_case(self, case):
        self.guard()
        ranking_path = path(case['ranking_file']); ranking = read(ranking_path)
        need(all(ranking[k] == case[k] for k in ('scene', 'frame', 'condition')), 'Ranking case identity differs')
        source = path(case['source_dir']); pool = load_sealed(source/'pool.json')
        generation = load_sealed(source/'generation.json')
        observations = source/'observations.npz'; observed_identity = identity(observations)
        need(observed_identity == generation['artifact'], 'Frozen observations changed')
        need(ranking['pool_indices'] == pool['indices'] and len(pool['indices']) == 20, 'Frozen twenty-candidate pool differs')
        need(path(ranking['source_observations']['path']) == observations
             and ranking['source_observations']['identity'] == observed_identity, 'Ranking observations binding differs')
        need(set(ranking['scores']) == set(self.groups), 'Missing/extra ranking groups')
        with np.load(observations, allow_pickle=False) as arrays:
            candidates = arrays['candidates'][pool['indices']].copy()
        for group in self.groups:
            metrics.normalize_ranking(candidates, ranking['scores'][group], pool['indices'])
        return dict(case=case, ranking=ranking, ranking_path=ranking_path, source=source,
            pool_indices=pool['indices'], candidates=candidates,
            bindings=dict(ranking=identity(ranking_path), pool=identity(source/'pool.json'), observations=observed_identity,
                          reference_config=identity(path(self.spec['reference_config']))))

    def invoke(self, kind, key, function, parent=None):
        self.guard()
        need(kind in ('eval_grasp', 'get_grasp_score'), 'Unknown evaluation kind')
        cap = 357 if kind == 'eval_grasp' else 7140
        count = self.current_ledger['started_by_kind'][kind]
        need(count < cap, 'Actual evaluation-call cap exceeded '+kind)
        need((kind, key) not in self.started_keys, 'Duplicate actual invocation: preserve prior returned/partial evidence')
        ident = uuid.uuid4().hex
        append(self.calls, dict(event='intent', id=ident, kind=kind, key=key, parent=parent, at=time.time()))
        self.started_keys.add((kind, key))
        self.current_ledger['started_by_kind'][kind] += 1
        tick = time.monotonic()
        result = function(ident)
        append(self.calls, dict(event='returned', id=ident, kind=kind, key=key, parent=parent,
                               at=time.time(), seconds=time.monotonic()-tick))
        self.current_ledger['returned_by_kind'][kind] += 1
        return result, ident

    def actual_oracle(self, geo, normalized, state, key):
        def evaluate(parent_id):
            original = geo.official.get_grasp_score
            sequence = [0]
            def tracked(*args, **kwargs):
                sequence[0] += 1
                result, _ = self.invoke('get_grasp_score', key+'/'+str(sequence[0]),
                    lambda ignored: original(*args, **kwargs), parent=parent_id)
                return result
            # Instrument only the exact original function boundary; restore in
            # finally, including exceptional exits. Inputs/results are untouched.
            geo.official.get_grasp_score = tracked
            try:
                return geo.oracle(normalized, state)
            finally:
                geo.official.get_grasp_score = original
        return self.invoke('eval_grasp', key, evaluate)

    @staticmethod
    def flatten(value, normalized, order, pool_indices):
        from scripts.verify_grasp_geometry_v10 import exact_indices
        rows = []
        need(len(value['groups']) == len(value['scores']) == len(value['collisions']), 'Official group schema differs')
        for slot, (group, scores, collisions) in enumerate(zip(value['groups'], value['scores'], value['collisions'])):
            a = np.asarray(group, dtype=np.float32).reshape(-1, 17)
            need(len(a) == len(scores) == len(collisions), 'Official row counts differ')
            positions = exact_indices(normalized, a, np)
            for g, position, score, collision in zip(a, positions, scores, collisions):
                rows.append(dict(index=int(pool_indices[int(order[position])]), object_slot=slot,
                    confidence=float(g[0]), grasp=g.tolist(), score=float(score), collision=bool(collision)))
        rows.sort(key=lambda r: -r['confidence'])
        need(len(rows) <= 20 and len({r['index'] for r in rows}) == len(rows), 'Official fixed support differs')
        return rows

    def score(self, cold=False):
        geo = self.geometry_instance()
        cases = [next(r for r in self.cases if r['scene'] == 2 and r['condition'] == 'clean')] if cold else self.cases
        for case in cases:
            unit = self.load_case(case); state = geo.scene(case)
            for group in self.groups:
                destination = self.output/('cold' if cold else 'results')/case_key(case)/(group+'.json')
                if destination.exists():
                    existing = load_sealed(destination)
                    need(existing['bindings'] == unit['bindings'] and existing['config'] == self.config_identity, 'Existing official result bindings differ')
                    continue
                normalized, order = metrics.normalize_ranking(unit['candidates'], unit['ranking']['scores'][group], unit['pool_indices'])
                value, cid = self.actual_oracle(geo, normalized, state, ('cold/' if cold else 'score/')+case_key(case)+'/'+group)
                selected = self.flatten(value, normalized, order, unit['pool_indices'])
                report = metrics.grasp_metrics([r['score'] for r in selected], np.asarray([r['collision'] for r in selected], bool))
                answer = dict(scene=case['scene'], frame=case['frame'], condition=case['condition'], group=group,
                    bindings=unit['bindings'], config=self.config_identity, normalized=normalized.tolist(), local_order=order.tolist(),
                    official=value, selected=selected, metrics=report, call_id=cid,
                    independent_full_benchmark=False, actual_official_entrypoint=True)
                if cold:
                    previous = load_sealed(self.output/'results'/case_key(case)/(group+'.json'))
                    need(all(answer[k] == previous[k] for k in ('bindings', 'normalized', 'local_order', 'official', 'selected', 'metrics')), 'Actual official cold reentry differs')
                sealed(destination, answer)

    def review(self):
        from scripts.verify_grasp_geometry_v10 import select_groups, exact_indices
        geo = self.geometry_instance(); checked = {}
        ledger = event_ledger(self.calls)
        need(not ledger['unknown'], 'Unknown invocation prevents review')
        returned = {r['id']:r for r in (json.loads(line) for line in self.calls.read_text('utf8').splitlines()) if r['event'] == 'returned'}
        used_calls = set()
        for case in self.cases:
            unit = self.load_case(case); state = geo.scene(case)
            labels_path = unit['source']/'labels.json'; labels = load_sealed(labels_path)
            physics = {r['index']:r for r in labels['records']}
            need(set(physics) == set(unit['pool_indices']), 'Cached singleton-label support differs')
            for group in self.groups:
                result_path = self.output/'results'/case_key(case)/(group+'.json'); result = load_sealed(result_path)
                need(result['bindings'] == unit['bindings'] and result['config'] == self.config_identity, 'New result bindings changed')
                need(all(result[k] == case[k] for k in ('scene', 'frame', 'condition')) and result['group'] == group,
                     'Official unit identity differs')
                call = returned.get(result['call_id'])
                need(call and call['kind'] == 'eval_grasp' and call['key'] == 'score/'+case_key(case)+'/'+group
                     and result['call_id'] not in used_calls, 'Official result lacks its unique actual returned call')
                used_calls.add(result['call_id'])
                normalized, order = metrics.normalize_ranking(unit['candidates'], unit['ranking']['scores'][group], unit['pool_indices'])
                need(result['normalized'] == normalized.tolist() and result['local_order'] == order.tolist(), 'Rank normalization changed')
                groups, _, _, _ = select_groups(normalized, state['models'], state['poses'], geo.Group, geo.official, np)
                expected = []
                for slot, grasps in enumerate(groups):
                    for row, j in zip(grasps, exact_indices(normalized, grasps, np)):
                        ident = int(unit['pool_indices'][int(order[j])]); label = physics[ident]
                        need(label['grasp'][1:] == row[1:].tolist() and label['object_slot'] == slot
                             and label['object_id'] == int(state['ids'][slot]), 'Cached label geometry/assignment differs')
                        expected.append(dict(index=ident, object_slot=slot, confidence=float(row[0]), grasp=row.tolist(),
                                             score=float(label['score']), collision=bool(label['collision'])))
                expected.sort(key=lambda r: -r['confidence'])
                need(expected == result['selected'], 'New method selected support/official singleton physics differs')
                need(self.flatten(result['official'], normalized, order, unit['pool_indices']) == expected, 'Saved official arrays differ')
                # Independent explicit loops, not a second call to the metric implementation.
                prefix = [[sum(0 < r['score'] <= mu and not r['collision'] for r in expected[:k])/k
                           for mu in metrics.FRICTIONS] for k in range(1, 51)]
                saved = result['metrics']
                need(np.array_equal(np.asarray(prefix), np.asarray(saved['prefix']))
                     and float(np.mean(prefix)) == saved['AP'] and float(np.mean(np.asarray(prefix)[:, 1])) == saved['AP_04']
                     and saved['selected'] == len(expected), 'Official AP denominator/formula differs')
                for k in metrics.TOP_K:
                    item = saved['topk'][str(k)]
                    need(item['denominator'] == k and item['available'] == min(k, len(expected))
                         and item['mu04'] == prefix[k-1][1] and item['six_average'] == float(np.mean(prefix[k-1]))
                         and item['any_mu04'] == any(0 < r['score'] <= .4 and not r['collision'] for r in expected[:k]), 'Official TopK differs')
                checked[result_path.relative_to(ROOT).as_posix()] = identity(result_path)
            checked[labels_path.relative_to(ROOT).as_posix()] = identity(labels_path)
        need(len(checked) == 400 and len(used_calls) == 350, 'Incomplete full350 evaluation plus50 label review')
        sealed(self.output/'review.json', dict(passed=True, official_units=350, label_sets=50, checked=checked,
            actual_eval_grasp_calls_verified=sorted(used_calls),
            new_official_calls=0, all_ranked_supports_rebuilt=True, all_singleton_physics_crosschecked=True,
            all_AP_TopK_recomputed=True, config=self.config_identity))

    def summarize(self):
        proof = load_sealed(self.output/'review.json')
        need(proof['passed'] and proof['config'] == self.config_identity, 'Full evaluation review missing')
        for name, expected in proof['checked'].items():
            need(identity(path(name)) == expected, 'Reviewed official evidence changed')
        rows, diagnostics, support_rows = [], [], []
        for case in self.cases:
            unit = self.load_case(case)
            labels = load_sealed(unit['source']/'labels.json'); physical = {r['index']:r for r in labels['records']}
            positive = np.asarray([0 < physical[ix]['score'] <= .4 and not physical[ix]['collision'] for ix in unit['pool_indices']], bool)
            scores = {g:np.asarray(unit['ranking']['scores'][g], np.float64) for g in self.groups}
            association = metrics.conditional_concordance(scores['A'], scores['A']-scores['C'], scores['B']-scores['D'], positive, unit['pool_indices'])
            diagnostics.append(dict(scene=case['scene'], condition=case['condition'], **association))
            for group in self.groups:
                result = load_sealed(self.output/'results'/case_key(case)/(group+'.json'))
                rows.append(dict(scene=case['scene'], condition=case['condition'], group=group, metrics=result['metrics']))
                support = metrics.candidate_support(unit['ranking'], group, result['selected'], physical)
                support_rows.append(dict(scene=case['scene'], condition=case['condition'], group=group, **support))
        for group in self.groups:
            cold_path = self.output/'cold'/'scene_0002_0064_clean'/(group+'.json')
            cold = load_sealed(cold_path); previous = load_sealed(self.output/'results'/'scene_0002_0064_clean'/(group+'.json'))
            need(cold['official'] == previous['official'] and cold['selected'] == previous['selected'], 'Actual cold reentry missing/different')
            returned = {r['id']:r for r in (json.loads(line) for line in self.calls.read_text('utf8').splitlines()) if r['event'] == 'returned'}
            event = returned.get(cold['call_id'])
            need(event and event['kind'] == 'eval_grasp' and event['key'] == 'cold/scene_0002_0064_clean/'+group,
                 'Cold evidence lacks actual returned entrypoint')
        ledger = event_ledger(self.calls)
        need(not ledger['unknown'] and ledger['returned_by_kind']['eval_grasp'] == 357, 'Official actual count incomplete')
        result = metrics.summarize(rows, diagnostics)
        sealed(self.output/'metrics.json', dict(**result, rows=rows, concordance_cases=diagnostics,
            candidate_support=metrics.summarize_candidate_support(support_rows), candidate_support_rows=support_rows,
            ledger=ledger, actual_get_grasp_score_returns=ledger['returned_by_kind']['get_grasp_score'],
            deeper_force_closure_call_count='not instrumented; do not infer from candidate slots',
            full_review=identity(self.output/'review.json'), config=self.config_identity))

    def run(self):
        self.guard()
        if self.mode == 'score': self.score()
        elif self.mode == 'cold': self.score(cold=True)
        elif self.mode == 'review': self.review()
        else: self.summarize()
        self.guard()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=('score', 'review', 'cold', 'metrics'), required=True)
    args = parser.parse_args(); config = read(path(args.config))
    log = path(config['evaluation'].get('log_dir', 'logs/joint_validation_v51/official'))
    with exclusive(log):
        run = Evaluation(args.config, args.mode); completed = False; error = None
        try:
            run.run(); completed = True
        except BaseException:
            error = traceback.format_exc(); raise
        finally:
            write_once(log/('usage_'+args.mode+'_'+uuid.uuid4().hex+'.json'), dict(seconds=time.monotonic()-run.start,
                completed=completed, mode=args.mode, error=error, actual_calls=event_ledger(run.calls)))


if __name__ == '__main__':
    main()
