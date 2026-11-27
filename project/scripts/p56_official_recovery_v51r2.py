"""Second bounded engineering recovery; frozen v51 science is inherited.

The complete combined ledger permanently retains historical unknown calls.
Eligible returned-event views are explicitly labelled read indexes, never total
ledgers. Existing scientific results and all historical event bytes are read-only.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, json, time, uuid, argparse, traceback, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from scripts import p56_official_v51 as frozen

need, path, read, identity = frozen.need, frozen.path, frozen.read, frozen.identity
KINDS = ('eval_grasp', 'get_grasp_score')
SHARDS = (39,)
CAPS = dict(eval_grasp=359, get_grasp_score=7140)


def raw_events(ledger_path):
    """Keep source bytes and line locations alongside parsed audit metadata."""
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        return []
    frozen.event_ledger(ledger_path)
    result = []
    for number, line in enumerate(ledger_path.read_bytes().splitlines(keepends=True), 1):
        need(line.endswith(b'\n'), 'Incomplete ledger line: preserve unknown raw evidence')
        result.append(dict(row=json.loads(line.decode('utf8')), raw=line,
                           source=ledger_path, line_number=number))
    return result


def combined_ledger(historical_paths, new_path, historical_unknowns, historical_retry, original_unknown):
    """Audit three actual ledgers without resolving either historical unknown."""
    paths = [Path(p) for p in historical_paths]
    need(len(paths) == len(set(paths)) == 2 and Path(new_path) not in paths, 'Recovery requires two distinct historical ledgers')
    historical = [e for p in paths for e in raw_events(p)]
    entries = historical+raw_events(new_path); starts = {}; returns = {}; by_key = {}
    for entry in entries:
        row = entry['row']; target = starts if row['event'] == 'intent' else returns
        need(row['id'] not in target, 'Duplicate call identity across actual ledgers')
        target[row['id']] = entry
        if row['event'] == 'intent':
            by_key.setdefault((row['kind'], row['key']), []).append(entry)
    need(set(returns) <= set(starts), 'Combined return has no actual intent')
    old_starts = {e['row']['id']:e for e in historical if e['row']['event'] == 'intent'}
    old_returns = {e['row']['id'] for e in historical if e['row']['event'] == 'returned'}
    historical_unknown = set(old_starts)-old_returns
    expected_unknowns = {r['id']:r for r in historical_unknowns}
    need(len(expected_unknowns) == len(historical_unknowns) == 2 and historical_unknown == set(expected_unknowns),
         'Historical unknown set differs')
    for ident, expected in expected_unknowns.items():
        event = old_starts[ident]
        need(event['row']['kind'] == 'eval_grasp' and event['row']['parent'] is None
             and all(event['row'][k] == expected[k] for k in ('id', 'kind', 'key'))
             and event['source'].relative_to(ROOT).as_posix() == expected['ledger_path'],
             'Historical unknown source/identity differs')
    need(original_unknown['id'] in historical_unknown
         and all(original_unknown[k] == expected_unknowns[original_unknown['id']][k] for k in ('id', 'kind', 'key'))
         and old_starts[original_unknown['id']]['source'] == paths[1], 'Current retry is not the second historical unknown')
    previous_unknown = historical_retry['unknown_id']; previous_retry = historical_retry['retry_id']
    need(previous_unknown in historical_unknown and previous_unknown != original_unknown['id']
         and old_starts[previous_unknown]['source'] == paths[0]
         and historical_retry['key'] == old_starts[previous_unknown]['row']['key']
         and previous_retry in old_returns and old_starts[previous_retry]['source'] == paths[1],
         'Previously returned scene22 retry identity differs')
    retry_ids, new_retry_ids = [], []
    for key, attempts in by_key.items():
        if len(attempts) > 1:
            need(len(attempts) == 2 and key[0] == 'eval_grasp', 'Repeated call is not an authorized engineering retry')
            first, second = attempts; retry_id = second['row']['id']
            if key[1] == historical_retry['key']:
                need(first['row']['id'] == previous_unknown and retry_id == previous_retry,
                     'Previously completed engineering retry was repeated or changed')
            else:
                need(key == ('eval_grasp', original_unknown['key'])
                     and first['row']['id'] == original_unknown['id'] and second['source'] == Path(new_path),
                     'Repeated call is not the current authorized engineering retry')
                new_retry_ids.append(retry_id)
            retry_ids.append(retry_id)
            need(second['row'].get('retry_of') == first['row']['id'], 'Retry intent lacks its historical unknown identity')
            if retry_id in returns:
                need(returns[retry_id]['row'].get('retry_of') == first['row']['id'],
                     'Retry return lacks its historical unknown identity')
    need(previous_retry in retry_ids, 'Previous returned retry missing from complete accounting')
    unknown_ids = sorted(set(starts)-set(returns))
    counts = dict(intents=len(starts), returned=len(returns), unknown=unknown_ids,
        started_by_kind={kind:sum(e['row']['kind'] == kind for e in starts.values()) for kind in KINDS},
        returned_by_kind={kind:sum(e['row']['kind'] == kind for e in returns.values()) for kind in KINDS})
    unknown = [dict(id=k, kind=starts[k]['row']['kind'], key=starts[k]['row']['key'],
                    ledger_path=starts[k]['source'].relative_to(ROOT).as_posix(),
                    intent_line_number=starts[k]['line_number'], returned=False,
                    raw_line_sha256=hashlib.sha256(starts[k]['raw']).hexdigest(),
                    historical=k in historical_unknown) for k in unknown_ids]
    return dict(**counts, unknown_evidence=unknown, engineering_retry_ids=retry_ids,
        historical_engineering_retry_ids=[previous_retry], new_engineering_retry_ids=new_retry_ids,
        historical_unknown_ids=sorted(historical_unknown),
        new_unknown_ids=sorted(set(unknown_ids)-historical_unknown),
        caps=dict(CAPS), count_meaning='Actual intent/return boundaries; unknown calls remain unknown',
        event_sources=[dict(id=e['row']['id'], event=e['row']['event'], kind=e['row']['kind'], key=e['row']['key'],
            ledger_path=e['source'].relative_to(ROOT).as_posix(), line_number=e['line_number'],
            raw_line_sha256=hashlib.sha256(e['raw']).hexdigest()) for e in entries])


def write_eligible_view(historical_paths, new_path, historical_unknowns, historical_retry, original_unknown, destination):
    """Write paired actual rows verbatim, with explicit exclusions and provenance."""
    destination = Path(destination)
    audit = combined_ledger(historical_paths, new_path, historical_unknowns, historical_retry, original_unknown)
    need(not audit['new_unknown_ids'], 'New unknown call prevents an eligible result view')
    all_paths = [*historical_paths, new_path]
    entries = [e for p in all_paths for e in raw_events(p)]
    starts = {e['row']['id']:e for e in entries if e['row']['event'] == 'intent'}
    eligible = {e['row']['id'] for e in entries if e['row']['event'] == 'returned'}
    eligible = {k for k in eligible if starts[k]['row']['kind'] == 'eval_grasp'
                or starts[k]['row']['parent'] in eligible}
    chosen = [e for e in entries if e['row']['id'] in eligible]
    excluded = [e for e in entries if e['row']['id'] not in eligible]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        for entry in chosen:
            stream.write(entry['raw'])
        stream.flush(); os.fsync(stream.fileno())
    frozen.write_once(destination.with_suffix(destination.suffix+'.receipt.json'), identity(destination))
    view_ledger = frozen.event_ledger(destination)
    need(not view_ledger['unknown'], 'Eligible read view is not complete')
    manifest = dict(kind='eligible actual returned-event read view', is_total_ledger=False,
        source_ledgers={str(Path(p).relative_to(ROOT).as_posix()):identity(p)
                        for p in all_paths if Path(p).exists()},
        view=identity(destination), view_ledger=view_ledger, complete_combined_ledger=audit,
        included_rows=[dict(id=e['row']['id'], event=e['row']['event'],
            ledger_path=e['source'].relative_to(ROOT).as_posix(), line_number=e['line_number'],
            raw_line_sha256=hashlib.sha256(e['raw']).hexdigest()) for e in chosen],
        excluded_rows=[dict(id=e['row']['id'], event=e['row']['event'], kind=e['row']['kind'], key=e['row']['key'],
            ledger_path=e['source'].relative_to(ROOT).as_posix(), line_number=e['line_number'],
            raw_line_sha256=hashlib.sha256(e['raw']).hexdigest(),
            reason='unknown intent or returned getter without an eligible returned parent') for e in excluded],
        raw_rows_unchanged=True, unknown_was_not_reclassified=True)
    frozen.sealed(destination.with_suffix('.manifest.json'), manifest)
    return manifest


def verify_eligible_view(destination):
    """Check raw provenance even after the actual recovery ledger is appended."""
    destination = Path(destination)
    manifest = frozen.load_sealed(destination.with_suffix('.manifest.json'))
    need(manifest['kind'] == 'eligible actual returned-event read view' and not manifest['is_total_ledger'],
         'Returned-event view misrepresented as complete ledger')
    need(identity(destination) == manifest['view'] == read(destination.with_suffix(destination.suffix+'.receipt.json')),
         'Eligible read view changed')
    sources = {}; prefix_entries = []
    for name, expected in manifest['source_ledgers'].items():
        raw = path(name).read_bytes(); prefix = raw[:expected['bytes']]
        need(len(prefix) == expected['bytes'] and hashlib.sha256(prefix).hexdigest() == expected['sha256'],
             'Eligible view source prefix changed')
        sources[name] = prefix.splitlines(keepends=True)
        for number, raw_line in enumerate(sources[name], 1):
            row = json.loads(raw_line.decode('utf8'))
            prefix_entries.append(dict(row=row, ledger_path=name, line_number=number, raw=raw_line))
    returned = {e['row']['id'] for e in prefix_entries if e['row']['event'] == 'returned'}
    starts = {e['row']['id']:e['row'] for e in prefix_entries if e['row']['event'] == 'intent'}
    eligible = {k for k in returned if starts[k]['kind'] == 'eval_grasp' or starts[k]['parent'] in returned}
    expected_included = [dict(id=e['row']['id'], event=e['row']['event'], ledger_path=e['ledger_path'],
        line_number=e['line_number'], raw_line_sha256=hashlib.sha256(e['raw']).hexdigest())
        for e in prefix_entries if e['row']['id'] in eligible]
    need(manifest['included_rows'] == expected_included, 'Eligible read view omitted or added actual paired events')
    expected_excluded = [(e['row']['id'], e['row']['event'], e['ledger_path'], e['line_number'],
                          hashlib.sha256(e['raw']).hexdigest()) for e in prefix_entries if e['row']['id'] not in eligible]
    need([(e['id'], e['event'], e['ledger_path'], e['line_number'], e['raw_line_sha256'])
          for e in manifest['excluded_rows']] == expected_excluded, 'Excluded historical event provenance differs')
    reproduced = []
    for entry in manifest['included_rows']:
        raw = sources[entry['ledger_path']][entry['line_number']-1]
        event = json.loads(raw.decode('utf8'))
        need(event['id'] == entry['id'] and event['event'] == entry['event']
             and hashlib.sha256(raw).hexdigest() == entry['raw_line_sha256'], 'Eligible row provenance differs')
        reproduced.append(raw)
    need(b''.join(reproduced) == destination.read_bytes(), 'Eligible rows are not verbatim source rows')
    need(frozen.event_ledger(destination) == manifest['view_ledger'], 'Eligible event accounting changed')
    return manifest


def audit_combined(config):
    """Read-only root/controller interface; no model, label or physics call."""
    config = read(path(config)) if not isinstance(config, dict) else config
    return combined_ledger([path(p) for p in config['historical_calls_paths']],
        path('logs/joint_validation_v51r2/official/calls.jsonl'), config['historical_unknowns'],
        config['historical_retry'], config['original_unknown'])


class RecoveryEvaluation(frozen.Evaluation):
    """New lifecycle/accounting only; ranking, physics, AP and gates stay frozen."""
    def __init__(self, recovery_config_path, mode, scene=None):
        self.recovery_config_path = path(recovery_config_path)
        self.recovery_config_identity = identity(self.recovery_config_path)
        self.recovery = read(self.recovery_config_path)
        self.config_path = path(self.recovery['base_config_path'])
        self.config_identity = identity(self.config_path)
        need(self.config_identity == self.recovery['base_config_identity'], 'Frozen scientific configuration differs')
        self.config = read(self.config_path); self.spec = self.config['evaluation']; self.mode = mode
        need(self.config['gates'] == self.spec['gates'] == frozen.metrics.GATE_THRESHOLDS,
             'Scientific thresholds changed in recovery')
        need(self.spec['concordance_scope'] == 'all' and self.spec['call_cap'] == 357,
             'Frozen scientific scope changed')
        self.cases = self.spec['cases']; self.groups = tuple(self.spec['groups'])
        expected = {(s, 64, c) for s in frozen.metrics.SCENES for c in frozen.metrics.CONDITIONS}
        need(len(self.cases) == 50 and {(r['scene'], r['frame'], r['condition']) for r in self.cases} == expected
             and self.groups == frozen.metrics.GROUPS, 'Changed fifty-case seven-group scope')
        need(tuple(self.recovery['shards']) == SHARDS and self.recovery['combined_caps'] == CAPS
             and self.recovery['formal_wall_seconds'] == self.spec['wall_seconds'] == 7200,
             'Changed recovery resource or shard limits')
        need((mode == 'score' and scene in SHARDS) or (mode in ('review', 'cold', 'metrics') and scene is None),
             'Recovery score requires one fixed scene; other modes require the full scope')
        self.scene = scene
        self.output = path(self.spec['out_dir'])
        need(self.output == path('results/joint_validation_v51/official'), 'Changed scientific output namespace')
        self.log = path('logs/joint_validation_v51r2/official')
        self.historical_calls = [path(p) for p in self.recovery['historical_calls_paths']]
        self.new_calls = self.log/'calls.jsonl'
        need(self.historical_calls == [path('logs/joint_validation_v51/official/calls.jsonl'),
                                      path('logs/joint_validation_v51r1/official/calls.jsonl')], 'Wrong historical ledgers')
        self.historical_unknowns = self.recovery['historical_unknowns']
        self.historical_retry = self.recovery['historical_retry']
        self.original_unknown = self.recovery['original_unknown']
        need(self.recovery['allowed_retry_key'] == self.original_unknown['key'], 'Authorized retry key differs')
        self.calls = self.new_calls; self.geometry = None; self.start = time.monotonic()
        self.prior_seconds = float(self.recovery['original_formal_seconds'])
        need(self.prior_seconds == 1316.3910000000615, 'Historical formal consumption was reset')
        self.prior_seconds += sum(float(read(p)['seconds']) for p in self.log.glob('usage_*.json')) if self.log.exists() else 0.
        for name, expected_identity in self.recovery['source_bindings'].items():
            need(identity(path(name)) == expected_identity, 'Frozen recovery prerequisite changed '+name)
        self.metadata_protected = {**self.recovery['protected'], **self.recovery['source_bindings']}
        self.last_metadata_check = -float('inf')
        need(all(p in self.recovery['source_bindings'] for p in self.recovery['historical_calls_paths']),
             'Historical ledger identities not frozen')
        self.initial_audit = self.audit()
        need(not self.initial_audit['new_unknown_ids'], 'Preserve new unknown calls; no second automatic retry')
        need(all(self.initial_audit['started_by_kind'][k] <= CAPS[k] for k in KINDS), 'Combined invocation cap exceeded')
        self.current_ledger = self.initial_audit
        all_events = [e for p in [*self.historical_calls, self.new_calls] for e in raw_events(p)]
        self.started_keys = {(e['row']['kind'], e['row']['key']) for e in all_events if e['row']['event'] == 'intent'}
        self.returned_keys = {(e['row']['kind'], e['row']['key']) for e in all_events if e['row']['event'] == 'returned'}
        self.new_started_keys = {(e['row']['kind'], e['row']['key']) for e in raw_events(self.new_calls)
                                 if e['row']['event'] == 'intent'}
        self.asset_preflight = path('results/joint_validation_v51r2/asset_preflight.json')
        self.asset_preflight_identity = None
        if self.mode in ('score', 'cold'):
            self.verify_asset_preflight()

    def verify_asset_preflight(self):
        preflight = frozen.load_sealed(self.asset_preflight)
        need(preflight['passed'] is True
             and preflight['adapter_source'] == identity(path('src/p56_lazy_geometry_v51r2.py')),
             'Exact model-loading resource preflight has not passed')
        self.asset_preflight_identity = identity(self.asset_preflight)

    def audit(self):
        return combined_ledger(self.historical_calls, self.new_calls, self.historical_unknowns,
                               self.historical_retry, self.original_unknown)

    def guard(self):
        need(Path.cwd().resolve() == ROOT.resolve(), 'Explicit project working directory required')
        need(identity(self.config_path) == self.config_identity
             and identity(self.recovery_config_path) == self.recovery_config_identity, 'Recovery/scientific config changed')
        need(self.recovery_config_identity == read(path('logs/joint_validation_v51r2/freeze.json'))['config'],
             'Recovery configuration does not match its freeze receipt')
        if time.monotonic()-self.last_metadata_check >= 2.:
            self.verify_protected_metadata()
        for name, old_calls in zip(self.recovery['historical_calls_paths'], self.historical_calls):
            need(identity(old_calls) == self.recovery['source_bindings'][name], 'Historical actual-call ledger changed')
        if self.asset_preflight_identity is not None:
            need(identity(self.asset_preflight) == self.asset_preflight_identity, 'Verified asset preflight changed')
        need(not path('logs/joint_validation_v51r2/STOP').exists(), 'User STOP marker')
        need(self.prior_seconds+time.monotonic()-self.start <= 7200, 'Combined formal wall cap exceeded')
        state = read(path(self.recovery['controller_state']))
        need(bool(state.get('run_token')) and os.environ.get('P56_V51R2_CHILD_TOKEN') == state['run_token'],
             'Recovery child token mismatch')
        need(time.time() < state['deadline_epoch'], 'Recovery cumulative deadline exceeded')

    def verify_protected_metadata(self):
        for name, expected in self.metadata_protected.items():
            st = path(name).stat()
            need(st.st_size == expected['bytes'] and st.st_mtime_ns == expected['mtime_ns'],
                 'Protected metadata changed '+name)
        self.last_metadata_check = time.monotonic()

    def geometry_instance(self):
        if self.geometry is None:
            from src.p56_lazy_geometry_v51r2 import LazyGeometry
            self.geometry = LazyGeometry(read(path(self.spec['reference_config'])),
                guard=self.guard, on_event=self.loader_event)
        return self.geometry

    def loader_event(self, event):
        """Actual model-load evidence is separate from official score calls."""
        need(isinstance(event, dict), 'Malformed lazy model-loading event')
        frozen.append(self.log/'model_loading_events.jsonl', dict(event,
            recovery_mode=self.mode, shard=self.scene,
            official_parent_id=getattr(self, 'active_official_call_id', None)))

    def invoke(self, kind, key, function, parent=None):
        self.guard()
        need(self.mode in ('score', 'cold') and self.calls == self.new_calls, 'Read-only mode cannot invoke official physics')
        need(kind in KINDS, 'Unknown evaluation kind')
        need(self.current_ledger['started_by_kind'][kind] < CAPS[kind], 'Combined actual call cap exceeded')
        pair = (kind, key)
        need(pair not in self.returned_keys and pair not in self.new_started_keys, 'Previously returned or newly attempted call cannot repeat')
        retry = pair in self.started_keys
        if retry:
            need(pair == ('eval_grasp', self.original_unknown['key']), 'Only the frozen historical unknown may be retried once')
        ident = uuid.uuid4().hex
        frozen.append(self.new_calls, dict(event='intent', id=ident, kind=kind, key=key, parent=parent, at=time.time(),
            retry_of=self.original_unknown['id'] if retry else None))
        self.started_keys.add(pair); self.new_started_keys.add(pair)
        self.current_ledger['started_by_kind'][kind] += 1
        tick = time.monotonic()
        previous_parent = getattr(self, 'active_official_call_id', None)
        if kind == 'eval_grasp':
            self.active_official_call_id = ident
        try:
            result = function(ident)
        finally:
            self.active_official_call_id = previous_parent
        frozen.append(self.new_calls, dict(event='returned', id=ident, kind=kind, key=key, parent=parent, at=time.time(),
            seconds=time.monotonic()-tick, retry_of=self.original_unknown['id'] if retry else None))
        self.returned_keys.add(pair); self.current_ledger['returned_by_kind'][kind] += 1
        return result, ident

    def score(self, cold=False):
        complete = self.cases
        if not cold:
            self.cases = [r for r in complete if r['scene'] == self.scene]
            need(len(self.cases) == 5, 'Shard must contain five original conditions')
        try:
            return super().score(cold=cold)
        finally:
            self.cases = complete

    def run(self):
        self.guard()
        if self.mode in ('review', 'metrics'):
            self.calls = self.log/('eligible_returned_'+self.mode+'.jsonl')
            write_eligible_view(self.historical_calls, self.new_calls, self.historical_unknowns,
                                self.historical_retry, self.original_unknown, self.calls)
        super().run()
        audit = self.audit()
        need(not audit['new_unknown_ids'], 'Unknown recovery call prevents completed stage')
        proof = dict(mode=self.mode, scene=self.scene, complete_combined_ledger=audit,
            recovery_config=self.recovery_config_identity, frozen_scientific_config=self.config_identity,
            historical_ledgers={p.relative_to(ROOT).as_posix():identity(p) for p in self.historical_calls},
            new_ledger=identity(self.new_calls),
            source_bindings_preserved=True, historical_unknown_not_returned=True)
        self.verify_protected_metadata()
        for name, expected in self.recovery['source_bindings'].items():
            need(identity(path(name)) == expected, 'Historical protected file changed '+name)
        if self.mode == 'metrics':
            answer = frozen.load_sealed(self.output/'metrics.json')
            need(audit['started_by_kind']['eval_grasp'] == 359 and audit['returned_by_kind']['eval_grasp'] == 357
                 and audit['unknown'] == sorted(r['id'] for r in self.historical_unknowns)
                 and len(audit['engineering_retry_ids']) == 2 and len(audit['new_engineering_retry_ids']) == 1,
                 'Final complete recovery accounting differs')
            proof.update(scientific_metrics=identity(self.output/'metrics.json'),
                scientific_gates=answer['gates'], proposal_expand=answer['proposal_expand'], automatic_expansion=False,
                eligible_view=identity(self.calls), eligible_manifest=identity(self.calls.with_suffix('.manifest.json')),
                disclosure='Scientific metrics ledger is an eligible returned-event read view. '
                    'The complete three-source actual ledger retains two historical unknowns and two separately returned engineering retries.')
        frozen.sealed(self.log/('recovery_'+self.mode+('_'+str(self.scene) if self.scene is not None else '')+'.json'), proof)
        self.guard()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=('score', 'review', 'cold', 'metrics'), required=True)
    parser.add_argument('--scene', type=int, choices=SHARDS)
    args = parser.parse_args(); log = path('logs/joint_validation_v51r2/official')
    with frozen.exclusive(log):
        run = RecoveryEvaluation(args.config, args.mode, args.scene); completed = False; error = None
        try:
            run.run(); completed = True
        except BaseException:
            error = traceback.format_exc(); raise
        finally:
            frozen.write_once(log/('usage_'+args.mode+'_'+uuid.uuid4().hex+'.json'),
                dict(seconds=time.monotonic()-run.start, completed=completed, mode=args.mode, scene=args.scene,
                     error=error, complete_combined_ledger=run.audit()))


if __name__ == '__main__':
    main()
