"""Zero-physics audit evidence guard. Does not import or launch a simulator."""
import json
import sys
from pathlib import Path
from sim_v52_common import ROOT, read, write, identity, relative, lock, now

BASE = ROOT/'logs/simulation_v52_dynamics_audit'
STATE = ROOT/'runtime/pipeline/simulation_v52_dynamics_audit.json'


def collect():
    protected = dict(read(ROOT/'logs/simulation_v52/historical_snapshot.json')['protected'])
    protected.update(read(ROOT/'logs/simulation_v52/final_review.json')['artifact_identities'])
    # New v52 delivery artifacts, every stored episode image/trace and the old ledgers.
    for directory in ('results/simulation_v52', 'logs/simulation_v52'):
        for path in (ROOT/directory).rglob('*'):
            if path.is_file():
                name = relative(path)
                current = identity(path)
                if name in protected:
                    assert protected[name] == current, name
                protected[name] = current
    return protected


def verify():
    protected = read(BASE/'before.json')['protected']
    changed = [p for p,v in protected.items() if not (ROOT/p).is_file() or identity(ROOT/p) != v]
    assert not changed, changed
    return len(protected)


def main():
    command = sys.argv[1]
    with lock(ROOT/'runtime/pipeline/execution.lock'), lock(ROOT/'runtime/pipeline/simulation_v52_dynamics_audit.lock'):
        if command == 'prepare':
            if (BASE/'before.json').exists():
                raise RuntimeError('Audit already prepared; do not replace evidence snapshot')
            protected = collect()
            write(BASE/'before.json', dict(at=now(), protected=protected,
                scope='Only static source/structure/old-log review; no new dynamics authorized',
                previous_actual_physics_episodes=48), once=True)
            write(STATE, dict(at=now(), phase='static_audit_in_progress',
                authorized_physics_steps=0, new_physics_steps=0, new_dynamics_tests=0,
                previous_v52_episodes=48, diagnostic_plan_execution_authorized=False,
                R4_authorized=False, automatic_expansion=False,
                all_tasks_percent_legacy=45, experiment_percent_legacy=56.25,
                expanded_scope_percent=None))
            print(json.dumps(dict(prepared=True, historical_artifacts=len(protected))))
        elif command == 'verify':
            print(json.dumps(dict(unchanged=verify(), new_physics_steps=0)))
        else:
            raise ValueError(command)


if __name__ == '__main__':
    main()
