"""Bind the new working copy to the sealed smoke19 artifacts before native access."""
import hashlib
import json
import zipfile
from pathlib import Path
from static_identity import HERE, ROOT, read, sha, write_once, now

PREVIOUS = ROOT / "IP140_R1_smoke19_asset_typed_zero_schema_and_resume_five01"

def main():
    archived = next(PREVIOUS.glob('*_*.zip'))
    expected_hash = Path(str(archived)+'.sha256').read_text(encoding='utf-8-sig').split()[0]
    if sha(archived) != expected_hash or expected_hash != 'bc7db85dab4ab27fc53730cc58c615535af05b7c6d8348d16429da40d0d661fb':
        raise ValueError('Prior review ZIP changed')
    old = read(PREVIOUS/'history_snapshot_before.json')['files']
    for name,h in old.items():
        path=ROOT/name
        if sha(path)!=h['sha256'] or path.stat().st_size!=h['bytes'] or path.stat().st_mtime_ns!=h['mtime_ns']:
            raise ValueError('Historical change: '+name)
    before=dict(old)
    for path in sorted(PREVIOUS.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            before[path.relative_to(ROOT).as_posix()]=dict(sha256=sha(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
    write_once(HERE/'history_snapshot_before.json',dict(version=HERE.name,utc=now(),files=before))
    local = read(PREVIOUS/'static_input_identity.json')['local_files']
    bindings={}
    for rel in sorted(local):
        if (rel.startswith('slots/') and not rel.startswith('slots/08/')) or rel.startswith('model/') or rel in ('external_assets18.json','candidate_registry.json',
            'mountpair_registration.json','base_table_registration.json','speed_policy.py','trajectory.py',
            'object_evaluator.py','aggregate_support.py','asset_schema19.py','asset_typed_schema.json','schema_replay19.py','asset_schema_builder.py'):
            path=HERE/rel
            if sha(path)!=local[rel] or sha(PREVIOUS/rel)!=local[rel]:
                raise ValueError('Copy differs from frozen parent: '+rel)
            bindings[rel]=local[rel]
    with zipfile.ZipFile(archived) as z:
        for rel,h in bindings.items():
            with z.open(rel) as member:
                if hashlib.file_digest(member,'sha256').hexdigest()!=h: raise ValueError('Parent ZIP binding '+rel)
    write_once(HERE/'inherited_byte_identities.json', dict(parent=PREVIOUS.name,parent_zip_sha256=expected_hash,
        verified_prior_files=len(before),files=bindings,scope='Candidates, registrations, trajectories, initial conditions, assets and control/evaluation definitions, unchanged byte-for-byte'))
if __name__=='__main__': main()
