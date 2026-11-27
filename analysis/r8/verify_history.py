"""Read both historical manifests end to end. Write evidence only in R8."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

version = Path(__file__).resolve().parents[1]
root = version.parent
identity = json.loads((version / 'tables/audit.json').read_text('utf-8'))
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

versions = []
for name, meta in identity['source_versions'].items():
    parent = root / name
    manifest = parent / 'SHA256SUMS.txt'
    assert sha(manifest) == meta['manifest_sha256']
    paths = set()
    byte_count = 0
    for line in manifest.read_text('utf-8-sig').splitlines():
        expected, rel = line.split('  ', 1)
        item = (parent / rel).resolve()
        assert item.is_relative_to(parent) and rel not in paths, rel
        paths.add(rel)
        assert sha(item) == expected, (name, rel)
        byte_count += item.stat().st_size
    archive = parent / meta['archive_path']
    assert sha(archive) == meta['archive_sha256']
    versions.append(dict(directory=name, manifest='SHA256SUMS.txt',
                         manifest_sha256=meta['manifest_sha256'], files_verified=len(paths),
                         bytes_verified=byte_count, archived_core=meta['archive_path'],
                         core_sha256=meta['archive_sha256'], archive_bytes=archive.stat().st_size))
out = dict(status='all_manifest_members_match', scope='all listed source files plus signed-off light archive identities',
           checked_utc=datetime.now(timezone.utc).isoformat(), source_versions=versions,
           original_files_written=0, new_physics_steps=0)
with (version / 'evidence/history_seal.json').open('x', encoding='utf-8') as f:
    json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
