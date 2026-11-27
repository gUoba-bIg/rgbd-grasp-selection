"""Bind original smoke03–20 deliveries, frozen definitions, assets and per-slot references."""
from pathlib import Path
import json, hashlib, re, zipfile
from datetime import datetime, timezone
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PARENT=ROOT/'IP140_R1_smoke20_positive_clearance_face_scope_and_resume_four01'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_bytes())
def save(name, x):
    with (HERE/name).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def main():
    zip_path=next(PARENT.glob('*.zip'))
    frozen_zip='00ecc0dcccc60fe90702c11d948067a519ad3541f8b3a0066257b01769046cbc'
    assert sha(zip_path)==frozen_zip
    prev=read(PARENT/'history_snapshot_before.json')['files']
    history={}
    roots=[]
    for d in sorted(ROOT.glob('IP140_R1_smoke*')):
        match=re.match(r'IP140_R1_smoke(\d+)',d.name)
        if not d.is_dir() or not match or not 3<=int(match[1])<=20:continue
        roots.append(d.name)
        for f in sorted(d.rglob('*')):
            if not f.is_file():continue
            name=f.relative_to(ROOT).as_posix();st=f.stat()
            rec={'sha256':sha(f),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
            if name in prev and rec!=prev[name]:raise ValueError(name)
            history[name]=rec
    assert not (set(prev)-set(history))
    save('history_snapshot_before.json',dict(version=HERE.name,utc=datetime.now(timezone.utc).isoformat(),directories=roots,files=history))
    loc=read(PARENT/'static_input_identity.json')['local_files']
    selected={r:h for r,h in loc.items() if r.startswith(('slots/09/','slots/10/','slots/11/','slots/12/','model/')) or r in {
     'candidate_registry.json','external_assets18.json','asset_typed_schema.json','asset_schema19.py','asset_schema_builder.py','schema_replay19.py',
     'trajectory.py','speed_policy.py','structural_filter.py','mountpair.py','base_table_mount.py','aggregate_support.py','object_evaluator.py','object_episode.py','object_scene_static.py','damping_write.py'}}
    with zipfile.ZipFile(zip_path) as z:
        for name,h in selected.items():
            with z.open(name) as f: zipped=hashlib.file_digest(f,'sha256').hexdigest()
            if h!=zipped or h!=sha(HERE/name) or h!=sha(PARENT/name):raise ValueError('Frozen input '+name)
    save('inherited_byte_identities.json',dict(version=HERE.name,parent=PARENT.name,parent_zip_sha256=frozen_zip,files=selected,historical_files=len(history)))
if __name__=='__main__':main()
