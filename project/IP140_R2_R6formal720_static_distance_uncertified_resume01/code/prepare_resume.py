"""Materialize byte-identical sealed R6 inputs and evidence, without launching an engine."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,shutil,sys,os,msvcrt
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent;OLD=ROOT/'IP140_R2_R6formal720_raw_a_c0_paired01'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rd(p):return json.loads(p.read_bytes())
def must(ok,err):
    if not ok:raise ValueError(err)
def put(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(d,f,ensure_ascii=False,allow_nan=False,separators=(',',':'));f.write('\n')
def main():
    sm=OLD/'SHA256SUMS.txt';catalog=dict((s.split('  ',1)[1],s.split('  ',1)[0]) for s in sm.read_text(encoding='utf8').splitlines() if '  ' in s)
    require_unc=rd(OLD/'delivery/stop_evidence_verification.json')
    must(sha(OLD/'delivery'/require_unc['package'])==require_unc['sha256']=='e1b134491689539f24a4883a1bc562539e6108b5fae170c8372e284bf985a923','prior stop archive identity')
    for p in sorted((OLD/'workspaces').glob('*/lane.lock')):
        with p.open('rb') as f:
            msvcrt.locking(f.fileno(),msvcrt.LK_NBRLCK,1);f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
    for k,h in catalog.items():must(sha(OLD/k)==h,'old manifest drift '+k)
    verification=rd(OLD/'delivery/verification.json')
    must(sha(sm)==verification['local_sha256_manifest'] and len(catalog)==verification['local_file_count'],'old manifest binding')
    ledger=rd(OLD/'analysis/formal_ledger.json');stop=rd(OLD/'run/METHOD_STOP.json');terminal=rd(OLD/'sealed_intents.json')
    seven=[r['trial_id'] for r in ledger['rows'] if r['status']=='adjudicated'];unknown=[r['trial_id'] for r in ledger['rows'] if r['status']=='unknown_or_shared_failure'];pending=[r['trial_id'] for r in ledger['rows'] if r['status']=='unstarted']
    must(len(seven)==7 and unknown==['R6FORMAL-001'] and len(pending)==712 and not rd(OLD/'run/progress.json')['active'],'prior trial ownership')
    must(all(not (OLD/'run/jobs/method'/k).exists() for k in pending) and {x['trial_id'] for x in stop['received']}==set(seven),'duplicate prior run')
    must(len(terminal['files'])==720 and terminal['version']==OLD.name,'parent seal')
    copied={}
    def copy(src,dest=None):
        key=src.relative_to(OLD).as_posix();dest=dest or HERE/key
        k=dest.relative_to(HERE).as_posix()
        if k in copied:return
        if src.is_symlink() or src.is_junction() or not src.is_file():raise ValueError(k)
        h=catalog.get(key) or sha(src)
        dest.parent.mkdir(parents=True,exist_ok=True)
        with src.open('rb') as inp,dest.open('xb') as out:shutil.copyfileobj(inp,out,1<<20)
        must(sha(dest)==h,'byte copy '+k)
        copied[k]=dict(source=key,sha256=h)
    for s in ['sealed_intents.json','audit/resource_preflight.json','audit/scene_inventory.json','audit/input_preflight.json']:copy(OLD/s)
    for p in (OLD/'preregistration').glob('*.json'):copy(p)
    units=rd(OLD/'preregistration/actual_unit_inputs.json')['units']
    for val in units.values():
        for name,h in val['source_files'].items():
            must(catalog[name]==h,'unit seal '+name);copy(OLD/name)
        for name in ('observed_rgbd.npz','settled_state.json','receipt.json'):
            copy(OLD/'scenes'/val['scene']/name)
    for k,h in terminal['files'].items():must(catalog[k]==h,'intent hash '+k);copy(OLD/k)
    for tid in seven+unknown:
        for src in sorted((OLD/'trials'/tid).rglob('*')):
            if not src.is_file():continue
            if tid in unknown and src.name=='method_result.json':
                copy(src,HERE/'history'/tid/'method_result.json')
            else:copy(src)
        for src in sorted((OLD/'run/jobs/method'/tid).glob('*')):
            if tid in unknown and src.name=='exception.json':copy(src,HERE/'history'/tid/'exception.json')
            else:copy(src)
    for name in ('analysis/formal_ledger.json','run/METHOD_STOP.json','run/progress.json','audit/stop_forensics/native_bracket_audit.json','delivery/stop_evidence_verification.json'):
        copy(OLD/name,HERE/'history'/name)
    for name in ('common.py','codec.py','task_worker.py','method_entry.py','analyze.py','delivery.py','formal_dispatch.py'):
        copy(OLD/'code'/name,HERE/('code' if name not in ('analyze.py','delivery.py','formal_dispatch.py') else 'history/original_code')/name)
    for src in sorted((OLD/'implementation').glob('*.py')):
        copy(src,HERE/('history/original_code' if src.name=='platform_trial.py' else 'implementation')/src.name)
    put(HERE/'audit/bytewise_inheritance.json',dict(source_version=OLD.name,prior_full_manifest_sha256=sha(sm),prior_full_manifest_files=len(catalog),frozen_design_sha256=sha(OLD/'preregistration/design.json'),original_seal_sha256=sha(OLD/'sealed_intents.json'),inherited=copied))
    put(HERE/'protocol/continuation.json',dict(version=HERE.name,parent=OLD.name,created=datetime.now(timezone.utc).isoformat(),adjudicated_unchanged=seven,supplemental_only=unknown,allowed_once_in_original_order=pending,budget=720,remaining=712,new_method_budget=0,regenerated_inputs=0,offline_replay_before_new_dispatch=True,absolute_distance_tolerance_m=1e-9,unknown_integrity_stops=True,ablation_budget=0))
    print('verified parent files',len(catalog),'copied',len(copied),'pending',len(pending))
if __name__=='__main__':main()
