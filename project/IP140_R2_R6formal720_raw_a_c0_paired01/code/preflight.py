"""Verify input identities and workers without opening an engine or running a method."""
import sys,ast,shutil
from collections import Counter
from common import *

def main():
    d=read(HERE/'preregistration/design.json');seal=read(HERE/'preregistration/design_seal.json')
    inputs=read(HERE/'preregistration/frozen_inputs.json');resources=read(HERE/'audit/resource_preflight.json')
    require(sha(HERE/'preregistration/design.json')==seal['design_sha256'] and sha(HERE/'preregistration/frozen_inputs.json')==seal['frozen_inputs_sha256'],'design provenance')
    require(all(sha(ROOT/k)==v for k,v in inputs.items()),'frozen dependency digest')
    sums=dict((t.split('  ',1)[1],t.split('  ',1)[0]) for t in (R4/'SHA256SUMS.txt').read_text(encoding='utf-8-sig').splitlines() if '  ' in t)
    gate='audit/zero_interface_admission.json';prev=read(R4/gate)
    require(sha(R4/gate)==sums[gate] and prev['passed'] and prev['method_trials']==prev['stepSimulation']==0,'archived original interface admission')
    require(len({u['unit_id'] for u in d['units']})==240 and Counter(t['method'] for t in d['trials'])=={'A':240,'C0':240,'Raw':240},'method allocation')
    source={}
    for p in sorted((HERE/'implementation').glob('*.py'))+sorted((HERE/'code').glob('*.py')):
        ast.parse(p.read_text(encoding='utf8'));source[p.relative_to(HERE).as_posix()]=sha(p)
    for name,ref in read(HERE/'audit/adapter_files.json').items():require(sha(R4/'implementation'/name)==ref['old_sha256'] and sha(HERE/'implementation'/name)==ref['new_sha256'],'adapter digest')
    lanes=sorted((HERE/'workspaces').iterdir());require(len(lanes)==5 and len({p.stat().st_ino for p in lanes})==5,'workspace identities')
    for p in lanes:
        require((p/'data').resolve()==ROOT/'data' and (p/PLATFORM.name).resolve()==PLATFORM and (p/'workspace.json').is_file(),'workspace asset identity')
    require(resources['admitted'] and shutil.disk_usage(HERE).free>resources['space_reserve_bytes']+resources['estimated_bytes_legacy_case_mix']+12*2**30,'formal storage')
    require(not any((HERE/'trials').glob('*/method_result.json')) and not (HERE/'sealed_intents.json').exists(),'outcome before preflight')
    put(HERE/'audit/input_preflight.json',dict(passed=True,utc=now(),method_trials=0,stepSimulation=0,phase='formal scene and input construction only',original_end_to_end_gate=sha(R4/gate),design_sha256=seal['design_sha256'],frozen_files=len(inputs),resource_sha256=sha(HERE/'audit/resource_preflight.json'),source_files=source,workspaces=[str(p) for p in lanes],conservative_extra_bytes=12*2**30,live_free_bytes=shutil.disk_usage(HERE).free,formal_intents_closed=False))
if __name__=='__main__':main()
