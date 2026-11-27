"""Read-only parent retention and terminal state attestation. No R6 per-trial success data feed the design/score code."""
from pathlib import Path
from common import HERE, ROOT, sha, put, now, read, require
from freeze_sample import PARENT
from collections import Counter

def main():
    design = HERE/'preregistration/design.json'
    require(sha(design)==read(HERE/'preregistration/design_seal.json')['design_sha256'],'sample not sealed')
    manifest=PARENT/'SHA256SUMS.txt'
    require(sha(manifest)=='e39fa2d1bd03f79b7732569b7307044aba1303f4686cb0eb63edb1d575f2510f','R6 manifest identity')
    entries={}
    for raw in manifest.read_text(encoding='utf8').splitlines():
        h,k=raw.split('  ',1)
        require(k not in entries and len(h)==64,'parent manifest duplicate')
        entries[k]=h
    require(len(entries)==51239,'R6 archive manifest cardinality')
    proof=read(PARENT/'delivery/verification.json')
    archive=PARENT/proof['core_path']
    require(sha(archive)==proof['core_sha256']=='aed378d1487c6be5895978e895106be375c73d964be2c29cf62104297bca234d','parent package drift')
    allids={f'R6FORMAL-{i:03d}' for i in range(1,721)}
    sealed=read(PARENT/'sealed_intents.json');result_hashes={};sources={}
    def verify(p,expected):
        require(sha(p)==expected,'frozen R6 file mismatch '+str(p))
        sources[p.relative_to(ROOT).as_posix()]=expected
    for tid in sorted(allids):
        ik='trials/'+tid+'/intent.json';rk='trials/'+tid+'/method_result.json';jk='run/jobs/method/'+tid+'/return.json'
        require(sealed['files'][ik]==entries[ik],'parent intent seal')
        for k in (ik,rk,jk): verify(PARENT/k,entries[k])
        receipt=read(PARENT/jk)
        require(receipt['id']==tid and receipt['intent_sha256']==entries[ik] and receipt['method_result_sha256']==entries[rk] and receipt['completed'],'parent worker receipt')
        result_hashes[tid]=dict(intent=entries[ik],result=entries[rk],receipt=entries[jk])
    for name in ['analysis/formal_ledger.json','sealed_intents.json','run/progress.json','protocol/storage_resume01/progress.jsonl','run/method_complete.json','preregistration/actual_unit_inputs.json','preregistration/design.json','protocol/release.json','protocol/frozen_rule.json','preregistration/frozen_inputs.json']:
        verify(PARENT/name,entries[name])
    final=read(PARENT/'run/method_complete.json')
    last=__import__('json').loads((PARENT/'protocol/storage_resume01/progress.jsonl').read_bytes().splitlines()[-1])
    require(final['returned']==720 and final['terminal_input_guard'] and last['stage']=='formal_execution_closed' and last['completed']==720 and not last['active'] and not last['faults'] and last['unlaunched']==0,'R6 terminal state')
    for k,h in read(PARENT/'preregistration/frozen_inputs.json').items(): verify(ROOT/k,h)
    for k,h in read(PARENT/'protocol/release.json')['files'].items(): verify(PARENT/k,h)
    put(HERE/'audit/r6_provenance.json',dict(parent=PARENT.name,files=sources,all_720_opaque_hashes=result_hashes))
    put(HERE/'audit/r6_final_state.json',dict(version=PARENT.name,completed=720,planned=720,unknown=0,
        terminal_utc=last['utc'],terminal_checkpoint=last['latest_checkpoint'],terminal_record='protocol/storage_resume01/progress.jsonl',
        legacy_snapshot='run/progress.json',legacy_snapshot_completed=read(PARENT/'run/progress.json')['completed'],
        historical_progress_snapshot_preserved=True,manifest=sha(manifest),package=proof['core_sha256'],
        parent_results_recomputed=0,parent_files_written=0,r7_sampling_already_sealed=sha(design),
        verified_result_receipt_intent_triplets=len(allids),utc=now()))
    print('R6 720/720 closure and seals verified',flush=True)

if __name__=='__main__':
    main()
