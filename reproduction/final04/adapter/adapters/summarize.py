"""Summarize fresh cold returns only; an absent native observation is not a zero."""
import json,hashlib,sys
from pathlib import Path
B=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_bytes()) if p.is_file() else None
def main(t):
    receipts=t/'cold_receipts';run=t/'p/IP140_R2_R6formal720_static_distance_uncertified_resume01'
    checks=[]
    for q in read(B/'metadata/cold_preregistration.json')['cases']:
        tid=q['trial_id'];d=receipts/tid;trial=run/'trials'/tid
        events=[json.loads(s) for s in (d/'model_invocations.jsonl').read_bytes().splitlines()] if (d/'model_invocations.jsonl').is_file() else []
        f,r,m=[read(d/(i+'_return.json')) for i in ('forward','rank','method')]
        engine=read(trial/'dynamic_api/summary.json')
        physical=engine['actual_native_step_calls'] if engine else (None if (trial/'dynamic_api').exists() else 0)
        returned=engine['returns'].get('stepSimulation',0) if engine else (None if (trial/'dynamic_api').exists() else 0)
        results=read(trial/'method_result.json');stat=read(trial/'static_preflight/path_result.json');ep=read(trial/'episode/result.json')
        checks.append(dict(trial_id=tid,expected=q['expected_detail'],cold_gates={'inference':bool(f),'ranking_and_E':bool(r),'method':bool(m)},model_calls={name:{k:sum(z['model']==name and z['event']==k for z in events) for k in ['entered','returned']} for name in ['P3','P5']},native_calls=physical,native_returns=returned,classification=m.get('classification') if m else None,selected=q['candidate'] if r else None,eligibles=r['eligible_count'] if r else None,grasp_success=results.get('grasp_success') if results else None,static_snapshots=stat.get('snapshots_completed') if stat else None,max_lift_m=ep.get('max_lift_m') if ep else None,final_hold_streak=ep.get('final_hold_streak') if ep else None))
    return dict(gate_passed=bool(read(receipts/'complete.json')) and all(all(c['cold_gates'].values()) for c in checks),complete_cases=sum(c['cold_gates']['method'] for c in checks),cases=checks,stop=read(receipts/'STOP.json'))
if __name__=='__main__':print(json.dumps(main(Path(sys.argv[1]).resolve()),indent=2,ensure_ascii=True))
