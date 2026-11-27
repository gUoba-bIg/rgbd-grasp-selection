"""Bounded serialization recovery; original failed return remains unknown."""
import copy,json,time,uuid
import numpy as np
from scripts.p7_common_v31 import ROOT,LOG,BASE,event,verify,read,stat,sha,need
MANIFEST=ROOT/'configs/p7_score_recovery_v31r2.json'

def normalize_oracle(groups,scores,collisions):
 return dict(groups=[np.asarray(x).tolist() for x in groups],scores=[np.asarray(x).tolist() for x in scores],collisions=[np.asarray(x).tolist() for x in collisions])

def verify_recovery():
 c=verify();m=read(MANIFEST);need(sha(MANIFEST)==read(ROOT/'logs/P7/score_repair_v31r2/freeze.json')['sha256'],'Recovery contract changed')
 for p,s in m['protected'].items():need(stat(ROOT/p)==s,'Recovery source/evidence changed '+p)
 for p,s in read(ROOT/m['preservation_inventory'])['existing_files'].items():need(stat(ROOT/p)==s,'Previously committed P7 result changed '+p)
 need(c['call_limits']['upstream_oracle']==3 and m['additional_recovery_oracle_attempts']==1,'Unexpected oracle recovery scope')
 c=copy.deepcopy(c);c['call_limits']['upstream_oracle']=4
 return c

def recovery_oracle(fn,case):
 m=read(MANIFEST);rows=[json.loads(s) for s in (LOG/'calls.jsonl').read_text(encoding='utf-8').splitlines()];attempts=[r for r in rows if r['kind']=='upstream_oracle_intent'];returned={r['call_id'] for r in rows if r['kind']=='upstream_oracle_returned'}
 unknown={r['call_id'] for r in attempts}-returned;need(unknown==set(m['preserved_unknown_oracle_calls']),'New unknown oracle requires a fresh failure audit; no retry loop')
 need(len(attempts)<4,'Original3 plus exactly1 serialization-recovery attempt exhausted')
 original=read(ROOT/'configs/risk_calibration_v31.json');need(case['scene']==original['groups'][case['split']][0] and case['condition']=='clean' and case['frame']==64,'Recovery oracle outside original three cases')
 cid=uuid.uuid4().hex;event('upstream_oracle_intent',call_id=cid,case=case,recovery_contract='v31r2');start=time.monotonic();value=fn();event('upstream_oracle_returned',call_id=cid,seconds=time.monotonic()-start,recovery_contract='v31r2');return value,cid
