"""Compare all planned commands with the archived gzip for this exact registration."""
from pathlib import Path
import gzip, hashlib, json, os, sys
V = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(V))
from trajectory import validate_plan, reference, PHASES, DT
from speed_policy import phase_speed
from batch_context import SLOT, SLOT_DIR

def main():
    if SLOT not in (10,11,12) or Path.cwd().resolve()!=V.parent:
        raise ValueError("Wrong review scope")
    plan=validate_plan()
    path=SLOT_DIR/"frozen_trajectory_commands.jsonl.gz"
    n=0
    with gzip.open(path,"rt",encoding="utf8") as f:
        for phase,count in PHASES:
            for tick in range(1,count+1):
                n+=1
                row=json.loads(next(f))
                q,dq,finger=reference(plan,phase,tick)
                assert (row["planned_index"],row["phase"],row["phase_tick"],row["dt_s"])==(n,phase,tick,DT)
                assert row["arm_target_q_rad"]==q.tolist() and row["arm_reference_dq_rad_s"]==dq.tolist()
                assert row["finger_reference_q_m"]==finger and row["single_finger_command_speed_m_s"]==phase_speed(phase)
        assert n==4728 and f.readline()==""
    with (V/"review"/f"frozen_commands_{SLOT}.json").open("x",encoding="utf8") as out:
        json.dump(dict(slot=SLOT,candidate_id=plan["candidate_id"],matched_commands=n,phases=dict(PHASES),
                       command_gzip_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),physics_steps=0,passed=True),out,indent=2)
    print(SLOT,n)

if __name__=="__main__":main()
