"""Interpret only the previous V1 report's exact, already-returned finite finger-clearance stop."""
from paths import *
from diagnostic_return_fields import CORNER, CLOSEST, gap_fields
import diagnostic_return_fields

def classify_if_known(returned, episode, ctx, tid):
    issues=set(returned.get("violations",[]))
    if not issues or not issues.issubset({CORNER,CLOSEST}):
        return None
    diagnostic_return_fields.STUDY=STUDY
    raw,last=diagnostic_return_fields.last_return(tid)
    need(raw["trial_id"]==tid and raw["telemetry_sha256"]==returned["telemetry_sha256"]
         and raw["violations"]==returned["violations"] and raw["step_returns"]==returned["step_returns"],
         "returned hard-stop identity mismatch")
    pads=tuple(ctx.links["ip140_"+side+"_pad"] for side in ("left","right"))
    proof=gap_fields(last,ctx.robot,pads)
    write("study/trials/"+tid+"/decoded_stop_fields.json",
          dict(utc=utc(),outcome_class=raw["outcome_class"],step=last["index"],phase=last["phase"],signed_fields=proof,
               frozen_episode_result_sha256=sha(episode/"result.json"),telemetry_sha256=raw["telemetry_sha256"],additional_steps=0))
    return "finite_opposing_pad_clearance_hard_stop"
