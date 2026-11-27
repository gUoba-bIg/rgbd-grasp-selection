"""Run the same cross-condition measurement for the V1 R7 arms and their paired R6 counterparts."""
from common_v1 import *
import numpy as np
import rank_stability as measurement
import history as aggregation

ABLATIONS=("C0_no_instance","C0_depth_only","C0_warp_only")

def cohort(root_dir,uid):
    d=root_dir/uid
    with np.load(d/"pool.npz",allow_pickle=False) as pool:rows=pool["candidate_camera"].copy()
    selection=read(d/"selection.json")["selected"];ranks={};scores={}
    for m in ABLATIONS:
        r=read(d/("ranking_"+m+".json"));need(r["unit_id"]==uid and r["shared_pool_sha256"]==sha(d/"pool.npz"),"R7 unit pool")
        scores[m]=r["scores_by_pool_index"];ranks[m]=r["complete_order"]
    return rows,scores,ranks,selection
def main():
    base=ROOT/R7NAME/"units";ro=ROOT/R6NAME/"units"
    units=sorted(d.name for d in base.iterdir() if d.is_dir())
    scenes=sorted({u.rsplit("-",1)[0] for u in units})
    need(len(units)==80 and len(scenes)==40 and all(s+"-"+m in units for s in scenes for m in ("clean","joint")),"R7 full clean/joint pairs")
    out=[]
    # Measurement enumerates its method labels externally; it never feeds a score to a model or ranker.
    for labels,readunit,folder in ((METHODS,lambda d,u:measurement.load_unit(d/u),ro),(ABLATIONS,cohort,base)):
        measurement.METHODS=labels
        for scene in scenes:
            a,b=readunit(folder,scene+"-clean"),readunit(folder,scene+"-joint")
            result,matching=measurement.compare(a,b,scene,"joint");out.extend(result)
    aggregation.table("analysis/six_methods_80units_clean_joint.csv",out)
    aggregation.METHODS=(*METHODS,*ABLATIONS)
    write("analysis/six_methods_80units_rank_stability.json",dict(cohort="identical 40 base scenes, each clean and joint",
          base_scenes=scenes,method_metrics=aggregation.stability_summary(out),clusters=40))
if __name__=="__main__":main()
