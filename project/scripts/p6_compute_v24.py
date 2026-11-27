"""P6 production geometry, no model inference or training."""
import os,sys,argparse
from pathlib import Path
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v24 import *
from scripts.pipeline import project_lock
from src import p6_image_geometry_v24 as g
from src import geometry_propagation_v19 as inherited
import numpy as np

def run(screen=False):
 c=verify();endpoint=BASE/('preflight100.json' if screen else 'geometry_complete.json')
 if endpoint.exists():print('P6 geometry SKIP: complete');return
 b=Budget(c,'geometry_preflight' if screen else 'geometry_production');rows=[];outcome='failed'
 try:
  for r,d in case_order(c)[:100 if screen else 800]:
   stem=name(r,d);rp=BASE/'geometry'/(stem+'.json');p=rp.with_suffix('.npz')
   if rp.exists():row=read(rp);need(sha(p)==row['cache_sha256'],'Geometry cache changed');rows.append(row);continue
   b.guard();cid=b.intent('corruption_replay',case=stem);s=g.source(c,r,d);b.returned('corruption_replay',cid,case=stem);instances,q=g.load(c,r,d);a=g.build(c,s,instances,q)
   qp=g.quality(r,d)
   if not qp.is_relative_to(OLD):need(read(qp.with_suffix('.json'))['observed_depth_sha256']==array_sha(s['observed']),'Observed depth differs from quality inference')
   else:
    with np.load(qp,allow_pickle=False) as old:need(np.array_equal(old['q_residual'][0],s['observed']-s['reference']),'Reused quality perturbation differs')
   for j in [0,len(a['flat_index'])//2,-1]:
    nominal=[*a['original_uv'][j],a['depth_observed'][j]];check=inherited.jacobian_diagnostics(nominal,s['K']);need(check['passed'],'Actual camera finite difference failed');exact=inherited.propagate_gaussian(nominal,s['K'],np.diag([0.,0.,a['sigma'][j]**2]),[0.,0.,a['sigma'][j]])['first_order_covariance'];need(np.allclose(exact,a['covariance'][j],rtol=1e-12,atol=1e-16),'R19 propagation mismatch')
   p.parent.mkdir(parents=True,exist_ok=True);need(not p.exists(),'Uncommitted geometry cache requires review');np.savez_compressed(p,**a);row=dict(scene=r['scene'],frame=r['frame'],condition=d,cache_sha256=sha(p),quality_sha256=sha(qp),mask_sha256=sha(g.fixed(r,d)),observed_depth_sha256=array_sha(s['observed']),K=s['K'].tolist(),factor_depth=s['factor'],coverage_counts=dict(original_valid=int(s['original_valid'].sum()),observed_valid=int(s['observed_valid'].sum()),eligible=int(s['eligible'].sum()),original_clipped=int((s['raw']>2).sum()),selected_main=int(a['main'].sum()),selected_with_boundary=len(a['flat_index'])),metrics=g.stats(a));once(rp,row);rows.append(row)
   if len(rows)%100==0:status('geometry',completed=len(rows),total=800);print('geometry '+str(len(rows))+'/800',flush=True)
  if screen:once(endpoint,dict(passed=True,frames=100,coordinate_quality_and_R19_checks_passed=True,not_scientific_adoption=True))
  else:
   summary=g.aggregate(rows);once(BASE/'summary.json',summary);once(endpoint,dict(frames=800,model_calls=0,positive_control_passed=summary['positive_control_passed'],summary_sha256=sha(BASE/'summary.json')))
  outcome='passed'
 finally:b.save(outcome)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--screen',action='store_true');a=ap.parse_args()
 with project_lock(ROOT):run(a.screen)
