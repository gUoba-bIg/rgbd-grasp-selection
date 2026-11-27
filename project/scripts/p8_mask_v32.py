"""Shared segmentation maps evaluated once; independent decoding and AP review."""
import os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p8_common_v32 import *
def frame(c,r,independent):
 import numpy as np
 from PIL import Image
 from scripts import accept_mask_ap_v17 as ref,evaluate_mask_ap_v17 as official,p4_complete_evaluation_v22r2 as inherited
 from scripts.extract_mask_scores_v17 import decode_with_scores
 with np.load(oldobservations(r)) as a:
  heads={k:a[k] for k in ['foreground_probability','center_probability','offsets']}
  if independent:
   v=ref.independent_decode(**heads,decoder=c['decoder']);pred,scores=v['instances'],v['retained_instances']
  else:pred,_,scores,_=decode_with_scores(heads,**c['decoder'])
  need(np.array_equal(pred,a['instances']),'P8 frozen segmentation decode differs')
 high=inherited.indexed_resize(pred,720,1280) if independent else inherited.high_resolution(pred);p=ROOT/c['data_root']/f"scenes/scene_{r['scene']:04d}/realsense/label/0064.png";need(stat(p)==c['mask_inputs'][p.relative_to(ROOT).as_posix()],'Mask GT changed')
 with Image.open(p) as im:gt=np.asarray(im)
 pids=[int(x) for x in np.unique(high) if x>0];gids=[int(x) for x in np.unique(gt) if x>0];image_id=r['scene']*1000+64;ann=[];dt=[]
 for gid in gids:
  m=gt==gid;yy,xx=np.nonzero(m);ann.append(dict(id=len(ann)+1,image_id=image_id,category_id=1,area=int(m.sum()),iscrowd=0,segmentation=official.encode_mask(m),bbox=[int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)]))
 for row in scores:dt.append(dict(image_id=image_id,category_id=1,score=row['score'],segmentation=official.encode_mask(high==row['instance_id'])))
 overlap=ref.dense_overlap(high,gt,pids,gids)['iou'];matching=ref.independent_image_matching(image_id,list(range(1,len(ann)+1)),list(range(1,len(dt)+1)),[x['score'] for x in scores],overlap)
 return dict(gt=dict(info={},images=[dict(id=image_id,height=720,width=1280)],categories=[dict(id=1,name='object')],annotations=ann),detections=dt),[matching]
def run(review=False):
 import numpy as np
 from scripts.evaluate_mask_ap_v17 import official_evaluation
 from scripts.p4_complete_evaluation_v22r2 import primary,audit_ap
 c=verify();b=Budget(c,'mask_review' if review else 'mask');rows=[]
 for n,r in enumerate([v for v in cases(c) if v['split']=='validation']):
  b.guard();inputs,matching=frame(c,r,review);p=BASE/'masks'/(key(r)+'.json')
  if not p.exists():
   need(not review,'Missing mask AP');value,cid=invoke('mask_AP',lambda:official_evaluation(inputs),case=r);sealed(p,dict(value=value,call_id=cid))
  value=load(p)['value'];audit_ap(value,matching);rows.append(dict(**r,**primary(value)));status('mask_review' if review else 'mask_evaluation',completed=n+1,total=75)
 summary=dict(rows=rows,condition_AP={k:float(np.mean([r['AP'] for r in rows if r['condition']==k])) for k in c['conditions']},clean_difference=0.,maps_shared_identically_by_all_eight_groups=True,interpretation='zero intergroup difference by design, not segmentation improvement')
 if review:need(load(BASE/'mask_summary.json')==summary,'Mask summary differs');sealed(LOG/'mask_readonly.json',dict(passed=True,frames=75,independent_decode_overlap_matching_primary_AP_arrays=True,new_official_calls=0))
 else:sealed(BASE/'mask_summary.json',summary)
 b.save('complete')
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run('--review' in sys.argv)
