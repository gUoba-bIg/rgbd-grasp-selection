"""Full bounded P2 asset/camera/split audit, separate from scientific performance."""
import os,sys,json,time,concurrent.futures,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v24 import ROOT,read,once,atomic,stat,sha,need,now
from scripts.p23_common_v26 import verify as verify_data,DATA,CONFIG
LOG=ROOT/'logs/P2P3/data_audit_v27';OUT=ROOT/'results/P2P3/data_audit_v27'

def freeze():
 p=LOG/'freeze.json'
 if p.exists():return read(p)
 c=dict(at=now(),data_config_sha256=sha(CONFIG),translation_m=.0001,rotation_angle_rad=.001,rigid_tolerance=1e-5,depth_factor=1000.,image_shape=[720,1280],workers=4,max_seconds=7200,tolerance_sample_indices='0,middle,last per present object; all 300x12x4 orientations',sources={q.relative_to(ROOT).as_posix():stat(q) for q in [Path(__file__),ROOT/'src/grasp_prerequisite_data_v27.py',ROOT/'docs/P2来源许可与最接近方法对照_20261030.txt']})
 once(p,c);return c

def scene_audit(scene,c):
 import numpy as np
 from PIL import Image
 from scipy.io import loadmat
 from scipy.spatial.transform import Rotation
 out=OUT/'scenes'/f'{scene:04d}.json'
 if out.exists():return read(out)
 d=DATA/'scenes'/f'scene_{scene:04d}'/'realsense';ids=sorted(map(int,(d.parent/'object_id_list.txt').read_text().split()))
 poses=np.load(d/'camera_poses.npy');align=np.load(d/'cam0_wrt_table.npy');K=np.load(d/'camK.npy')
 def rigid(a):
  need(np.isfinite(a).all() and a.shape[-2:]==(4,4),'Nonfinite/invalid rigid transform')
  rot=a[...,:3,:3];need(np.max(np.abs(np.swapaxes(rot,-1,-2)@rot-np.eye(3)))<=c['rigid_tolerance'] and np.max(np.abs(np.linalg.det(rot)-1))<=c['rigid_tolerance'],'Rigid rotation invalid')
  need(np.max(np.abs(a[...,3,:]-[0,0,0,1]))<=1e-7,'Rigid homogeneous row invalid')
 need(poses.shape==(256,4,4),'Camera pose coverage');rigid(poses);rigid(align)
 valid_ratios=[];foreground_ratios=[];maxtrans=maxangle=0.;objects_pixels={i:0 for i in ids}
 for frame in range(256):
  f=f'{frame:04d}';rgb=np.asarray(Image.open(d/'rgb'/(f+'.png')));depth=np.asarray(Image.open(d/'depth'/(f+'.png')));label=np.asarray(Image.open(d/'label'/(f+'.png')));m=loadmat(d/'meta'/(f+'.mat'))
  need(rgb.shape==(720,1280,3) and depth.shape==label.shape==(720,1280),'Image layout differs');need(depth.dtype==np.uint16 and np.issubdtype(label.dtype,np.integer),'Depth/label type differs')
  factor=float(m['factor_depth'].item());intrinsic=m['intrinsic_matrix'];mi=(m['cls_indexes'].reshape(-1)-1).astype(int).tolist();mp=m['poses']
  need(factor==c['depth_factor'] and np.allclose(K,intrinsic,rtol=0,atol=1e-8),'Depth unit or camera K differs')
  need(K.shape==(3,3) and np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0 and np.array_equal(K[2],[0,0,1]),'Invalid intrinsics')
  need(sorted(mi)==ids and len(set(mi))==len(mi) and mp.shape==(3,4,len(ids)),'Object pose scope differs')
  labs,counts=np.unique(label,return_counts=True);need(set(labs)<=set([0]+[i+1 for i in ids]),'Mask identity absent from scene')
  for x,n in zip(labs,counts):
   if x:objects_pixels[int(x)-1]+=int(n)
  raw=(d/'annotations'/(f+'.xml')).read_bytes();need(b'<!DOCTYPE' not in raw.upper() and b'<!ENTITY' not in raw.upper(),'External XML declaration refused');tree=ET.fromstring(raw);seen=[]
  for obj in tree:
   oi=int(obj.findtext('obj_id'));seen.append(oi);xyz=np.fromstring(obj.findtext('pos_in_world'),sep=' ');q=np.fromstring(obj.findtext('ori_in_world'),sep=' ')
   need(q.shape==(4,) and np.isfinite(q).all() and abs(np.linalg.norm(q)-1)<=.0002,'XML quaternion invalid');R=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();j=mi.index(oi);a=np.eye(4);a[:3]=mp[:,:,j];rigid(a)
   dt=float(np.max(np.abs(xyz-mp[:,3,j])));cos=np.clip((np.trace(R.T@mp[:,:3,j])-1)/2,-1,1);angle=float(np.arccos(cos));maxtrans=max(maxtrans,dt);maxangle=max(maxangle,angle)
   need(dt<=c['translation_m'] and angle<=c['rotation_angle_rad'],'XML/MAT coordinate mismatch')
  need(sorted(seen)==ids,'XML objects differ')
  valid=depth>0;valid_ratios.append(float(valid.mean()));foreground_ratios.append(float((label>0).mean()));need(valid.sum()>0,'Empty depth frame')
 row=dict(scene=scene,frames=256,object_ids=ids,object_mask_pixels=objects_pixels,mean_valid_depth_fraction=float(np.mean(valid_ratios)),min_valid_depth_fraction=min(valid_ratios),mean_foreground_fraction=float(np.mean(foreground_ratios)),max_XML_MAT_translation_m=maxtrans,max_XML_MAT_rotation_rad=maxangle)
 once(out,row);return row

def tolerance_audit(ids):
 import numpy as np
 from src.grasp_prerequisite_data_v27 import mapped_npz
 rows=[]
 for obj in ids:
  out=OUT/'tolerance'/f'{obj:03d}.json'
  if out.exists():rows.append(read(out));continue
  p=str(DATA/'grasp_label'/f'{obj:03d}_labels.npz');points=mapped_npz(p,'points');scores=mapped_npz(p,'scores');offsets=mapped_npz(p,'offsets');tol=np.load(DATA/'tolerance'/f'{obj:03d}_tolerance.npy',mmap_mode='r')
  need(points.shape[1:]==(3,) and scores.shape==(len(points),300,12,4) and offsets.shape==scores.shape+(3,) and tol.shape==scores.shape,'Dense label dimensions differ')
  sample=sorted(set([0,len(points)//2,len(points)-1]));error=0.
  for index in sample:
   d=np.linalg.norm(points-points[index],axis=1);expected=np.zeros((300,12,4),np.float32)
   for rad in [0.001*x for x in range(51)]:
    s=scores[d<=rad];m=((s>0)&(s<=.55)).mean(axis=0)>=.8
    if not m.any():break
    expected[m]=rad
   err=float(np.max(np.abs(tol[index]-expected)));error=max(error,err);need(err<=1e-7,'Official tolerance sample differs '+str(obj))
  row=dict(object_id=obj,points=len(points),sample_indices=sample,maximum_error=error,all_elements_recomputed=False);once(out,row);rows.append(row)
 return rows

def run():
 c=freeze();need(sha(CONFIG)==c['data_config_sha256'],'Acquisition config changed')
 for p,s in c['sources'].items():need(stat(ROOT/p)==s,'Frozen audit source changed '+p)
 need((ROOT/'logs/P2P3/prerequisites_v26/acquisition_complete.json').exists(),'Data still acquiring')
 data=verify_data();start=time.monotonic();rows=[json.loads(s) for s in (ROOT/'logs/P2P3/prerequisites_v26/member_receipts.jsonl').read_text(encoding='utf-8').splitlines()];paths={r['path'] for r in rows}
 need(set(data['selected'])<=paths and len(paths)==len(rows),'Data member coverage differs')
 def check(r):
  need(time.monotonic()-start<c['max_seconds'],'Audit time budget');p=DATA/r['path'];need(sha(p)==r['sha256'] and p.stat().st_size==r['bytes'] and p.stat().st_mtime_ns==r['mtime_ns'],'Acquired member changed '+r['path'])
 with concurrent.futures.ThreadPoolExecutor(max_workers=c['workers']) as pool:
  for _ in pool.map(check,rows):pass
  scenes=list(pool.map(lambda scene:scene_audit(scene,c),range(100)))
 ids=sorted(set(i for s in scenes for i in s['object_ids']));tolerance=tolerance_audit(ids)
 split=data['split'];coverage={}
 for group in ['train','validation','calibration']:
  subset=[s for s in scenes if s['scene'] in split[group]];coverage[group]=dict(scenes=len(subset),frames=256*len(subset),object_ids=sorted(set(i for s in subset for i in s['object_ids'])),mean_valid_depth_fraction=sum(s['mean_valid_depth_fraction'] for s in subset)/len(subset),mean_foreground_fraction=sum(s['mean_foreground_fraction'] for s in subset)/len(subset))
 result=dict(at=now(),status='passed',files_verified=len(rows),bytes_verified=sum(r['bytes'] for r in rows),frames_verified=25600,official_test_frames_opened=0,split_coverage=coverage,unknown_network_calls=read(ROOT/'logs/P2P3/prerequisites_v26/acquisition_complete.json')['unknown_network_calls'],tolerance_sample_check=tolerance,seconds=time.monotonic()-start,scientific_model_performance_not_tested=True)
 once(OUT/'result.json',result);print(json.dumps(dict(status='passed',frames=25600,seconds=result['seconds'])),flush=True)

if __name__=='__main__':
 if '--freeze' in sys.argv:freeze();print('audit frozen')
 else:run()
