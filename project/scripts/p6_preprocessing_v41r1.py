"""One bounded synthetic diagnostic through the unchanged v40 disk loader."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='2'
import sys,json,time,uuid,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import numpy as np
from PIL import Image
from scipy.io import loadmat,savemat
from scripts.p6_correspondence_v40 import load_frame,peak_rss,runtime_identity
from src import correspondence_v40 as core
from scripts.p6_background_v40 import verify as verify_old,completed_verify
from scripts.p5_fusion_delivery_v38 import read,sha,stat,atomic,need
from scripts.pipeline import project_lock
LOG=ROOT/'logs/P6/preprocessing_v41r1';OUT=ROOT/'results/P6/preprocessing_v41r1'
CONF=ROOT/'configs/p6_preprocessing_v41r1.json';INPUT=ROOT/'runtime/fixtures/p6_preprocessing_v41r1'
PROTOCOL=ROOT/'docs/P6输入几何一致性诊断协议_v41r1.txt'
K=np.array([[927.17,0,651.32],[0,927.37,349.62],[0,0,1.]])
NAMES=['coordinate_x','coordinate_y','plane_validity','shared_edges_p00','shared_edges_p10','shared_edges_p01','shared_edges_p12','shared_edges_native']
CAPS=dict(real_png_headers=240,real_meta_reads=80,synthetic_loader=8,synthetic_estimator=5,synthetic_xyz=2)

def event(event_type,**value):
    with (LOG/'calls.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(event=event_type,**value),ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())

def audit():
    starts={};ends={};keys=set()
    if (LOG/'calls.jsonl').exists():
        for line in (LOG/'calls.jsonl').read_text(encoding='utf-8').splitlines():
            e=json.loads(line);cid=e['id']
            need(e['kind'] in CAPS,'Undeclared call kind')
            if e['event']=='started':
                need(cid not in starts and (e['kind'],e['key']) not in keys,'Duplicate intent/key')
                starts[cid]=e;keys.add((e['kind'],e['key']))
            else:
                need(cid in starts and cid not in ends and e['event']=='returned','Unmatched return')
                need((e['kind'],e['key'])==(starts[cid]['kind'],starts[cid]['key']),'Changed return identity')
                ends[cid]=e
    return dict(counts={k:sum(e['kind']==k for e in ends.values()) for k in CAPS},unknown=sorted(set(starts)-set(ends)),
                returned_keys={k:sorted(e['key'] for e in ends.values() if e['kind']==k) for k in CAPS},
                synthetic_fit=0,real_image_estimator=0,real_image_pixel_decodes=0,neural_forward=0,official_ap=0)

def verify_calls(ledger):
    expected={k:[] for k in CAPS};data=read(ROOT/'logs/P6/correspondence_v40/data_audit.json')
    for group,records in data['records'].items():
        for r in records:
            key=f'{group}/{r["scene"]}/{r["frame"]}'
            expected['real_meta_reads'].append(key)
            expected['real_png_headers'].extend(key+'/'+kind for kind in ('rgb','depth','label'))
    expected['synthetic_loader']=NAMES
    expected['synthetic_estimator']=[n for n in NAMES if n.startswith('shared_edges')]
    expected['synthetic_xyz']=['coordinate_x','coordinate_y']
    need(ledger['counts']==CAPS and not ledger['unknown'],'Incomplete diagnostic ledger')
    need(ledger['returned_keys']=={k:sorted(v) for k,v in expected.items()},'Actual call scope differs')

class Budget:
    def __init__(self):self.start=time.monotonic();self.peak=0;self.counts={k:0 for k in CAPS}
    def guard(self):
        self.peak=max(self.peak,peak_rss());current=time.monotonic()-self.start;used=2.0+current
        size=sum(p.stat().st_size for d in (LOG,OUT,INPUT) if d.exists() for p in d.rglob('*') if p.is_file())
        atomic(LOG/'usage.json',dict(seconds=used,current_attempt_seconds=current,prior_attempt_conservative_charge_seconds=2.0,peak_rss_bytes=self.peak,output_bytes=size,counts=self.counts))
        need(used<120 and self.peak<=1024**3 and size<=64*1024**2,'Diagnostic resource cap')
    def invoke(self,kind,key,fn):
        self.guard();need(self.counts[kind]<CAPS[kind],'Call cap '+kind)
        cid=uuid.uuid4().hex;event('started',id=cid,kind=kind,key=key)
        answer=fn();event('returned',id=cid,kind=kind,key=key)
        self.counts[kind]+=1;self.guard();return answer

def nearest_indices(width,height):
    # Independent integer arithmetic, not core.xyz_at or a second Pillow resize.
    x=np.array([((2*u+1)*width)//1280 for u in range(640)],np.int64)
    y=np.array([((2*v+1)*height)//960 for v in range(480)],np.int64)
    return x,y

def triangle_resize(profile,out_n):
    profile=np.asarray(profile,float);scale=len(profile)/out_n;radius=max(1.,scale);out=[]
    for j in range(out_n):
        centre=(j+.5)*scale-.5
        lo=max(0,int(np.floor(centre-radius)));hi=min(len(profile),int(np.ceil(centre+radius))+1)
        idx=np.arange(lo,hi);weight=np.maximum(0,1-np.abs(idx-centre)/radius)
        out.append(float(weight@profile[idx]/weight.sum()))
    return np.array(out)

def fixtures(name):
    native=name=='shared_edges_native';w,h=(640,480) if native else (1280,720)
    yy,xx=np.indices((h,w));labels=((xx+3*yy)%251).astype(np.uint8)
    rgb=np.zeros((h,w,3),np.uint8);depth=np.full((h,w),1000,np.uint16);factor=1000.;boxes=[]
    kk=K.copy()
    if native:
        kk[0,0]/=2;kk[1,1]/=1.5;kk[0,2]=(K[0,2]+.5)/2-.5;kk[1,2]=(K[1,2]+.5)/1.5-.5
    if name.startswith('coordinate'):
        axis=xx if name.endswith('x') else yy;length=w if name.endswith('x') else h
        depth=(1000+axis).astype(np.uint16);factor=4000.
        rgb[:]=np.rint(axis/(length-1)*255).astype(np.uint8)[...,None]
    elif name=='plane_validity':
        depth[100:180,200:360]=0;depth[400:530,800:1050]=2500;rgb[:]=100
    else:
        phase={'shared_edges_p00':(0,0),'shared_edges_p10':(1,0),'shared_edges_p01':(0,1),'shared_edges_p12':(1,2),'shared_edges_native':(0,0)}[name]
        rgb[:]=40;depth[:]=800;labels[:]=0
        for index,(x0,x1,y0,y1,color,z) in enumerate([(193,497,113,331,160,1200),(673,941,383,599,220,1600),(1031,1197,83,277,110,1000)],1):
            x0=int(x0*w/1280)+phase[0];x1=int(x1*w/1280)+phase[0];y0=int(y0*h/720)+phase[1];y1=int(y1*h/720)+phase[1]
            rgb[y0:y1,x0:x1]=color;depth[y0:y1,x0:x1]=z;labels[y0:y1,x0:x1]=index
            boxes.append([x0,x1,y0,y1])
    folder=INPUT/name;folder.mkdir(parents=True,exist_ok=False)
    Image.fromarray(rgb).save(folder/'rgb.png');Image.fromarray(depth).save(folder/'depth.png');Image.fromarray(labels).save(folder/'label.png')
    savemat(folder/'meta.mat',dict(intrinsic_matrix=kk,factor_depth=np.array([[factor]])))
    record=dict(files={k:f'{name}/{k}'+('.mat' if k=='meta' else '.png') for k in ('rgb','depth','label','meta')})
    return dict(rgb=rgb,depth=depth,labels=labels,factor=factor,K=kk,boxes=boxes,record=record)

def expected_geometry(frame,xx,yy,budget,name):
    h,w=frame['depth'].shape;vv,uu=np.indices((h,w));uv=np.stack([uu.ravel(),vv.ravel()],1)
    z=frame['depth'].ravel().astype(np.float64);sx=xx.ravel();sy=yy.ravel();k=frame['K']
    actual=budget.invoke('synthetic_xyz',name,lambda:core.xyz_at(uv,z,k,frame['original_size'],(w,h)))
    expected=np.stack([(sx-k[0,2])*z/k[0,0],(sy-k[1,2])*z/k[1,1],z],1)
    projected=np.stack([actual[:,0]*k[0,0]/actual[:,2]+k[0,2],actual[:,1]*k[1,1]/actual[:,2]+k[1,2]],1)
    source=np.stack([sx,sy],1)
    return dict(max_xyz_error_m=float(np.max(np.abs(actual-expected))),max_source_reprojection_error_px=float(np.max(np.abs(projected-source))))

def edge_positions(frame,source):
    h,w=frame['depth'].shape;ow,oh=frame['original_size'];sx,sy=ow/w,oh/h;rows=[]
    def centroid(profile,expected):
        weights=np.abs(np.diff(np.asarray(profile,float)));positions=np.arange(len(weights))+.5
        mask=np.abs(positions-expected)<=4
        need(weights[mask].sum()>0,'Missing fixed synthetic edge')
        return float(positions[mask]@weights[mask]/weights[mask].sum())
    for i,(x0,x1,y0,y1) in enumerate(source['boxes']):
        ry=int(((y0+y1)/2+.5)/sy-.5);rx=int(((x0+x1)/2+.5)/sx-.5)
        for axis,edge in [('x',x0),('x',x1),('y',y0),('y',y1)]:
            scale=sx if axis=='x' else sy;expected=edge/scale-.5
            color=frame['rgb'][ry,:,0] if axis=='x' else frame['rgb'][:,rx,0]
            depth=frame['depth'][ry,:] if axis=='x' else frame['depth'][:,rx]
            a=centroid(color,expected);b=centroid(depth,expected)
            rows.append(dict(box=i,axis=axis,source_boundary_index=edge-.5,ideal_output_boundary=expected,rgb=a,depth=b,depth_minus_rgb=b-a))
    return rows

def diagnostic(budget):
    data=read(ROOT/'logs/P6/correspondence_v40/data_audit.json');metadata=[]
    for group,records in data['records'].items():
        for r in records:
            base=ROOT/data['data_root'];headers={}
            for kind in ('rgb','depth','label'):
                def header():
                    with Image.open(base/r['files'][kind]) as im:return dict(size=list(im.size),mode=im.mode)
                headers[kind]=budget.invoke('real_png_headers',f'{group}/{r["scene"]}/{r["frame"]}/{kind}',header)
            m=budget.invoke('real_meta_reads',f'{group}/{r["scene"]}/{r["frame"]}',lambda:loadmat(base/r['files']['meta']))
            metadata.append(dict(scene=r['scene'],frame=r['frame'],headers=headers,K=np.asarray(m['intrinsic_matrix']).tolist(),
                                 factor_depth=float(m['factor_depth'].reshape(-1)[0]),fields=sorted(k for k in m if not k.startswith('__'))))
    frames={};sources={};results={};decoded={};checks={}
    for name in NAMES:
        source=fixtures(name);sources[name]=source
        frame=budget.invoke('synthetic_loader',name,lambda:load_frame(dict(data_root=INPUT.relative_to(ROOT).as_posix()),source['record'],labels=True));frames[name]=frame
        ow,oh=frame['original_size'];xs,ys=nearest_indices(ow,oh)
        expected_raw=source['depth'][ys[:,None],xs[None,:]].astype(np.float32)/source['factor']
        expected_valid=expected_raw>0;expected_depth=np.where(expected_valid,np.minimum(expected_raw,2.),0)
        expected_labels=source['labels'][ys[:,None],xs[None,:]]
        result=dict(original_size=[ow,oh],output_size=[640,480],depth_matches_independent_nearest=bool(np.array_equal(frame['depth'],expected_depth)),
                    validity_matches=bool(np.array_equal(frame['valid'],expected_valid)),eligible_matches=bool(np.array_equal(frame['eligible'],expected_valid&(expected_raw<=2.))),
                    labels_match=bool(np.array_equal(frame['labels'],expected_labels)),K_unchanged=bool(np.array_equal(frame['K'],source['K'])),
                    invalid_count=int((~frame['valid']).sum()),valid_but_ineligible_count=int((frame['valid']&~frame['eligible']).sum()))
        if name.startswith('coordinate'):
            axis='x' if name.endswith('x') else 'y'
            decoded[axis]=np.rint(frame['depth']*source['factor']).astype(np.int64)-1000
            expected=np.broadcast_to(xs[None,:],(480,640)) if axis=='x' else np.broadcast_to(ys[:,None],(480,640))
            result['actual_source_index_mismatches']=int(np.count_nonzero(decoded[axis]!=expected))
            original=source['rgb'][0,:,0] if axis=='x' else source['rgb'][:,0,0]
            reference=triangle_resize(original,640 if axis=='x' else 480)/255
            actual=frame['rgb'][0,:,0] if axis=='x' else frame['rgb'][:,0,0]
            result['rgb_bilinear_reference_max_error']=float(np.max(np.abs(actual-reference)))
        if name.startswith('shared_edges'):
            estimate=budget.invoke('synthetic_estimator',name,lambda:core.estimate(frame['rgb'],frame['depth'],frame['valid']))
            result['estimate']={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in estimate.items() if k!='edge_indices'}
            result['edges']=edge_positions(frame,source)
        results[name]=result
    for name in ('coordinate_x','coordinate_y'):
        results[name]['geometry']=expected_geometry(frames[name],decoded['x'],decoded['y'],budget,name)
    vv,uu=np.indices((480,640));centre_x=(uu+.5)*2-.5;centre_y=(vv+.5)*1.5-.5
    kp=K.copy();kp[0,0]/=2;kp[1,1]/=1.5;kp[0,2]=(kp[0,2]+.5)/2-.5;kp[1,2]=(kp[1,2]+.5)/1.5-.5
    center_err=max(float(np.max(np.abs((uu-kp[0,2])/kp[0,0]-(centre_x-K[0,2])/K[0,0]))),float(np.max(np.abs((vv-kp[1,2])/kp[1,1]-(centre_y-K[1,2])/K[1,1]))))
    offsets=dict(x_source_unique=np.unique(decoded['x']-centre_x).tolist(),y_source_unique=np.unique(decoded['y']-centre_y).tolist(),
                 x_output_unique=np.unique((decoded['x']-centre_x)/2).tolist(),y_output_unique=np.unique((decoded['y']-centre_y)/1.5).tolist())
    checks['raw_header_meta_consistent']=all(all(q['size']==[1280,720] for q in r['headers'].values()) and np.array_equal(np.array(r['K']),K) and r['factor_depth']==1000 for r in metadata)
    checks['nearest_depth_valid_eligible_label_K']=all(all(r[k] for k in ('depth_matches_independent_nearest','validity_matches','eligible_matches','labels_match','K_unchanged')) for r in results.values())
    checks['encoded_actual_indices_exact']=all(results[n]['actual_source_index_mismatches']==0 for n in ('coordinate_x','coordinate_y'))
    checks['source_xyz_and_reprojection']=all(r['geometry']['max_xyz_error_m']<=1e-10 and r['geometry']['max_source_reprojection_error_px']<=1e-10 for n,r in results.items() if n.startswith('coordinate'))
    checks['bilinear_independent_reference']=all(results[n]['rgb_bilinear_reference_max_error']<=2/255 for n in ('coordinate_x','coordinate_y'))
    checks['continuous_center_K_equivalence']=center_err<=1e-12
    checks['NN_vs_center_sampling_bound']=max(abs(v) for k in ('x_source_unique','y_source_unique') for v in offsets[k])<=.5
    fixture_ids={p.relative_to(ROOT).as_posix():stat(p) for p in INPUT.rglob('*') if p.is_file()}
    result=dict(version='v41r1',scope='first-step synthetic input geometry diagnostic only',checks=checks,
                input_geometry_implementation_error_detected=not all(v for k,v in checks.items() if k!='raw_header_meta_consistent'),
                dataset_contract_mismatch=not checks['raw_header_meta_consistent'],diagnostic_preconditions_passed=checks['raw_header_meta_consistent'],
                raw_metadata=metadata,fixtures=results,
                sampling_offsets=offsets,continuous_center_K=kp.tolist(),continuous_center_ray_max_error=center_err,
                native_RGBD_registration_certified=False,real_matching_bias_cause_proven=False,
                old_P6_scientific_failure_unchanged=True,P6_full_passed=False,all_tasks_percent=45,experiment_percent=56.25,
                artifacts=fixture_ids,ledger=audit(),runtime=runtime_identity())
    verify_calls(result['ledger'])
    atomic(OUT/'result.json',result);budget.guard()
    return result

def check_frozen():
    c=read(CONF)
    for p,s in c['protected'].items():need(stat(ROOT/p)==s,'Diagnostic protected identity changed '+p)
    verify_old();return c

def main(mode):
    if mode=='verify':
        c=check_frozen();result=read(OUT/'result.json');need(audit()==result['ledger'],'Result ledger changed');verify_calls(result['ledger'])
        for p,s in result['artifacts'].items():need(stat(ROOT/p)==s,'Synthetic artifact changed '+p)
        receipt=read(LOG/'run_receipt.json')
        need(sha(CONF)==receipt['config_sha256'],'Diagnostic frozen config changed')
        need(stat(OUT/'result.json')==receipt['result'],'Result changed')
        print(json.dumps(dict(readonly_verified=True,no_new_calls=True,checks=result['checks']),ensure_ascii=False));return
    with project_lock(ROOT):
        LOG.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
        need(not CONF.exists() and not (OUT/'result.json').exists() and not INPUT.exists(),'Existing diagnostic evidence; no automatic rerun')
        budget=Budget();old=verify_old();completed_verify(old)
        protect={p:stat(ROOT/p) for p in ['scripts/p6_preprocessing_v41r1.py',PROTOCOL.relative_to(ROOT).as_posix(),
            'logs/P6/correspondence_v40/data_audit.json','runtime/pipeline/state.json','runtime/pipeline/p6_correspondence_v40.json',
            'results/P6/correspondence_v40/generated.json','results/P6/correspondence_v40/model.json','results/P6/correspondence_v40/result.json',
            'results/P6/correspondence_v40/closeout.json','docs/P6对应可识别性结果_v40.txt']}
        atomic(CONF,dict(version='v41r1',protected=protect,source_protocol=PROTOCOL.relative_to(ROOT).as_posix(),cases=NAMES,call_caps=CAPS,
                         wall_seconds_cap=120,prior_attempt_conservative_charge_seconds=2.0,rss_bytes_cap=1024**3,output_bytes_cap=64*1024**2,raw_images_estimated=0,
                         scope='User-authorized step1 only; stop after result and independent review; no P6 retraining'))
        try:
            result=diagnostic(budget);check_frozen();budget.guard()
            atomic(LOG/'run_receipt.json',dict(completed=True,config_sha256=sha(CONF),result=stat(OUT/'result.json'),usage=read(LOG/'usage.json'),ledger=audit()))
            print(json.dumps(dict(completed=True,checks=result['checks'],sampling_offsets=result['sampling_offsets'],
                edges={n:r['estimate']['raw_uv'] for n,r in result['fixtures'].items() if n.startswith('shared_edges')}),ensure_ascii=False))
        except BaseException as e:
            atomic(LOG/'failure.json',dict(error=repr(e),ledger=audit()));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','verify']);main(parser.parse_args().mode)
