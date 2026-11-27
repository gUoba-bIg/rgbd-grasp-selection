"""Audit the actual decoder once; offline legacy decoding receives the same tensors.

This entry does not replace a function in the production import graph. Diagnostic
old candidate rows are isolated from the network output consumed by the ranker.
"""
import sys
from pathlib import Path
import json
import hashlib
import inspect
import argparse
import types
import traceback
import threading
from datetime import datetime, timezone

VERSION=Path(__file__).resolve().parents[1]
ROOT=VERSION.parent
RUNTIME=ROOT/'runtime/build/graspnet'
FIXED=RUNTIME/'utils/loss_utils.py'
BEFORE=VERSION/'audit/cross_source/loss_utils.before.py'
AFTER=VERSION/'audit/cross_source/loss_utils.after.py'
LEGACY_SHA='fa32ad1363fecd955bdafdf4bd1cc97ae3630915c5e4965e4fc593062b474321'
FIXED_SHA='0a8dea7a09020f869c329781a6b2a713f8ba62d385d07dab43deaab55e007eb7'
PREVIOUS=('IP140_R2_R4dev72_nonnegative_score_domain_resume01','IP140_R2_R4dev72_raw_a_c0_paired01')
EP_KEYS=('point_clouds','objectness_score','grasp_score_pred','fp2_xyz','grasp_top_view_xyz',
         'grasp_angle_cls_pred','grasp_width_pred','grasp_tolerance_pred','sa1_inds','sa2_inds')
IMPORT_NAMES={'graspnet','loss_utils','label_generation','modules','backbone','pointnet2_utils',
              'src.grasp_prerequisite_data_v27','src.modal_fusion_v38','policy_network','torch'}
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,o):
    with p.open('x',encoding='utf8') as f:json.dump(o,f,ensure_ascii=False,allow_nan=False,indent=2)
def recpath(p):
    p=Path(p).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)
def subcode(root,code):
    if root.co_name==code.co_name and root.co_firstlineno==code.co_firstlineno:return root
    for c in root.co_consts:
        if isinstance(c,types.CodeType):
            match=subcode(c,code)
            if match is not None:return match
def function_identity(f):
    c=f.__code__;p=Path(c.co_filename).resolve()
    ref=subcode(compile(p.read_bytes(),str(p),'exec',dont_inherit=True,optimize=sys.flags.optimize),c)
    if ref is None or ref!=c:raise RuntimeError('executing bytecode differs from on-disk source: '+str(p)+' '+f.__qualname__)
    return dict(function=f.__qualname__,module=f.__module__,co_filename=str(p),source_file=recpath(p),
                file_sha256=sha(p),source_text_sha256=hashlib.sha256(inspect.getsource(f).encode('utf8')).hexdigest(),
                first_line=c.co_firstlineno,live_code_equals_compiled_source=True)
def digest_array(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def as_np(a):
    import numpy as np
    if hasattr(a,'detach'):a=a.detach().cpu().numpy()
    return np.ascontiguousarray(a)
def byte_equal(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes(order='C')==b.tobytes(order='C')
def save_npz(path,**arrays):
    import numpy as np
    with path.open('xb') as f:np.savez_compressed(f,**arrays)

class Capture:
    def __init__(self,bound):
        self.codes={f.__code__:n for n,f in bound.items()}
        self.counts={n:{'call':0,'return':0} for n in bound}
        self.rot=[];self.decode=None;self.generate=None;self.returned=None;self.events=[];self.inputs={}
    def __call__(self,frame,event,arg):
        if event not in ('call','return'):return
        name=self.codes.get(frame.f_code)
        if not name:return
        self.counts[name][event]+=1
        caller=frame.f_back
        self.events.append(dict(event=event,function=name,caller=caller.f_code.co_qualname,
                      caller_source=recpath(caller.f_code.co_filename),caller_line=caller.f_lineno))
        q=frame.f_locals
        if event=='call':
            if name=='grasp_forward':self.inputs['P3']=as_np(q['end_points']['point_clouds']).copy()
            if name=='segmentation_forward':self.inputs['P5']=as_np(q['inputs']).copy()
            return
        if name=='rotation' and arg is not None:
            self.rot.append(dict(caller=caller.f_code,axis_x=q['axis_x'],axis_y=q['axis_y'],axis_z=q['axis_z'],
                                 towards=q['batch_towards'],angle=q['batch_angle'],matrix=arg))
        elif name=='pred_decode' and arg is not None:
            self.decode=dict(endpoints={k:q['end_points'][k] for k in EP_KEYS},
                objectness_mask=q['objectness_mask'],Ns=q['Ns'],rows=arg[0])
        elif name=='generate' and 'cand' in q and 'idx' in q:
            self.generate={k:q[k] for k in ('samples','source_pixels','where','idx','pixels','cand','cfg','scene','frame','condition')}
        elif name=='grasp_forward' and arg is not None:self.returned=arg
def diagnostic_decoder(graspnet,torch,endpoints):
    # The original decoder source and globals are copied into private namespaces.
    # Its only rebound dependency is the unmodified archived legacy rotation function.
    # Neither private namespace is placed in sys.modules, and no network is built here.
    old={'__name__':'isolated_r4_diagnostic_legacy_loss_utils','__file__':str(BEFORE)}
    exec(compile(BEFORE.read_bytes(),str(BEFORE),'exec',dont_inherit=True),old)
    src=Path(graspnet.pred_decode.__code__.co_filename).resolve()
    live=graspnet.pred_decode
    ref=subcode(compile(src.read_bytes(),str(src),'exec',dont_inherit=True,optimize=sys.flags.optimize),live.__code__)
    if ref!=live.__code__:raise RuntimeError('unaligned decoder source')
    scope={**live.__globals__,'batch_viewpoint_params_to_matrix':old['batch_viewpoint_params_to_matrix']}
    off=types.FunctionType(ref,scope,live.__name__,live.__defaults__,live.__closure__)
    with torch.inference_mode():original=off(endpoints)[0]
    return as_np(original)

def run(observation,directory,scene,frame_index,condition):
    if Path.cwd().resolve()!=ROOT or not directory.is_relative_to(VERSION) or not directory.is_dir():
        raise RuntimeError('explicit project root and existing version-owned unit directory required')
    if directory.parent not in (VERSION/'units',VERSION/'audit'):raise RuntimeError('unit/audit folder required')
    out=directory/'forward_audit';out.mkdir(exist_ok=False)
    import_log=[]
    def imports(event,args):
        if event=='import' and args[0] in IMPORT_NAMES:
            import_log.append(dict(module=args[0],supplied_filename=args[1],utc=now()))
    sys.addaudithook(imports)
    cap=None
    try:
        if (sha(BEFORE),sha(AFTER),sha(FIXED))!=(LEGACY_SHA,FIXED_SHA,FIXED_SHA):
            raise RuntimeError('archived/live cross source identity differs')
        if 'graspnet' in sys.modules or 'loss_utils' in sys.modules or sys.getprofile() is not None:
            raise RuntimeError('fresh standalone process with no existing decoder/profile required')
        import policy_network
        import numpy as np
        import torch
        from src.grasp_prerequisite_data_v27 import SOURCE
        import graspnet,loss_utils,label_generation,modules
        # policy_network imports the same objects normally when generate() is called.
        from src.modal_fusion_v38 import ModalFusionV38
        bound=dict(generate=policy_network.generate,grasp_forward=graspnet.GraspNet.forward,
                   segmentation_forward=ModalFusionV38.forward,pred_decode=graspnet.pred_decode,
                   rotation=loss_utils.batch_viewpoint_params_to_matrix)
        if Path(SOURCE).resolve()!=RUNTIME or Path(loss_utils.__file__).resolve()!=FIXED:
            raise RuntimeError('wrong active gripper decoder runtime')
        if (Path(graspnet.__file__).resolve()!=RUNTIME/'models/graspnet.py' or
            any(f is not bound['rotation'] for f in (label_generation.batch_viewpoint_params_to_matrix,
                modules.batch_viewpoint_params_to_matrix,graspnet.pred_decode.__globals__['batch_viewpoint_params_to_matrix']))):
            raise RuntimeError('decoder or stage 1 is bound to another rotation function')
        pidmods={}
        for n in sorted(IMPORT_NAMES):
            m=sys.modules.get(n)
            if m is not None and getattr(m,'__file__',None):
                p=Path(m.__file__).resolve();pidmods[n]=dict(file=str(p),sha256=sha(p))
        identity={k:function_identity(f) for k,f in bound.items()}
        import os
        runtime=dict(utc=now(),python_executable=sys.executable,pid=os.getpid(),thread_id=threading.get_ident(),
            fresh_module_load=True,torch_version=torch.__version__,cuda_version=torch.version.cuda,
            python_runtime=sys.version,source_runtime=recpath(RUNTIME),modules=pidmods,functions=identity,
            import_audit_events=import_log,argv=sys.argv,sys_path=list(sys.path),old_sha256=LEGACY_SHA,new_sha256=FIXED_SHA,
            legacy_in_production_import_tree=False,rotation_aliases_verified=['modules','label_generation','graspnet.pred_decode.__globals__'])
        write(out/'runtime.json',runtime)
        cap=Capture(bound)
        sys.setprofile(cap)
        try:policy_network.generate(observation,directory,scene,frame_index,condition)
        finally:sys.setprofile(None)
        write(out/'production_call_log.json',dict(counts=cap.counts,events=cap.events,load_events=import_log,utc=now()))
        if any(cap.counts[k]!={'call':1,'return':1} for k in ('generate','grasp_forward','segmentation_forward','pred_decode')):
            raise RuntimeError('one complete production forward/P5/decoder call required')
        if cap.decode is None or cap.returned is None or cap.generate is None:raise RuntimeError('incomplete native frame evidence')
        endpoints=cap.decode['endpoints']
        if any(cap.returned[k] is not x for k,x in endpoints.items()):raise RuntimeError('decoder did not consume P3 forward return')
        orig_inputs={k:as_np(x) for k,x in endpoints.items()}
        save_npz(out/'diagnostic_decoder_inputs.npz',p5_forward_input=cap.inputs['P5'],**orig_inputs)
        archive=diagnostic_decoder(graspnet,torch,endpoints)
        if any(not byte_equal(a,as_np(endpoints[k])) for k,a in orig_inputs.items()):raise RuntimeError('diagnostic mutated original decoder data')
        with (out/'diagnostic_legacy_candidates.npy').open('xb') as f:np.save(f,archive,allow_pickle=False)
        with np.load(directory/'network.npz',allow_pickle=False) as dat:
            actual=dat['candidates'];pixels=dat['pixels']
        loc=cap.generate;n=len(actual)
        if n!=cap.decode['Ns'] or not byte_equal(actual,as_np(cap.decode['rows'])) or not byte_equal(actual,as_np(loc['cand'])):
            raise RuntimeError('official decoder output identity or count changed')
        if archive.shape!=actual.shape or actual.dtype!=archive.dtype or actual.shape!=(n,17):
            raise RuntimeError('diagnostic decoder row shape/dtype mismatch')
        where=as_np(loc['where']);seed_idx=as_np(loc['idx'])
        samples=as_np(loc['samples']);objmask=as_np(cap.decode['objectness_mask'])
        native_indices=orig_inputs['sa1_inds'][0].astype(np.int64)[orig_inputs['sa2_inds'][0].astype(np.int64)][where]
        source_idx=samples[seed_idx]
        if not (np.array_equal(where,np.flatnonzero(objmask)) and np.array_equal(native_indices,seed_idx)
                and np.array_equal(pixels,as_np(loc['source_pixels'])[source_idx])):
            raise RuntimeError('native decoder/seed/point/pixel order mismatch')
        array=dict(decoder_native_row=np.arange(n,dtype=np.int64),objectness_mask=objmask,
                   object_seed_output_indices=where,sampled_input_indices=seed_idx,observed_cloud_source_indices=source_idx,
                   sample_draw=samples,source_pixel_coordinates=pixels,native_candidates_before_output_write=as_np(cap.decode['rows']))
        calls=[];maincalls=[]
        for k,r in enumerate(cap.rot):
            prefix='call_'+str(k)+'_';who=r['caller']
            for name in ('axis_x','axis_y','axis_z','towards','angle','matrix'):array[prefix+name]=as_np(r[name])
            d=dict(call_index=k,caller=who.co_qualname,caller_source=recpath(who.co_filename),N=len(r['axis_x']),
                is_decoder=who is bound['pred_decode'].__code__,npz_prefix=prefix)
            if d['is_decoder']:
                maincalls.append(d)
                if not byte_equal(array[prefix+'matrix'].reshape(n,9),actual[:,4:13]):
                    raise RuntimeError('rotation frame is not the decoded 17D row')
            calls.append(d)
        if len(maincalls)!=1 or maincalls[0]['N']!=n:raise RuntimeError('one actual decoder cross invocation required')
        save_npz(out/'call_evidence.npz',**array)
        w=actual.dtype.itemsize
        diff=(np.ascontiguousarray(archive).view(np.uint8).reshape(n,17,w)!=np.ascontiguousarray(actual).view(np.uint8).reshape(n,17,w)).any(axis=2)
        rows=[dict(row_index=i,changed_17d_columns=np.flatnonzero(diff[i]).tolist(),legacy_row_sha256=digest_array(archive[i]),
                current_row_sha256=digest_array(actual[i]),seed_index=int(where[i]),native_sample_source_index=int(seed_idx[i]),
                cloud_index=int(source_idx[i]),source_pixel=pixels[i].tolist()) for i in range(n)]
        independent=[]
        for v in PREVIOUS:
            # Historical production and archived optional challenge references are separate identities.
            old=ROOT/v/directory.parent.name/directory.name/'network.npz'
            if not old.exists():continue
            meta=json.loads((old.parent/'network.json').read_bytes())
            if meta['observation_sha256']!=sha(observation) or meta['condition']!=condition:
                raise RuntimeError('existing archive input/source differs')
            with np.load(old,allow_pickle=False) as z:
                oldrows=z['candidates'];oldpixels=z['pixels']
                independent.append(dict(path=recpath(old),sha256=sha(old),
                    same_input_legacy_raw_rows_byte_identical=byte_equal(oldrows,archive),
                    same_pixels_byte_identical=byte_equal(oldpixels,pixels)))
        nonrot=not diff[:,[0,1,2,3,13,14,15,16]].any()
        passed=nonrot and (not diff.any() or n==3) and all(r['same_input_legacy_raw_rows_byte_identical'] and r['same_pixels_byte_identical'] for r in independent)
        if any(not np.isfinite(x).all() for x in (archive,actual)):passed=False
        received={k:as_np(x) for k,x in cap.inputs.items()}
        if set(received)!={'P3','P5'} or received['P3'].shape!=(1,20000,3) or received['P5'].shape!=(1,5,480,640):
            raise RuntimeError('registered native network input tensor missing or malformed')
        if not byte_equal(received['P3'],orig_inputs['point_clouds']):raise RuntimeError('P3 point cloud no longer matches decoder endpoints')
        native_inputs={k:dict(sha256=digest_array(x),shape=list(x.shape),dtype=str(x.dtype),serialization='C order tensor elements, batch dimension included') for k,x in received.items()}
        report=dict(native_forward_inputs=native_inputs,passed=bool(passed),unit=directory.name,scene=scene,frame=frame_index,condition=condition,N=n,seed=int(loc['cfg']['seed'])+scene*1000+frame_index,
            network_observation_sha256=sha(observation),rotation_calls=calls,candidate_counts=dict(official=n,legacy_same_input=n),
            fresh_production_forward_returns=cap.counts['grasp_forward']['return'],formal_decoder_returns=cap.counts['pred_decode']['return'],
            diagnostic_legacy_decoder_returns=1,independent_extra_network_forwards=0,
            native_row_order_proven=True,all_nonrotation_fields_byte_identical=bool(nonrot),entire_17d_matrix_byte_identical=bool(not diff.any()),
            changed_rows=int(diff.any(axis=1).sum()),old_archive_references=independent,official_network_sha256=sha(directory/'network.npz'),
            isolation='Only network.npz is the official output; the old decoder was executed in a private namespace on the captured forward return. No diagnostic candidate is fed to a method.',
            arrays={x:sha(out/x) for x in ('diagnostic_legacy_candidates.npy','diagnostic_decoder_inputs.npz','call_evidence.npz')},rows=rows)
        write(out/'row_comparison.json',report)
        if not passed:raise RuntimeError('17D identity outside the predeclared rotational scope or archived old output disagrees')
    except BaseException as exc:
        write(out/'STOP.json',dict(utc=now(),reason=repr(exc),traceback=traceback.format_exc(),
               production_counters=(cap.counts if cap else None),load_events=import_log,physics_calls=0))
        raise
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('observation',type=Path);ap.add_argument('unitdir',type=Path);ap.add_argument('scene',type=int);ap.add_argument('frame',type=int)
    ap.add_argument('condition',choices=('clean','joint'));a=ap.parse_args()
    run(a.observation.resolve(),a.unitdir.resolve(),a.scene,a.frame,a.condition)
if __name__=='__main__':main()

