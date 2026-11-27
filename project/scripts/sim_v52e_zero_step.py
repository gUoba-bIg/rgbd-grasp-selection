"""Single precharged v52e parameter-routing admission; no physical integration.

Only the root controller may launch this worker. No test helper imports it.
The worker never calls stepSimulation, setRealTimeSimulation, resetJointState,
setJointMotorControl*, createConstraint or a second connect.
"""
import os, sys, json, math, hashlib, time, traceback, argparse, importlib, importlib.metadata, importlib.util
from pathlib import Path
from datetime import datetime

STARTED=time.monotonic()
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parents[1]


def now():return datetime.now().astimezone().isoformat()
def identity(path):
    p=Path(path);s=p.stat()
    with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(sha256=sha,bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def read(path):return json.loads(Path(path).read_text('utf-8-sig'))
def project_path(value):
    p=(ROOT/value).resolve();p.relative_to(ROOT);return p

def jsonable(value):
    if isinstance(value,bytes):return {'bytes_hex':value.hex()}
    if isinstance(value,(tuple,list)):return [jsonable(x) for x in value]
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    return value

def write_once(path,value):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(jsonable(value),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def append_event(path,value):
    with Path(path).open('a',encoding='utf8') as f:
        f.write(json.dumps(jsonable(value),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def numeric_equal(a,b,atol=1e-12,rtol=1e-10):
    if isinstance(a,(float,int)) and isinstance(b,(float,int)):
        return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=atol+rtol*max(abs(a),abs(b))
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(numeric_equal(x,y,atol,rtol) for x,y in zip(a,b))
    if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(numeric_equal(a[k],b[k],atol,rtol) for k in a)
    return a==b


def process_rss():
    # Native process query, no subprocess or simulator side effect.
    if os.name!='nt':raise RuntimeError('This frozen admission is Windows-only')
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
    ctr=Counters();ctr.cb=ctypes.sizeof(ctr)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(ctr),ctr.cb):raise ctypes.WinError(ctypes.get_last_error())
    return int(ctr.WorkingSetSize)


def validate(args):
    if Path.cwd().resolve()!=ROOT:raise RuntimeError('Explicit project workdir is required')
    config_path=project_path(args.config);cfg=read(config_path);directory=project_path(args.attempt_dir)
    if not directory.is_dir():raise RuntimeError('Root must create exclusive charged directory')
    if cfg.get('schema')!='v52e_zero_step_v1' or cfg.get('execution_authorized') is not True:raise RuntimeError('No current v52e authorization')
    if cfg.get('max_connections')!=1 or cfg.get('max_physics_steps')!=0:raise RuntimeError('Only one zero-step connection authorized')
    if cfg.get('targets')!={'lateralFriction':1.0,'collisionMargin':.0005,'restitution':0.0}:raise RuntimeError('Frozen targets mismatch')
    if cfg.get('limits')!={'wall_seconds':120,'rss_bytes':6442450944}:raise RuntimeError('Frozen resource limits mismatch')
    if cfg.get('expected_distribution_version')!='3.2.7+v52e1':raise RuntimeError('Wrong approved patched build version')
    if Path(sys.executable).resolve()!=project_path(cfg['python']):raise RuntimeError('Must use isolated v52e Python')
    old=read(project_path(cfg['base_config']))
    if cfg['base_config']!='configs/simulation_v52r2.json' or cfg['robot_urdf']!=old['robot_urdf'] or cfg['robot_manifest']!=old['robot_manifest']:raise RuntimeError('Unchanged original URDF and manifest required')
    required={cfg['robot_urdf'],cfg['robot_manifest'],cfg['base_config'],cfg['pybullet_extension'],cfg['python'],'scripts/sim_v52e_zero_step.py'}
    if not required.issubset(cfg.get('bindings',{})):raise RuntimeError('Required provenance is not bound')
    for path,expected in cfg['bindings'].items():
        if identity(project_path(path))!=expected:raise RuntimeError('Frozen binding mismatch: '+path)
    charge=read(directory/'precharge.json');intent=read(directory/'intent.json')
    if charge.get('schema')!='sim_v52e_precharge_v1' or charge.get('unit_id')!='ZERO001' or charge.get('charged_index')!=1:raise RuntimeError('Invalid precharge')
    if charge['config_identity']!=identity(config_path) or charge['worker_identity']!=identity(__file__):raise RuntimeError('Precharge source binding mismatch')
    token=os.environ.pop('SIM_V52E_CONTROLLER_TOKEN','')
    if not token or hashlib.sha256(token.encode()).hexdigest()!=charge['token_hash']:raise RuntimeError('Missing current controller token')
    if intent.get('attempt_id')!=charge['attempt_id'] or intent.get('unit_id')!='ZERO001' or intent.get('charged_index')!=1:raise RuntimeError('Intent and precharge differ')
    if (directory/'result.json').exists() or (directory/'worker_claim.json').exists():raise RuntimeError('No retries or result overwrite')
    write_once(directory/'worker_claim.json',dict(at=now(),attempt_id=charge['attempt_id'],unit_id='ZERO001',pid=os.getpid(),config_identity=identity(config_path),worker_identity=identity(__file__),precharge_identity=identity(directory/'precharge.json'),intent_identity=identity(directory/'intent.json')))
    return cfg,charge,config_path,directory


def run(cfg,charge,config_path,directory):
    result=dict(schema='sim_v52e_zero_step_result_v1',at=now(),attempt_id=charge['attempt_id'],unit_id='ZERO001',status='failed',completion=False,admission_passed=False,
        actual_physics_steps=0,physics_step_calls_started=0,physics_step_calls_returned=0,connect_calls_started=0,connect_calls_returned=0,
        changeDynamics_calls_started=0,changeDynamics_calls_returned=0,unknown_calls=[],snapshots={},checks=[],exception=None,disconnect_returned=False,
        config_identity=identity(config_path),worker_identity=identity(__file__),base_config_identity=identity(project_path(cfg['base_config'])),
        boundary='No integration, real-time simulation, reset, motors, constraints, planning, grasping or dynamic validation; only one precharged DIRECT load/read/change/read session.')
    events=directory/'events.jsonl';active={};sequence=0;peak_rss=0;client_id=None;engine=None
    def limits():
        nonlocal peak_rss
        elapsed=time.monotonic()-STARTED;rss=process_rss();peak_rss=max(peak_rss,rss)
        if elapsed>cfg['limits']['wall_seconds']:raise TimeoutError('zero_step_wall_cap')
        if rss>cfg['limits']['rss_bytes']:raise MemoryError('zero_step_RSS_cap')
    def invoke(name,*args,cleanup=False,**kwargs):
        nonlocal sequence
        if not cleanup:limits()
        sequence+=1;call_id=sequence
        start=dict(event='call_started',at=now(),call_id=call_id,name=name,args=jsonable(args),kwargs=jsonable(kwargs))
        append_event(events,start);active[call_id]=start
        if name=='connect':result['connect_calls_started']+=1
        if name=='changeDynamics':result['changeDynamics_calls_started']+=1
        value=getattr(engine,name)(*args,**kwargs)
        append_event(events,dict(event='call_returned',at=now(),call_id=call_id,name=name,result=jsonable(value)))
        active.pop(call_id)
        if name=='connect':result['connect_calls_returned']+=1
        if name=='changeDynamics':result['changeDynamics_calls_returned']+=1
        if not cleanup:limits()
        return value
    try:
        spec=importlib.util.find_spec('pybullet')
        if spec is None or Path(spec.origin).resolve()!=project_path(cfg['pybullet_extension']):raise RuntimeError('Resolved engine binary not approved isolated extension')
        distribution_version=importlib.metadata.version('pybullet')
        if distribution_version!=cfg['expected_distribution_version']:raise RuntimeError('Patched distribution metadata version mismatch')
        engine=importlib.import_module('pybullet')
        if Path(engine.__file__).resolve()!=project_path(cfg['pybullet_extension']):raise RuntimeError('Loaded engine is not bound isolated extension')
        result['engine']=dict(python_executable=sys.executable,python_identity=identity(sys.executable),extension_path=Path(engine.__file__).resolve().relative_to(ROOT).as_posix(),extension_identity=identity(engine.__file__),distribution_version=distribution_version,python_version=sys.version)
        append_event(events,dict(event='provenance_verified',at=now(),engine=result['engine'],precharge_identity=identity(directory/'precharge.json')))
        client_id=invoke('connect',engine.DIRECT)
        if not isinstance(client_id,int) or client_id<0:raise RuntimeError('DIRECT connection did not succeed')
        result['client_id']=client_id
        result['engine']['api_version']=invoke('getAPIVersion')
        flags=engine.URDF_USE_INERTIA_FROM_FILE|engine.URDF_USE_SELF_COLLISION|engine.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
        body=invoke('loadURDF',cfg['robot_urdf'],basePosition=[0,0,0],useFixedBase=True,flags=flags,physicsClientId=client_id)
        count=invoke('getNumJoints',body,physicsClientId=client_id);names={-1:'BASE'}
        for index in range(count):
            raw=invoke('getJointInfo',body,index,physicsClientId=client_id)
            names[index]=raw[12].decode('utf8') if isinstance(raw[12],bytes) else str(raw[12])
        manifest=read(project_path(cfg['robot_manifest']));pads=manifest['pad_links']
        if len(pads)!=2 or len(set(pads))!=2 or 'left' not in pads[0] or 'right' not in pads[1]:raise RuntimeError('Fixed left/right pad sequence unavailable')
        link_indices={name:index for index,name in names.items()}
        if any(name not in link_indices for name in pads):raise RuntimeError('Target pad absent')
        result['body_id']=body;result['target_link_indices']={name:link_indices[name] for name in pads}
        def snapshot(label):
            data=dict(schema='sim_v52e_full_readback_v1',label=label,at=now(),body_id=body,actual_physics_steps=0,engine=result['engine'],links=[])
            for index in range(-1,count):
                dynamics=invoke('getDynamicsInfo',body,index,physicsClientId=client_id)
                if len(dynamics)<12:raise RuntimeError('Collision margin not available in full dynamics tuple')
                collision=invoke('getCollisionShapeData',body,index,physicsClientId=client_id)
                state=invoke('getBasePositionAndOrientation',body,physicsClientId=client_id) if index==-1 else invoke('getLinkState',body,index,computeForwardKinematics=1,computeLinkVelocity=1,physicsClientId=client_id)
                joint=None if index==-1 else invoke('getJointState',body,index,physicsClientId=client_id)
                data['links'].append(dict(link_index=index,link_name=names[index],dynamics_raw=jsonable(dynamics),collision_geometry_raw=jsonable(collision),pose_velocity_raw=jsonable(state),joint_state_raw=jsonable(joint),mass_kg=dynamics[0],lateral_friction=dynamics[1],restitution=dynamics[5],collision_margin_m=dynamics[11]))
            path=directory/(label+'.json');write_once(path,data);result['snapshots'][label]=dict(path=path.relative_to(ROOT).as_posix(),identity=identity(path));return data
        previous=snapshot('baseline')
        for stage,name in zip(('afterL','afterR'),pads):
            target=link_indices[name]
            append_event(events,dict(event='target_change_intent',at=now(),stage=stage,body_id=body,target_link_index=target,target_link_name=name,parameters=cfg['targets']))
            invoke('changeDynamics',body,target,lateralFriction=cfg['targets']['lateralFriction'],collisionMargin=cfg['targets']['collisionMargin'],restitution=cfg['targets']['restitution'],physicsClientId=client_id)
            current=snapshot(stage);failures=[];allowed={1,5,11}
            for before,after in zip(previous['links'],current['links']):
                idx=before['link_index'];same_target=idx==target
                if after['link_index']!=idx or after['link_name']!=before['link_name']:failures.append(dict(link=idx,field='link_identity'))
                for k,(a,b) in enumerate(zip(before['dynamics_raw'],after['dynamics_raw'])):
                    if same_target and k in allowed:continue
                    if a!=b:failures.append(dict(link=idx,field='dynamics_raw',index=k,before=a,after=b))
                if len(before['dynamics_raw'])!=len(after['dynamics_raw']):failures.append(dict(link=idx,field='dynamics_length'))
                for field in ('collision_geometry_raw','pose_velocity_raw','joint_state_raw'):
                    equal=numeric_equal(before[field],after[field]) if same_target and field=='collision_geometry_raw' else before[field]==after[field]
                    if not equal:failures.append(dict(link=idx,field=field,before=before[field],after=after[field]))
                if same_target:
                    for field,key in [('lateral_friction','lateralFriction'),('restitution','restitution'),('collision_margin_m','collisionMargin')]:
                        if not numeric_equal(after[field],cfg['targets'][key],atol=1e-12,rtol=0):failures.append(dict(link=idx,field=field,expected=cfg['targets'][key],actual=after[field]))
            entry=dict(stage=stage,target_link_index=target,target_link_name=name,passed=not failures,failures=failures,
                non_target_dynamics_and_geometry_rule='exact tuple equality',target_unchanged_geometry_numeric_tolerance=dict(atol=1e-12,rtol=1e-10),
                before=result['snapshots']['baseline' if stage=='afterL' else 'afterL'],after=result['snapshots'][stage])
            result['checks'].append(entry);append_event(events,dict(event='stage_verified',at=now(),**entry))
            if failures:raise RuntimeError('Immediate parameter-routing admission failure at '+stage)
            previous=current
        result['completion']=True;result['status']='completed';result['admission_passed']=True
    except BaseException as exc:
        result['exception']=repr(exc);result['traceback']=traceback.format_exc();result['status']='failed';result['completion']=False;result['admission_passed']=False
    finally:
        if client_id is not None and client_id>=0 and engine is not None:
            try:invoke('disconnect',physicsClientId=client_id,cleanup=True);result['disconnect_returned']=True
            except BaseException as exc:
                result['disconnect_exception']=repr(exc);result['status']='failed';result['completion']=False;result['admission_passed']=False
        result['unknown_calls']=list(active.values());result['peak_sampled_process_rss_bytes']=peak_rss;result['wall_seconds']=time.monotonic()-STARTED
        if result['wall_seconds']>cfg['limits']['wall_seconds'] or peak_rss>cfg['limits']['rss_bytes'] or active:
            result['status']='failed';result['completion']=False;result['admission_passed']=False
        result['events_identity']=identity(events) if events.exists() else None;result['finished_at']=now()
        write_once(directory/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--attempt-dir',required=True);args=parser.parse_args()
    cfg,charge,config_path,directory=validate(args)
    result=run(cfg,charge,config_path,directory)
    print(json.dumps({key:result[key] for key in ('status','admission_passed','connect_calls_started','connect_calls_returned','changeDynamics_calls_started','changeDynamics_calls_returned','actual_physics_steps','unknown_calls')},ensure_ascii=False))
    return 0 if result['admission_passed'] else 1

if __name__=='__main__':raise SystemExit(main())
