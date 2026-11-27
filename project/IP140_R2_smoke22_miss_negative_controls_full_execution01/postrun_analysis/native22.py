"""Reconcile actual call/return streams, the registered filters and the ledger after closure."""
from pathlib import Path
import gzip,hashlib,json,sys
from collections import Counter
V=Path(__file__).resolve().parents[1];sys.path.insert(0,str(V))
from structural_filter import STRUCTURAL_PAIRS
def read(f):return json.loads(f.read_bytes())
def sha(f):
    with f.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def inspect(folder, scope):
    path=folder/'full_api/calls.jsonl.gz'
    if not path.is_file():return None
    zero=read(folder/'zero_readback.json')
    robot,target,table,client=[zero[x] for x in ('robot_body_id','object_body_id','table_body_id','actual_client')]
    links=read(folder/'zero_step/all_links_readback.json')
    expected=[[robot,robot,links[a]['index'],links[b]['index'],0] for a,b in STRUCTURAL_PAIRS]+[[robot,table,links['base_link_inertia']['index'],-1,0]]
    calls=Counter();returns=Counter();filters=[];errors=[];pending=None;count=0;first=None;ids=[]
    with gzip.open(path,'rb') as f:
        for line in f:
            row=json.loads(line);seq,event=row[:2]
            if event=='call':
                count+=1
                if seq!=count or pending is not None:errors.append(['sequence',seq])
                pending=row;method,args,kw=row[3:];calls[method]+=1
                if method=='stepSimulation':
                    if first is None:first=seq
                    if scope=='static':errors.append(['static_step',seq])
                if ('physicsClientId' in kw and kw['physicsClientId']!=client):errors.append(['client',seq])
                if method=='setCollisionFilterPair':filters.append([seq,args])
                if method.startswith('setCollisionFilter') and method!='setCollisionFilterPair': errors.append(['foreign_filter',seq])
                if method=='resetJointState' and (not args or args[0]!=robot or first is not None):errors.append(['unexpected_reset',seq])
                if method in {'resetBasePositionAndOrientation','resetBaseVelocity','createConstraint','resetSimulation','removeBody'}:errors.append(['forbidden',seq,method])
            elif event=='return' and pending is not None and pending[0]==seq:
                method=pending[3];returns[method]+=1
                if method=='connect':ids.append(row[3])
                if method=='setCollisionFilterPair' and row[3] is not None:errors.append(['filter_return',seq])
                pending=None
            else:errors.append([event,seq]);pending=None
    if pending:errors.append(['unreturned',pending[0]])
    summ=read(folder/'full_api/summary.json')
    if calls!=returns or dict(calls)!=summ['calls'] or dict(returns)!=summ['returns'] or summ['exceptions']:errors.append('counts/summary')
    if [x[1] for x in filters]!=expected or ids!=[client] or returns['disconnect']!=1:errors.append('filter/connection')
    return dict(scope=scope,paired_without_exception=not errors,first_native_physics_sequence=first,step_calls=calls['stepSimulation'],step_returns=returns['stepSimulation'],calls=count,returns=sum(returns.values()),connections=ids,disconnects=returns['disconnect'],filter_calls=filters,pre_step_robot_joint_resets=calls['resetJointState'],errors=errors,source=path.relative_to(V).as_posix(),sha256=sha(path))
