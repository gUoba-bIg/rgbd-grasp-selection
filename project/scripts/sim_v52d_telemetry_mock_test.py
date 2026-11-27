"""Pure fake-client checks. No physics package imported or simulation executed."""
from pathlib import Path
import ast
import gzip
import hashlib
import json
import sys
import numpy as np
sys.dont_write_bytecode = True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.sim_v52d_telemetry import Recorder

cfg=json.loads((ROOT/'configs/simulation_v52r2.json').read_text(encoding='utf-8'))
manifest=json.loads((ROOT/cfg['robot_manifest']).read_text(encoding='utf-8'))
structure=json.loads((ROOT/'data/simulation_v52/robot/structure_check.json').read_text(encoding='utf-8'))
table=structure['joint_table']
names={x['joint']:x['index'] for x in table}
links={x['link']:x['index'] for x in table}
active=manifest['arm_joint_names']+[manifest['master_joint']]+[x['joint'] for x in manifest['mimic_joints']]
order=[x['joint'] for x in table if x['joint'] in active]

class Fake:
    """All APIs return algebraic fixtures; no real engine or integration."""
    def __init__(self,object_present=False):self.after=False;self.object_present=object_present
    def getNumJoints(self,body):return len(table)
    def getJointInfo(self,body,i):
        x=table[i];n=x['joint'];k=order.index(n) if n in order else -1
        return (i,n.encode(),0 if k>=0 else 4,7+k if k>=0 else -1,6+k if k>=0 else -1,0,0.,0.,-1.,1.,10.,1.,x['link'].encode(),(1.,0.,0.),(0.,0.,0.),(0.,0.,0.,1.),-1)
    def getJointStates(self,body,indices):
        return [(0.01*order.index(table[i]['joint'])+(0.002 if self.after else 0),0.4 if self.after else .2,(0.,)*6,float(order.index(table[i]['joint'])+1)) for i in indices]
    def getLinkState(self,body,i,**kw):
        xyz=[i*.01,.002 if self.after else 0.,.3]
        return (xyz,(0.,0.,0.,1.),(0.,0.,0.),(0.,0.,0.,1.),xyz,(0.,0.,0.,1.),(0.,.4,0.),(0.,0.,.2))
    def getLinkStates(self,body,indices,**kw):return [self.getLinkState(body,i) for i in indices]
    def getBasePositionAndOrientation(self,body):return ([0.,0.,1.],(0.,0.,0.,1.))
    def getBaseVelocity(self,body):return ((0.,0.,0.),(0.,0.,0.))
    def getDynamicsInfo(self,body,link):return (0. if link==-1 and body==89 else 1.,.6,(1.,2.,3.),(0.,0.,0.),(0.,0.,0.,1.),0.,0.,0.,-1.,-1.,2,.0005)
    def getConstraintInfo(self,cid):return (89,0,89,1,6,(1.,0.,0.),(0.,0.,0.),(0.,0.,0.),(0.,0.,0.,1.),(0.,0.,0.,1.),80.,1.,-1,0.,.2)
    def getConstraintState(self,cid):return [float(cid)/10]
    def getPhysicsEngineParameters(self):return {'fixedTimeStep':1/240,'numSubSteps':0,'numSolverIterations':150,'mock_fixture_not_real_engine':True}
    def getAPIVersion(self):return 'mock-no-physics'
    def getCollisionShapeData(self,*args):return []
    def calculateMassMatrix(self,body,q):return np.diag(np.arange(1,13)).tolist()
    def getContactPoints(self):
        if not self.object_present:return []
        return [(0,89,91,links[manifest['pad_links'][0]],-1,(.2,.1,.3),(.2,.1,.3),(0.,1.,0.),-.0001,3.,.4,(1.,0.,0.),-.2,(0.,0.,1.))]
    def connect(self,*a,**k):raise AssertionError('Forbidden connect')
    def stepSimulation(self,*a,**k):raise AssertionError('Forbidden physics step')

robot=dict(body=89,joints=names,links=links,arm=[names[n] for n in manifest['arm_joint_names']],pads=[links[n] for n in manifest['pad_links']],gear_ids=[1,2,3,4,5])
targets=dict(arm=dict(joint_indices=robot['arm'],positions_rad=[0.]*6,forces_Nm=[150.,150.,150.,28.,28.,28.],position_gains=[.1]*6,velocity_gains=[1.]*6,max_velocity_rad_s=None),gripper=dict(master_joint_index=names[manifest['master_joint']],position_rad=.3,force_Nm=10.,position_gain=.1,velocity_gain=1.,max_velocity_rad_s=.5),followers=dict(mode='VELOCITY_CONTROL',force_Nm=0.))
run_tag=sys.argv[1] if len(sys.argv)>1 else '001'
assert run_tag.isdigit() and len(run_tag)==3
base=ROOT/'runtime/simulation_v52d'/('telemetry_mock_tests_'+run_tag)
assert not base.exists(),'Preserve previous mock evidence'
base.mkdir(parents=True)
tests=[]
for scenario in ['L','C']:
    p=Fake(object_present=scenario=='C');out=base/scenario
    rec=Recorder(p,robot,90,91 if scenario=='C' else None,cfg,out,scenario,manifest)
    ext=[]
    if scenario=='L':
        i=robot['pads'][0];xyz=list(p.getLinkState(89,i)[0]);xyz[0]+=.03
        ext=[dict(link_index=i,point_world_m=xyz,force_world_N=[0.,1.,0.])]
    rec.pre_step(1,'close',targets,ext)
    p.after=True  # fixture switch only; this is NOT time integration
    got=rec.capture(1,'close',targets,ext)
    summary=rec.finalize()
    assert summary['sample_count']==1 and summary['telemetry_complete']
    with gzip.open(out/'telemetry.jsonl.gz','rt',encoding='utf-8') as f:saved=[json.loads(line) for line in f]
    assert len(saved)==1 and saved[0]['step']==1
    assert len(saved[0]['mass_matrix']['matrix'])==12
    assert all(abs(v['d_full_over_d_source']-1.)<1e-12 for v in got['mass_matrix']['gear_rows'])
    for row in got['joints']:
        assert abs(row['work_discrete_J']-row['applied_motor_torque_Nm']*.002)<1e-12
    for row in got['gears']:
        assert abs(row['work_discrete_J']-row['reaction_Nm']*(row['e_post_rad']-row['e_pre_rad']))<1e-12
    if scenario=='L':
        x=got['external_loads'][0]
        assert np.allclose(x['torque_about_com_pre_Nm'],[0.,0.,.03])
        assert abs(x['work_discrete_J']-.002)<1e-12
        assert abs(x['power_pre_W']-.406)<1e-12
    else:
        x=got['contacts'][0]
        assert np.allclose(x['force_on_A_world_N'],[.4,3.,-.2])
        assert np.allclose(x['force_on_B_world_N'],[-.4,-3.,.2])
        assert got['flags']['pad_links_with_normal_contact']==[robot['pads'][0]]
    tests.append(scenario+'_schema_work_force_matrix_pass')
# AST inspection explicitly rules out executable physics mutation calls.
code=(ROOT/'scripts/sim_v52d_telemetry.py').read_text(encoding='utf-8');parsed=ast.parse(code)
for node in ast.walk(parsed):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
        assert node.func.attr not in {'connect','stepSimulation','setJointMotorControl2','setJointMotorControlArray','createConstraint','changeConstraint','applyExternalForce','resetJointState','enableJointForceTorqueSensor'}
tests.append('no_physics_mutation_calls_AST')
(base/'result.json').write_text(json.dumps(dict(passed=True,tests=tests,real_physics_steps=0,real_connections=0,
    telemetry_source_sha256=hashlib.sha256((ROOT/'scripts/sim_v52d_telemetry.py').read_bytes()).hexdigest()),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,tests=tests,real_physics_steps=0)))
