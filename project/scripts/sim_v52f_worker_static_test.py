"""Pure AST/extracted-function/fake-object checks; no worker or engine import."""
import ast,copy,io,json,math,os,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['PYTHONDONTWRITEBYTECODE']='1'
p=ROOT/'scripts/sim_v52f_episode.py';source=p.read_text('utf8');tree=ast.parse(source)
old=ast.parse((ROOT/'scripts/sim_v52d_episode.py').read_text('utf8'))
def node(t,name):return next(x for x in ast.walk(t) if isinstance(x,ast.FunctionDef) and x.name==name)
checks=[]
def ck(condition,label):
    if not condition:raise AssertionError(label)
    checks.append(label)
for name in ('phase_schedule','known_loads','original_evaluation','move'):
    ck(ast.dump(node(tree,name),include_attributes=False)==ast.dump(node(old,name),include_attributes=False),'unchanged_'+name)
ck(source.count('client.stepSimulation()')==1,'one_owner_step_call_site')
ns=dict(json=json,math=math,time=time,PROCESS_STARTED=time.monotonic())
for name in ('normalize','geometry_equal','read_setup_robot_parameters','read_pad_parameter_assertion','tick'):
    exec(compile(ast.Module(body=[node(tree,name)],type_ignores=[]),'<extracted '+name+'>','exec'),ns)
ref=json.loads((ROOT/'results/simulation_v52e/attempts/001_ZERO001_2b433b45/afterR.json').read_text('utf8'))
records={x['link_index']:x for x in ref['links']};manifest={'pad_links':['rq_left_inner_finger_pad','rq_right_inner_finger_pad']}
robot={'body':7,'pads':[16,21],'links':{x['link_name']:x['link_index'] for x in ref['links'] if x['link_index']!=-1}}
class Fake:
    def __init__(self,mutate=None,fail=False):self.mutate=mutate;self.fail=fail;self.fake_steps=0
    def getDynamicsInfo(self,body,index):
        if self.fail:raise RuntimeError('injected readback failure')
        raw=copy.deepcopy(records[index]['dynamics_raw'])
        if self.mutate and index==self.mutate[0]:raw[self.mutate[1]]=self.mutate[2]
        return tuple(raw)
    def getCollisionShapeData(self,body,index):
        shape=copy.deepcopy(records[index]['collision_geometry_raw'])
        for item in shape:item[0]=body
        return shape
    def stepSimulation(self):self.fake_steps+=1
fake=Fake();setup=ns['read_setup_robot_parameters'](fake,robot,ref);ck(setup['passed'],'all_links_match_afterR_with_body_ID_normalization')
ck(next(x for x in setup['links'] if x['collision_geometry_raw'])['collision_geometry_raw'][0][0]==7,'raw_body_ID_preserved')
for mutation,label in [((16,1,.5),'friction'),((16,11,.001),'margin'),((21,5,.1),'restitution')]:
    row=ns['read_pad_parameter_assertion'](Fake(mutation),robot,manifest,1,'settle',0);ck(not row['passed'],'reject_actual_'+label)
row=ns['read_pad_parameter_assertion'](Fake(fail=True),robot,manifest,1,'settle',0)
ck(not row['passed'] and row['readback_calls_started']==1 and row['readback_calls_returned']==0 and row['exception'],'unknown_readback_not_fabricated')
invalid=ns['read_pad_parameter_assertion'](Fake((16,1,float('nan'))),robot,manifest,1,'settle',0)
ck(not invalid['passed'] and invalid['readback_calls_started']==invalid['readback_calls_returned']==1 and invalid.get('invalid_returned_raw_repr'),'returned_nonfinite_readback_is_failed_not_unknown')
row=ns['read_pad_parameter_assertion'](fake,robot,manifest,1,'settle',0)
ck(row['passed'] and row['readback_calls_returned']==2 and row['physics_steps_returned_before']==0,'two_real_field_reads_required_in_mock')
ck(not ns['read_setup_robot_parameters'](Fake((-1,1,1.)),robot,ref)['passed'],'reject_base_pollution')
ck(not ns['read_setup_robot_parameters'](Fake((16,0,.006)),robot,ref)['passed'],'reject_changed_mass')
class Stream(io.StringIO):
    def fileno(self):return 123
class FakeOS:
    calls=0
    @staticmethod
    def fsync(fd):FakeOS.calls+=1
class Recorder:
    def pre_step(self,*args):pass
    def capture(self,*args):pass
for bad in (False,True):
    FakeOS.calls=0
    client=Fake((16,1,.5) if bad else None)
    result=dict(actual_physics_steps=0,pad_assertions_attempted=0,pad_assertions_passed=0,last_failed_pad_assertion=None,physics_step_calls_started=0,physics_step_calls_returned=0,unknown_step_call=None)
    stream=Stream();ns.update(client=client,robot=robot,manifest=manifest,phase='settle',result=result,limits=lambda:None,assertion_stream=stream,os=FakeOS,known_loads=lambda:[],recorder=Recorder(),targets={})
    rejected=False
    try:ns['tick']()
    except RuntimeError:rejected=True
    row=json.loads(stream.getvalue())
    ck(row['passed']==(not bad),'persisted_pass_or_failure_before_any_mock_step_'+str(bad))
    ck(client.fake_steps==(0 if bad else 1) and result['actual_physics_steps']==client.fake_steps,'no_current_or_later_step_on_assertion_failure_'+str(bad))
    ck(rejected==bad,'failure_is_terminal_not_auto_repaired_'+str(bad))
    ck(FakeOS.calls==(1 if bad else 0),'failure_immediately_fsync_nonboundary_success_flush_only_'+str(bad))
stat=p.stat();out=ROOT/'logs/simulation_v52f/preparation';out.mkdir(parents=True,exist_ok=True)
report=dict(passed=True,checks=checks,worker_identity=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns),real_connections=0,real_physics_steps=0,worker_imported=False,engine_imported=False,scope='AST plus Python fake objects only; fake step counter is not a simulator call')
path=out/('worker_static_'+str(time.time_ns())+'.json');path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(str(path.relative_to(ROOT)));print(json.dumps(report,ensure_ascii=False))
