"""Offline tests of the same receipt validator used before physical execution."""
from copy import deepcopy
from pathlib import Path
from hashlib import sha256
import json
import io
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

import asset_schema19 as verifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD17 = ROOT / 'IP140_R1_smoke17_slot14_postentry_scope_fix_and_run01'
OLD18 = ROOT / 'IP140_R1_smoke18_remaining_five_bridge_confirmation01'

def digest(p):
    h = sha256()
    with p.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def historical(base, registry, slot):
    return (base, registry, slot, *verifier.load_receipt(base, registry, slot))

def validate(sample):
    base, registry, slot, zero, rd, native = sample
    result = verifier.validate_receipt(base, registry, slot, zero=zero, readback=rd, native=native)
    if isinstance(result, dict) and result.get('passed') is False:
        raise ValueError(str(result))
    return result

def synthetic_017(source):
    """Construct the unobserved object's geometry explicitly from its slot9 frozen manifest."""
    base, _, _, zero, rd, api = deepcopy(source)
    registry = HERE/'slots/09/preregistration.json'
    registration = json.loads(registry.read_bytes())['slots'][8]
    scene = registration['scene_setup_only']
    path = ROOT / scene['urdf']['project_relative']
    assert digest(path) == scene['urdf']['sha256']
    tree = ET.parse(path)
    parse3 = lambda node,attr: [float(v) for v in node.get(attr).split()]
    boxes = []
    for node in tree.findall('./link/collision'):
        origin = node.find('origin')
        assert parse3(origin,'rpy') == [0,0,0]
        boxes.append(dict(name=node.get('name'), dims=parse3(node.find('geometry/box'),'size'),pos=parse3(origin,'xyz'),quat=[0.,0.,0.,1.]))
    assert len(boxes)==640
    base_pose = scene['position_world_m']
    dims = np.array([r['dims'] for r in boxes])
    pos = np.array([r['pos'] for r in boxes])
    margin = rd['dynamics'][11]
    bounds = [(np.min(pos-dims/2,axis=0)+base_pose-margin).tolist(), (np.max(pos+dims/2,axis=0)+base_pose+margin).tolist()]
    rows = [[rd['body_id'] if i==0 else -901, -1 if i==0 else 2901,3, b['dims'],'', b['pos'], b['quat']] for i,b in enumerate(boxes)]
    inertia = tree.find('./link/inertial/inertia')
    principal = [float(inertia.get(x)) for x in ('ixx','iyy','izz')]
    rd['dynamics'][2]=principal
    rd['body_info']=[tree.find('link').get('name'),tree.getroot().get('name')]
    rd['base_pose'] = [base_pose,scene['quaternion_xyzw']]
    rd['collision_shapes']=rows; rd['actual_AABB']=bounds
    proof=rd['compound_box_proof']
    proof.update(expected_count=len(rows),observed_count=len(rows),expected_boxes=boxes,expected_world_AABB=bounds,max_abs_AABB_error_m=0.0,maximum_abs_errors={'dims':0.,'pos':0.,'quat':0.})
    api['loadURDF']['args']=[scene['urdf']['project_relative']]
    api['loadURDF']['kwargs'].update(basePosition=base_pose,baseOrientation=scene['quaternion_xyzw'])
    for name,values in [('getCollisionShapeData',rows),('getAABB',bounds),('getDynamicsInfo',rd['dynamics'])]:
        api[name]['return']=deepcopy(values)
    zero.update(slot=9,static_candidate_slot=9,object_id=registration['object_id'],candidate_id=registration['candidate_id'],preregistration_sha256=digest(registry),
                trajectory_plan_sha256=digest(HERE/'slots/09/trajectory_plan.json'))
    for f in zero['fields']:
        name=f['field'].replace('object014.','object017.').replace('slot08.','slot09.')
        f['field']=name
        vals={'object017.compound_collision_shape_count':len(rows),'object017.initial_position':base_pose,'object017.principal_inertia':principal}
        if name in vals:
            f['expected']=deepcopy(vals[name]);f['readback']=deepcopy(vals[name])
    return base,registry,9,zero,rd,api

def run():
    mesh=historical(OLD17/'static_preflight',OLD17/'preregistration.json',14)
    compound=historical(OLD18/'slots/08/static_preflight',OLD18/'slots/08/preregistration.json',8)
    other=synthetic_017(compound)
    results=[]
    def case(name,source_id,sample,expected,mutate=None):
        try:
            s=deepcopy(sample) if mutate else sample
            if mutate:mutate(s)
            accepted=validate(s); actual=True; reason=None
        except (ValueError,KeyError,TypeError,IndexError,AssertionError,FileNotFoundError) as exc:
            actual=False;reason=f'{type(exc).__name__}: {exc}'
        results.append(dict(case=name,sample=source_id,expected_accept=expected,actual_accept=actual,passed=(expected==actual),reject_reason=reason))
    case('archived_smoke17_complete_named_mesh_217','021 actual',mesh,True)
    case('archived_smoke18_complete_compound_216','014 actual',compound,True)
    case('registered_640_ordered_boxes_and_owner_mapped_common_fields','017 synthetic',other,True)
    def set_field(key,value):
        return lambda s:s[3]['fields'][0].__setitem__(key,value)
    def add(name):
        def mutate(s):
            r=deepcopy(s[3]['fields'][0]);r['field']=name; s[3]['fields'].append(r)
        return mutate
    bad=[
        ('omit_common_field',lambda s:s[3]['fields'].pop(0)),
        ('duplicate_common_field',lambda s:s[3]['fields'].append(deepcopy(s[3]['fields'][1]))),
        ('substitute_with_duplicate_keep_count',lambda s:s[3]['fields'].__setitem__(0,deepcopy(s[3]['fields'][1]))),
        ('add_unknown_field',add('unregistered.extra')),
        ('compound_requires_no_mesh_filename',add('object014.mesh_filename')),
        ('status_not_pass',set_field('status','warn')),
        ('missing_status',lambda s:s[3]['fields'][0].pop('status')),
        ('missing_expected',lambda s:s[3]['fields'][0].pop('expected')),
        ('missing_readback',lambda s:s[3]['fields'][0].pop('readback')),
        ('null_expected',set_field('expected',None)),('null_readback',set_field('readback',None)),
        ('wrong_type_on_both_sides',lambda s:s[3]['fields'][0].update(expected='fake',readback='fake')),
        ('wrong_shape_on_both_sides',lambda s:s[3]['fields'][0].update(expected=[0],readback=[0])),
        ('negative_tolerance',set_field('atol',-1)),('nonfinite_tolerance',set_field('atol',float('nan'))),
        ('slot_owner_mismatch',lambda s:s[3].__setitem__('slot',13)),
        ('candidate_owner_mismatch',lambda s:s[3].__setitem__('candidate_id','wrong:candidate')),
        ('object_owner_mismatch',lambda s:s[3].__setitem__('object_id','021')),
        ('prereg_sha_mismatch',lambda s:s[3].__setitem__('preregistration_sha256','0'*64)),
        ('shape_call_body_mismatch',lambda s:s[5]['getCollisionShapeData']['args'].__setitem__(0,713)),
        ('shape_call_client_mismatch',lambda s:s[5]['getCollisionShapeData']['kwargs'].__setitem__('physicsClientId',73)),
        ('different_loaded_urdf_path',lambda s:s[5]['loadURDF']['args'].__setitem__(0,'wrong.urdf')),
    ]
    for name, mutate in bad:case(name,'014 actual / mutated',compound,False,mutate)
    def edit_shapes(edit):
        def mutate(s):
            rows=s[4]['collision_shapes']; edit(rows)
            s[5]['getCollisionShapeData']['return']=deepcopy(rows)
        return mutate
    def swap(rows):rows[0][3:],rows[1][3:]=rows[1][3:],rows[0][3:]
    shape_cases={
        'wrong_row_count':lambda x:x.pop(),'wrong_child_order_preserving_headers':swap,
        'wrong_child_type':lambda x:x[11].__setitem__(2,5),
        'wrong_child_dimensions':lambda x:x[200][3].__setitem__(1,.123),
        'wrong_child_position':lambda x:x[200][5].__setitem__(1,.123),
        'wrong_child_quaternion':lambda x:x[200][6].__setitem__(0,.2),
        'wrong_first_row_body':lambda x:x[0].__setitem__(0,71),
        'wrong_first_row_link':lambda x:x[0].__setitem__(1,19),
    }
    for name,mutate in shape_cases.items():case(name,'014 geometry / coupled mutation',compound,False,edit_shapes(mutate))
    case('later_compound_headers_non_authoritative','014 actual / header-only mutation',compound,True,edit_shapes(lambda x:x[11].__setitem__(slice(0,2),[987654,-987654])))
    def shift_bounds(s):
        s[4]['actual_AABB'][1][2]+=.0002
        s[5]['getAABB']['return']=deepcopy(s[4]['actual_AABB'])
    case('wrong_world_aabb','017 synthetic / coupled mutation',other,False,shift_bounds)
    case('named_mesh_reported_as_anonymous','021 actual / coupled mutation',mesh,False,edit_shapes(lambda x:x[0].__setitem__(4,'unknown_file')))
    case('wrong_named_obj_path','021 actual / coupled mutation',mesh,False,edit_shapes(lambda x:x[0].__setitem__(4,'unregistered/collision.obj')))
    case('wrong_obj_content_sha','021 actual / mutated',mesh,False,lambda s:s[4]['mesh_proof'].__setitem__('source_sha256','f'*64))
    case('compound_fields_on_named_mesh','021 actual / mixed schema',mesh,False,lambda s:s[3].__setitem__('fields',deepcopy(compound[3]['fields'])))
    case('mesh_fields_on_compound','014 actual / mixed schema',compound,False,lambda s:s[3].__setitem__('fields',deepcopy(mesh[3]['fields'])))
    case('missing_asset_specific_field','021 actual / mutated',mesh,False,lambda s:s[3]['fields'].__setitem__(slice(None),[r for r in s[3]['fields'] if not r['field'].endswith('.mesh_filename')]))
    case('wrong_mesh_scale','021 actual / coupled mutation',mesh,False,edit_shapes(lambda x:x[0][3].__setitem__(0,1.02)))
    case('wrong_mesh_pose','021 actual / coupled mutation',mesh,False,edit_shapes(lambda x:x[0][5].__setitem__(1,.001)))
    # Prove coverage over every logical field, including all asset-specific names.
    for sample, kind in [(mesh,'named_mesh'),(compound,'compound_boxes'),(other,'compound_boxes')]:
        defs,binding,asset,_=verifier.registration(sample[1],sample[2])
        assert asset['mode']==kind
        names=verifier.named_fields(defs,sample[2],binding['object_id'],kind)
        formats=verifier.named_fields({**defs,'common_required':defs['common_value_format'],'field_atol_source':defs['specific_value_format']},sample[2],binding['object_id'],kind)
        records=sample[3]['fields']
        for operator in ('delete','duplicate'):
            erroneously_accepted=[]
            for idx,row in enumerate(records):
                corrupt=records[:idx]+records[idx+1:] if operator=='delete' else records+[row]
                try: verifier.verify_fields(corrupt,names,formats)
                except ValueError: continue
                erroneously_accepted.append(row['field'])
            results.append(dict(case='every_field_'+operator,sample=binding['object_id']+' independent mutations',
                                expected_accept=False,actual_accept=bool(erroneously_accepted),passed=not erroneously_accepted,
                                mutations_checked=len(records),wrongly_accepted=erroneously_accepted))
    # A hypothetical pre-registered anonymous schema is an isolated test dependency.
    # No such asset is registered for this batch. Retain native hull/OBJ binding.
    defs,_,asset,_=verifier.registration(mesh[1],mesh[2])
    prospective_defs=deepcopy(defs)
    prospective_defs['assets'][asset['urdf_relative']]['mode']='anonymous_mesh'
    actual_read=verifier.read
    def test_only_read(p):
        return prospective_defs if Path(p).resolve()==verifier.DEFINITION.resolve() else actual_read(p)
    anonymous=deepcopy(mesh)
    anonymous[3]['fields']=[r for r in anonymous[3]['fields'] if not r['field'].endswith('.mesh_filename')]
    with patch.object(verifier,'read',test_only_read):
        case('anonymous_schema_rejects_named_endpoint','021 hypothetical pre-registered anonymous branch',anonymous,False)
        transform=edit_shapes(lambda x:x[0].__setitem__(4,'unknown_file'))
        case('anonymous_uses_urdf_obj_and_native_hull','021 hypothetical branch with anonymized endpoint',anonymous,True,transform)
        invalid=deepcopy(anonymous);transform(invalid)
        def corrupt_hull(s):
            s[4]['mesh'][1][0][0]+=.1
            s[5]['getMeshData']['return']=deepcopy(s[4]['mesh'])
        case('anonymous_bad_hull_fails','021 hypothetical branch, coupled native and report mutation',invalid,False,corrupt_hull)
    # Change bytes presented to the verifier without touching the registered assets.
    # This exercises the *actual* verifier hash path, independently of receipt mutations.
    for asset_file,source_sample in [
        (ROOT/'data/simulation_v52/objects/voxel_repair/017/object.urdf',other),
        (ROOT/'data/simulation_v52/objects/final_static_audit/021/collision.obj',mesh),
    ]:
        original_open = Path.open
        corrupted = asset_file.read_bytes() + b'\n'
        substitutions = []
        def altered_open(path,mode='r',*args,**kwargs):
            if path.resolve()==asset_file.resolve() and mode=='rb':
                substitutions.append(True)
                return io.BytesIO(corrupted)
            return original_open(path,mode,*args,**kwargs)
        with patch.object(Path,'open',altered_open):
            case('changed_registered_file_bytes_'+asset_file.suffix[1:],'simulated file-content corruption, no disk changes',source_sample,False)
        results[-1]['hash_path_reads_intercepted'] = len(substitutions)
        results[-1]['passed'] = results[-1]['passed'] and bool(substitutions)
    return results

def main():
    rows=run(); passed=all(r['passed'] for r in rows)
    path=HERE/'schema_replay19.json'
    sources=[OLD17/'static_preflight/zero_readback.json', OLD17/'static_preflight/zero_step/object021_readback.json', OLD17/'static_preflight/full_api/calls.jsonl.gz',OLD18/'slots/08/static_preflight/zero_readback.json', OLD18/'slots/08/static_preflight/zero_step/object014_readback.json',OLD18/'slots/08/static_preflight/full_api/calls.jsonl.gz', HERE/'slots/09/preregistration.json',ROOT/'data/simulation_v52/objects/voxel_repair/017/object.urdf']
    result=dict(version=HERE.name,passed=passed,cases=rows,pass_count=sum(r['passed'] for r in rows),case_count=len(rows),
                sources={p.relative_to(ROOT).as_posix():digest(p) for p in sources},physical_calls=0,connections=0,
                provenance='021 and 014 are archived receipts and native return prefixes; 017 is a constructed URDF-derived 640-box regression fixture, not an engine measurement.')
    with path.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    for r in rows:
        if not r['passed']: print(r)
    print(result['pass_count'], '/',len(rows))
    return 0 if passed else 2

if __name__=='__main__':
    raise SystemExit(main())
