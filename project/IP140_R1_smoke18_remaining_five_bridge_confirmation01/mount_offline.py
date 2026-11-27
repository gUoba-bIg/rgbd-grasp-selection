"""Synthetic admission cases over the exact pure fixed-world mount validator."""
from copy import deepcopy
import json
import math
from pathlib import Path

from base_table_mount import validate
from object_evaluator import validate_filter_calls
from structural_filter import STRUCTURAL_PAIRS

HERE=Path(__file__).resolve().parent


def reversed_row(row):
    row[1],row[2]=row[2],row[1]
    row[3],row[4]=row[4],row[3]
    row[5],row[6]=row[6],row[5]
    row[7]=[-x for x in row[7]]
    row[11]=[-x for x in row[11]]
    row[13]=[-x for x in row[13]]


def mount_cases(links):
    spec=json.loads((HERE/'base_table_registration.json').read_bytes())
    baseline=spec['baseline_observation']
    def case(name,mutate=None,expected=False,stage='pre'):
        observed=deepcopy(baseline)
        observed['stage']=stage
        if stage!='pre':
            observed['manifolds']=[]
        if mutate:
            mutate(observed)
        result=validate(observed)
        return dict(name=name,passed=result['passed'] is expected,expected_admission=expected,
                    input_observation=observed,actual=result,synthetic=True,
                    source='smoke11 zero-step observations plus explicitly injected mutations')

    yield case('mount_prefilter_registered_evidence',expected=True)
    yield case('mount_postfilter_invariant',expected=True,stage='post')
    yield case('mount_dynamic_invariant',expected=True,stage='dynamic')
    def reverse(observed):
        for key in ('closest','manifolds'):
            for row in observed[key]:
                reversed_row(row)
    yield case('mount_native_endpoint_order_reversed',reverse,True)
    yield case('mount_postfilter_endpoint_order_reversed',reverse,True,stage='post')
    for kind,key,position in [('body','closest',1),('table','closest',2),('base_link','closest',3),
                              ('table_link','closest',4),('manifold_body','manifolds',1)]:
        def wrong(observed,key=key,position=position):
            observed[key][0][position]+=57
        yield case('mount_reject_wrong_'+kind,wrong)
    for key,value in [('robot',71),('table',72),('base_index',73)]:
        yield case('mount_reject_context_'+key,lambda o,k=key,v=value:o.update({k:v}))
    def other_pair(observed):
        other=deepcopy(observed['manifolds'][0])
        other[3]=links['ip140_palm']
        other[9]=other[10]=other[12]=0.
        observed['manifolds'].append(other)
    yield case('mount_other_robot_table_zero_force_fails',other_pair)
    for sign in (-1,1):
        def drift(observed,sign=sign):
            for row in observed['closest']:
                shift=sign*2e-9
                row[8]+=shift
                row[5]=[a+shift*n for a,n in zip(row[5],row[7])]
        yield case('mount_reject_distance_drift_'+('minus' if sign<0 else 'plus'),drift,stage='dynamic')
    def translation(observed):
        observed['base_W_T'][0][3]+=2e-9
    yield case('mount_reject_relative_translation',translation,stage='dynamic')
    def rotation(observed):
        theta=2e-9;c=math.cos(theta);s=math.sin(theta)
        matrix=observed['base_W_T']
        old=deepcopy(matrix)
        for col in range(3):
            matrix[0][col]=c*old[0][col]-s*old[1][col]
            matrix[1][col]=s*old[0][col]+c*old[1][col]
    yield case('mount_reject_relative_rotation',rotation,stage='dynamic')
    def out_of_face(observed):
        row=observed['closest'][0]
        row[5][0]+=10.
        row[6][0]+=10.
    yield case('mount_reject_witness_outside_region',out_of_face)
    def above_table(observed):
        row=observed['closest'][0]
        row[5][2]+=2e-9
        row[6][2]+=2e-9
    yield case('mount_reject_witness_above_top_face',above_table)
    def nonunit(observed):
        observed['closest'][0][7]=[0.,0.,.5]
    yield case('mount_reject_nonunit_normal',nonunit)
    def wrong_direction(observed):
        row=observed['closest'][0]
        row[7]=[0.,0.,-1.]
        row[5]=[a+row[8]*n for a,n in zip(row[6],row[7])]
    yield case('mount_reject_wrong_normal_direction',wrong_direction)
    yield case('mount_reject_nonfinite_distance',lambda o:o['closest'][0].__setitem__(8,float('nan')))
    yield case('mount_reject_nonfinite_force',lambda o:o['manifolds'][0].__setitem__(10,float('inf')))
    yield case('mount_reject_nonfinite_transform',lambda o:o['base_W_T'][0].__setitem__(3,float('inf')))
    def bad_witness_residual(observed):
        observed['closest'][0][5][0]+=2e-9
    yield case('mount_reject_full_witness_residual',bad_witness_residual)
    for i,label in [(9,'normal'),(10,'tangent1'),(12,'tangent2')]:
        yield case('mount_reject_prefilter_'+label+'_force',lambda o,k=i:o['manifolds'][0].__setitem__(k,2e-9))
    yield case('mount_reject_prefilter_impulse',lambda o:o['manifolds'][0].__setitem__(9,3e-10))
    yield case('mount_missing_closest_rejected',lambda o:o.update(closest=[]),stage='dynamic')
    yield case('mount_dynamic_reappearance',lambda o:o.update(manifolds=deepcopy(baseline['manifolds'])),stage='dynamic')
    yield case('mount_postfilter_reappearance',lambda o:o.update(manifolds=deepcopy(baseline['manifolds'])),stage='post')
    yield case('mount_changed_table_mass_rejected',lambda o:o['table_dynamics'].__setitem__(0,1.))
    yield case('mount_changed_base_margin_rejected',lambda o:o['base_dynamics'].__setitem__(11,.002))

    robot,target,table=0,91,92
    calls=[dict(method='setCollisionFilterPair',sequence=i+1,args=[robot,robot,links[a],links[b],0],
                kwargs={'physicsClientId':0},returned=True,return_value=None) for i,(a,b) in enumerate(STRUCTURAL_PAIRS)]
    calls.append(dict(method='setCollisionFilterPair',sequence=14,
                      args=[robot,table,links['base_link_inertia'],-1,0],
                      kwargs={'physicsClientId':0},returned=True,return_value=None))
    def filter_case(name,mutate=None,expected=False):
        actual_calls=deepcopy(calls)
        if mutate:mutate(actual_calls)
        observed=validate_filter_calls(robot,target,table,links,actual_calls,require_mount=True)
        return dict(name=name,passed=observed['passed'] is expected,expected_admission=expected,
                    actual=observed,filter_calls=actual_calls,synthetic=True)
    yield filter_case('full14_exact_filter_sequence',expected=True)
    yield filter_case('full14_reject_missing_mount',lambda c:c.pop())
    yield filter_case('full14_reject_duplicate_mount',lambda c:c.__setitem__(12,dict(c[-1],sequence=13)))
    yield filter_case('full14_reject_mount_enabled',lambda c:c[-1]['args'].__setitem__(4,1))
    yield filter_case('full14_reject_fifteenth_filter',lambda c:c.append(dict(c[-1],sequence=15)))
    yield filter_case('full14_reject_other_robot_table',lambda c:c[-1]['args'].__setitem__(2,links['ip140_palm']))
    yield filter_case('full14_reject_object_table',lambda c:c[-1].update(args=[target,table,-1,-1,0]))
    yield filter_case('full14_reject_robot_object',lambda c:c[-1].update(args=[robot,target,links['base_link_inertia'],-1,0]))
    yield filter_case('full14_reject_unknown_environment',lambda c:c[-1]['args'].__setitem__(1,93))
    yield filter_case('full14_reject_missing_return',lambda c:c[-1].update(returned=False))
