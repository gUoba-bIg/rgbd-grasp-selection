"""Versioned pure support calculation; no physics or control side effects."""
from math import isfinite, hypot
OBJECT_MASS_KG = .04
GRAVITY_WORLD_M_S2 = (0., 0., -9.81)
SUPPORT_THRESHOLD_N = .2
PENETRATION_LIMIT_M = .001
SCHEMA = "IP140-smoke13-aggregate-table-support-v1"

def finite(values):
    return all(isinstance(v,(int,float)) and isfinite(v) for v in values)

def verify_physics_context(mass, gravity):
    if (not isinstance(gravity,(list,tuple)) or len(gravity)!=3
            or not finite((mass,*gravity)) or mass!=OBJECT_MASS_KG
            or tuple(gravity)!=GRAVITY_WORLD_M_S2):
        raise ValueError("aggregate support requires frozen actual-readback mass .04 kg and gravity (0,0,-9.81)")
    return mass*abs(gravity[2])

def aggregate_table_contacts(raw, object_body, table_body, *, object_mass_kg=OBJECT_MASS_KG,
                             gravity_world_m_s2=GRAVITY_WORLD_M_S2):
    weight=verify_physics_context(object_mass_kg,gravity_world_m_s2)
    if object_body==table_body or min(object_body,table_body)<0:
        raise ValueError("unique actual support body identities required")
    present=False;forces=[];indices=[];problems=[]
    for i,c in enumerate(raw):
        if not isinstance(c,(list,tuple)) or len(c)!=14:
            problems.append("nonfinite_or_incomplete_contact")
            continue
        # Unrelated contacts are handled independently by the complete scene classifier.
        if {c[1],c[2]}!={object_body,table_body}:continue
        present=True
        if any(not isinstance(c[k],(list,tuple)) or len(c[k])!=3 for k in (5,6,7,11,13)):
            problems.append("nonfinite_or_incomplete_contact");continue
        values=(*c[:5],*c[5],*c[6],*c[7],*c[8:11],*c[11],c[12],*c[13])
        if not finite(values):
            problems.append("nonfinite_or_incomplete_contact");continue
        invalid=False
        if c[3]!=-1 or c[4]!=-1:
            problems.append("wrong_support_link");invalid=True
        if c[8]<-PENETRATION_LIMIT_M:
            problems.append("external_penetration_below_minus_1mm");invalid=True
        if invalid:continue
        forces.append(max(c[9],0.));indices.append(i)
    total=sum(forces)
    if not isfinite(total):
        problems.append("nonfinite_aggregate_table_force")
        total=None
    fraction=None if total is None else total/weight
    if fraction is not None and not isfinite(fraction):
        problems.append("nonfinite_aggregate_table_load_fraction")
        fraction=None
    return dict(table_contact_present=present,table_normal_force_sum_N=total,
                table_load_fraction=fraction,
                aggregate_table_support=(None if problems else total>SUPPORT_THRESHOLD_N),
                aggregation_valid=not problems,aggregation_violations=list(dict.fromkeys(problems)),
                legal_table_contact_indices=indices,positive_table_normal_forces_N=forces,
                aggregate_support_schema=SCHEMA,object_mass_kg=object_mass_kg,
                gravity_world_m_s2=list(gravity_world_m_s2),weight_N=weight)
