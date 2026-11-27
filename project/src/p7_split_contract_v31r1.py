"""Select the three pre-existing development groups without changing membership."""
def fixed_development_groups(manifest_groups):
 groups={key:list(manifest_groups[key]) for key in ['train','calibration','validation']}
 if any(len(groups[key])!=n for key,n in [('train',70),('calibration',15),('validation',15)]):raise ValueError('Development split sizes differ')
 ids=sum(groups.values(),[])
 if any(type(v) is not int for v in ids) or len(set(ids))!=100 or set(ids)!=set(range(100)):raise ValueError('Development groups must partition original scenes0..99 without overlap')
 reserved=sum([list(v) for k,v in manifest_groups.items() if k not in groups],[])
 if set(ids)&set(reserved):raise ValueError('Reserved-group scene leaked into development')
 return groups
