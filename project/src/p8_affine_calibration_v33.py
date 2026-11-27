"""A single preregistered convex two-scalar monotone calibrator."""
import numpy as np
from scipy.special import expit

def objective(parameters,z,y,penalty=.001):
 a,b=parameters;q=a*z+b;error=expit(q)-y
 value=np.mean(np.logaddexp(0,q)-y*q)+penalty/2*((a-1)**2+b*b)
 gradient=np.array([np.mean(error*z)+penalty*(a-1),np.mean(error)+penalty*b],float)
 return float(value),gradient

def transform(parameters,z):return parameters[0]*np.asarray(z,dtype=np.float64)+parameters[1]

def holdout_masks(scene_ids,held_scene):
 ids=np.asarray(scene_ids);train=ids!=held_scene;held=ids==held_scene
 if not train.any() or not held.any():raise ValueError('Both calibration training and held-out scenes are required')
 return train,held

def projected_gradient(parameters,gradient,bounds=((.05,20.),(-3.,3.))):
 result=[]
 for x,g,(lo,hi) in zip(parameters,gradient,bounds):
  result.append(min(0.,float(g)) if abs(x-lo)<1e-8 else max(0.,float(g)) if abs(x-hi)<1e-8 else float(g))
 return max(abs(g) for g in result)
