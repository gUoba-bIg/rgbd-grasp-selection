"""Read checkpoint/export equality with no forward or optimization calls."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p7_common_v28 import *

def run():
 import numpy as np,torch
 c=verify();r=load_sealed(BASE/'fit_result.json')
 if not r['fitted']:return
 p=ROOT/'experiments/risk_v28'/f"epoch_{r['selected_epoch']:02d}.pt";need(stat(p)==r['selected_checkpoint'],'Selected checkpoint differs');state=torch.load(p,map_location='cpu',weights_only=False);a=np.load(BASE/'selected_weights.npz');need(set(a.files)==set(state['model']) and all(np.array_equal(a[k],v.numpy()) for k,v in state['model'].items()),'Exported weights differ')
 for split in c['groups']:
  d=np.load(BASE/'predictions'/f'{split}.npz');need(np.array_equal(d['mean'],state['mean']) and np.array_equal(d['scale'],state['scale']),'Checkpoint standardization differs')
 sealed(BASE/'model_identity.json',dict(passed=True,checkpoint=stat(p),weights=stat(BASE/'selected_weights.npz'),new_forwards=0,new_updates=0))

if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()
