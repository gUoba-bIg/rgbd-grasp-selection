"""Independent observation/seed reuse audit after settlement, before any grasp outcomes."""
import gzip
import numpy as np
from common import *
def arr(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def obs(path):
    with np.load(path,allow_pickle=False) as o:
        require(set(o.files)=={'rgb','depth_buffer','view','projection'},'unexpected policy observation fields')
        return {k:arr(o[k]) for k in sorted(o.files)}
def main():
    d=read(HERE/'preregistration/design.json')
    old={}
    for hist in (R4,R5):
        for p in (hist/'scenes').glob('*/observed_rgbd.npz'):old[p.relative_to(ROOT).as_posix()]=obs(p)
    ex=set((z['rgb'],z['depth_buffer']) for z in old.values())
    calib={}
    for p in (R5/'calibration').glob('*frames.jsonl.gz'):
        with gzip.open(p,'rt',encoding='utf8') as fin:
            for line in fin:
                r=json.loads(line);calib[int(r['seed'])]=r['input_hashes']
    def digests(t):
        if isinstance(t,str):return {t} if len(t)==64 else set()
        if isinstance(t,(dict,list)):return set().union(*(digests(x) for x in (t.values() if isinstance(t,dict) else t)))
        return set()
    registered=read(R5/'admission/seed_registry.json')['scene_identity']
    calibration_fingerprints=digests(list(calib.values()))
    require(set(calib)==set(registered['calibration']+registered['holdout']),'calibration source incomplete')
    out={};newpairs=set();poses=set()
    for s in d['scenes']:
        key=s['base_scene'];base=HERE/'scenes'/key
        src=base/'observed_rgbd.npz';stable=read(base/'settled_state.json');r=read(base/'receipt.json');ss=read(base/'api/summary.json')
        a=obs(src);pair=(a['rgb'],a['depth_buffer'])
        require(pair not in ex and pair not in newpairs and not set(pair)&calibration_fingerprints,'duplicate physical observation '+key)
        pose=(s['object_id'],*stable['object_pose'][0],*stable['object_pose'][1]);require(pose not in poses,'duplicate stable pose '+key)
        newpairs.add(pair);poses.add(pose)
        require(r['actual_steps']==r['step_returns']==ss['actual_native_step_calls']==240 and ss['calls']==ss['returns'] and not ss['exceptions'] and r['method_trials']==r['grasp_outcomes']==0,'unknown scene acquisition')
        require(sha(src)==r['observation_rgbd_sha256'] and sha(base/'settled_state.json')==r['settled_state_sha256'] and r['full_native_api_summary_sha256']==sha(base/'api/summary.json'),'scene evidence hash')
        out[key]=dict(array_sha256=a,observation_sha256=sha(src),state_sha256=r['settled_state_sha256'],native_log_sha256=sha(base/'api/calls.jsonl.zst'),receipt_sha256=sha(base/'receipt.json'),sampling_seed=s['sampling_seed'],condition_seed=s['condition_seed'])
    put(HERE/'audit/scene_inventory.json',dict(utc=now(),scenes=out,history_rgbd=old,calibration_input_hashes=calib,scope='Scene images and depth compared by original pixel bytes. Calibration fiducials are independent parametric assets; hashes of available calibration pixel arrays also checked.',history_source_sha256=sha(R5/'admission/seed_registry.json'),unique_new_scenes=len(out),historical_camera_scenes=len(old),calibration_frames=len(calib)))
if __name__=='__main__':main()
