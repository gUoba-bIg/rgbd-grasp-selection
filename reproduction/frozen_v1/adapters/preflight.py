"""Imports only; validates OS fingerprint and installed role inputs without touching a model or a physics connection."""
import os,sys,hashlib,json,re,csv,base64,subprocess
from pathlib import Path
import importlib.metadata as md
T=Path(sys.argv[1]).resolve();role=sys.argv[2];B=Path(__file__).resolve().parents[1]
P=T/'p';V=P/'IP140_R2_R6formal720_static_distance_uncertified_resume01';F=P/'IP140_R2_R4dev72_cross_dim_explicit_resume01'
assert Path(sys.prefix).resolve()==T/'env'/role and Path(sys.base_prefix).resolve()==T/'bin'
sys.path[:0]=list(map(str,[V/'code',V/'implementation',F/'implementation',P]))
os.environ['PROGRAMFILES']=str(P/'vendor/ProgramFiles')
dll_dirs=[T/'bin',T/'bin/DLLs',P/'vendor/ProgramFiles/Git/mingw64/bin',T/'env'/role/'Lib/site-packages/torch/lib']
handles=[os.add_dll_directory(str(x)) for x in dll_dirs if x.is_dir()]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').digest()
checked=0;unhashed=[]
for dist in md.distributions():
    for f in dist.files or []:
        path=Path(dist.locate_file(f)).resolve()
        if not path.is_relative_to(T):raise RuntimeError(('external installed file',f,path))
        if not f.hash:
            unhashed.append(str(path.relative_to(T)));continue
        if f.hash.mode!='sha256' or base64.urlsafe_b64encode(sha(path)).decode().rstrip('=')!=f.hash.value:raise RuntimeError(('record mismatch',str(path)))
        checked+=1
if role=='vision':
    import torch
    from src.grasp_prerequisite_data_v27 import SOURCE
    from graspnet import GraspNet,pred_decode
    from src.modal_fusion_v38 import ModalFusionV38
    from policy_network import generate
    from src.p5_quality_v23 import corrupt
    from src.risk_observations_v31 import points
    from scripts.extract_mask_scores_v17 import decode_with_scores
    assert torch.__version__=='2.6.0+cu126' and torch.version.cuda=='12.6' and torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,9)
    capture=json.loads((B/'metadata/platform_capture.json').read_bytes())
    smi=Path(os.environ['SYSTEMROOT'])/'System32/nvidia-smi.exe'
    gpu=subprocess.check_output([str(smi),'--query-gpu=name,driver_version,memory.total,compute_cap','--format=csv,noheader'],encoding='utf8').strip()
    ref=[v.strip() for v in capture['gpu']['stdout'].split(',')]
    if [v.strip() for v in gpu.split(',')]!=[ref[i] for i in (0,2,3,4)]:raise RuntimeError('OS driver and GPU mismatch: '+gpu)
else:
    from task_worker import bind_engine
    bind_engine()
    import pybullet
    # Merely import the exact native extension: no connect, no model load or stepping.
    import platform_trial, platform_trace
    if role=='rank':import policy_rank
print(json.dumps(dict(role=role,checked_wheel_RECORDs=checked,unhashed_RECORD_entries=unhashed,unhashed_note='pip-generated RECORD and bytecode entries where present',native_engine=str(getattr(sys.modules.get('pybullet'),'__file__','')) or None,torch=getattr(sys.modules.get('torch'),'__version__',None),gpu=locals().get('gpu'),physical_steps=0,model_forwards=0),ensure_ascii=True))
