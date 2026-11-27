"""Read-only observers for the independent smoke budget and the actual top-level model tensors."""
import hashlib, json, os
from contextlib import contextmanager
@contextmanager
def count_models(out,reference):
    import torch, numpy as np
    from src.grasp_prerequisite_data_v27 import SOURCE
    from graspnet import GraspNet
    from src.modal_fusion_v38 import ModalFusionV38
    identities={GraspNet:'P3',ModalFusionV38:'P5'};counts={}
    with (out/'model_invocations.jsonl').open('x',encoding='utf8') as log:
        def record(item):
            log.write(json.dumps(item,sort_keys=True)+'\n');log.flush();os.fsync(log.fileno())
        def before(mod,args):
            tag=identities.get(type(mod))
            if not tag:return
            if tag in counts:raise RuntimeError('second model call forbidden')
            tensor=args[0]['point_clouds'] if tag=='P3' else args[0]
            v=np.ascontiguousarray(tensor.detach().cpu().numpy())
            fp=dict(shape=list(v.shape),dtype=v.dtype.str,sha256=hashlib.sha256(v.tobytes()).hexdigest())
            record({'event':'input_check','model':tag,'observed':fp,'match':reference[tag]==fp})
            if fp!=reference[tag]:raise RuntimeError('registered tensor identity '+tag)
            counts[tag]=1;record({'event':'entered','model':tag})
        def after(mod,args,ret):
            tag=identities.get(type(mod))
            if tag:record({'event':'returned','model':tag})
        a=torch.nn.modules.module.register_module_forward_pre_hook(before)
        b=torch.nn.modules.module.register_module_forward_hook(after)
        try:yield
        finally:a.remove();b.remove()
