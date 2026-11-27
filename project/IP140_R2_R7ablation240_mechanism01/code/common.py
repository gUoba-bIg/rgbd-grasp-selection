from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os
HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parent
R4=ROOT/"IP140_R2_R4dev72_cross_dim_explicit_resume01"
R5=ROOT/"IP140_R2_R5shadow24_observed_contactability_admission01"
PLATFORM=ROOT/"IP140_R2_smoke22_miss_negative_controls_full_execution01"
BINARY=ROOT/"runtime/simulation_v52e/venv/Lib/site-packages/pybullet.cp312-win_amd64.pyd"

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def put(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf8",newline="\n") as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False,separators=(",",":"));f.write("\n");f.flush();os.fsync(f.fileno())
    return sha(p)
def rep(p,value):
    p=Path(p);tmp=p.with_suffix(".writing");put(tmp,value);os.replace(tmp,p)
def require(ok,message):
    if not ok:raise ValueError(message)
def workspace_ok():
    cwd=Path.cwd().resolve()
    return cwd.parent==HERE/"workspaces" and (cwd/"workspace.json").is_file() and all((cwd/n).resolve()==q for n,q in [("data",ROOT/"data"),(PLATFORM.name,PLATFORM)])
def scene_binding(key):
    d=HERE/"preregistration/design.json";seal=read(HERE/"preregistration/design_seal.json")
    require(sha(d)==seal["design_sha256"],"scene registry hash drift")
    scenes=read(d)["scenes"]
    s=next(x for x in scenes if x["base_scene"]==key)
    b=s["asset"]
    require(sha(ROOT/b["urdf_relative"])==b["urdf_sha256"],"object URDF")
    if "obj_sha256" in b:require(sha(ROOT/b["obj_relative"])==b["obj_sha256"],"object collision OBJ")
    return s
