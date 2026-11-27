from seal_io import *
driver=next(x for x in (Path(os.environ['SystemRoot'])/'System32/DriverStore/FileRepository').glob('nv_dispig.inf_amd64_*') if sha(x/'nv_dispig.inf')==sha(Path(os.environ['SystemRoot'])/'INF/oem49.inf'))
files={str(p.relative_to(driver).as_posix()):(p,None) for p in walk(driver)}
pack('signed_display_driver',files)
put(REPO/'metadata/display_driver.json',dict(installed_inf='oem49.inf',upstream_inf='nv_dispig.inf',folder=driver.name,inf_sha256=sha(driver/'nv_dispig.inf'),driver_version='32.0.16.1088',provider='NVIDIA',display='RTX 4060 Ti',hardware_scope='Windows NVIDIA matched driver; stored, never installed or changed by setup'))
extras={}
for key,phase in [(P0,'source_R6_initial'),(R6,'source_R6_final'),(R7,'source_R7_final'),(R8,'source_R8_paper')]:
    with gzip.open(REPO/'locks'/(phase+'.jsonl.gz'),'rt',encoding='utf8') as f:prior={json.loads(l)[0] for l in f}
    for file in walk(ROOT/key):
        rel=file.relative_to(ROOT).as_posix()
        if rel not in prior:extras[rel]=(file,None)
pack('source_supplement',extras)
