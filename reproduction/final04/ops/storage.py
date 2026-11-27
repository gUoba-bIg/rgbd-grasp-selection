"""Read-only sizing, excludes junction targets from totals; reserves the registered cold evidence maximum."""
import os,stat,time,json,shutil
from pathlib import Path
V=Path(__file__).resolve().parents[1];P=V.parent;previous=P/'IP140_R3_V2_original_title_recovery01'
roots=[previous/'v2_control02',previous/'recovery02_ops',previous/'V1_repro_adapter02',previous/'c3',P/'V1_repro_dependency_closure03',V,Path('E:/IP140_R3_V2_original_title_recovery01/V1_archive_02'),Path('E:/IP140_R3_V2_original_title_recovery01/V2_work02')]
def size(path):
    n=0
    if path.exists():
        for dn,ds,fs in os.walk(path,followlinks=False):
            ds[:]=[x for x in ds if not ((Path(dn)/x).is_junction() or (Path(dn)/x).is_symlink())]
            for x in fs:
                f=Path(dn)/x
                if not f.is_symlink():n+=((f.stat().st_size+4095)//4096)*4096
    return n
usage={str(p):size(p) for p in roots};cap=2*2**30;free={d:shutil.disk_usage(d+':/').free for d in ('C','E')}
passed=sum(usage.values())+cap<=150*2**30 and min(free.values())>=100*2**30+cap
with (V/'audit/pre_cold_storage.json').open('x',encoding='utf8')as f:json.dump(dict(utc_epoch=time.time(),allocated_upper_estimate=usage,case_output_reservation_bytes=cap,global_limit_bytes=150*2**30,volume_free_bytes=free,passed=passed),f,indent=2,ensure_ascii=False)
print('allocated GiB',sum(usage.values())/2**30,'free GiB', {x:y/2**30 for x,y in free.items()},'pass',passed)
if not passed:raise SystemExit(2)
