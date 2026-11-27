"""Create-only supplementary evidence. All scientific inputs are bound to the V1 tag."""
from datetime import datetime, timezone
import ctypes as c
from ctypes import wintypes as w
from functools import lru_cache
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

NAME="V1_supplement_robustness_admission01"
WORK=Path(__file__).resolve().parents[1]
ROOT=WORK.parents[1]
OUT=ROOT/"outputs"/NAME
C=OUT/"c"
R6NAME="IP140_R2_R6formal720_static_distance_uncertified_resume01"
R6FIRST="IP140_R2_R6formal720_raw_a_c0_paired01"
R7NAME="IP140_R2_R7ablation240_mechanism01"
R4="IP140_R2_R4dev72_cross_dim_explicit_resume01"
PL="IP140_R2_smoke22_miss_negative_controls_full_execution01"
KIT=ROOT/"V1_repro_final04/adapter"
ARCHIVES=ROOT/"IP140_R3_V2_original_title_recovery01/v1_archive"
TAG="paper_v1_r6r7_frozen"
COMMIT="e184a77ab06fe1f1c0a0f1514b04da65befcf8ed"
ENGINE="882cd523a255e66c94d5b91c119f9c688fdbc0956161eff30f2171e21677ab49"
METHODS=("Raw","A","C0")
MODES=("clean","depth_only","shift_only","missing_only","block_only","joint")
def need(test,what):
    if not test:raise RuntimeError(what)
def utc():return datetime.now(timezone.utc).isoformat()
def read(p):
    def bad(x):raise ValueError("nonfinite JSON "+x)
    return json.loads(Path(p).read_bytes(),parse_constant=bad)
def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def write(rel,obj):
    p=(OUT/rel).resolve()
    need(p.is_relative_to(OUT),"write scope")
    body=(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2)+"\n").encode()
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("xb") as f:f.write(body);f.flush();os.fsync(f.fileno())
    return hashlib.sha256(body).hexdigest()
@lru_cache
def index(name):
    rec=read(WORK/"locks"/(name+".receipt.json"))
    p=WORK/rec["member_manifest"];need(sha(p)==rec["member_manifest_sha256"],"V1 lock index "+name)
    with gzip.open(p,"rt",encoding="utf8") as f:rows=[json.loads(l) for l in f if l.strip()]
    items={n:(size,h) for n,size,h in rows}
    need(len(items)==len(rows)==rec["members"],"manifest count "+name)
    return items
def validate_file(path,size,h):
    if not path.is_file():
        raise FileNotFoundError(f"frozen file absent: {path}; expected {size} bytes, sha256 {h}")
    data_size=path.stat().st_size
    actual=sha(path)
    need((data_size,actual)==(size,h),f"frozen file mismatch: {path}; expected ({size}, {h}), observed ({data_size}, {actual})")
def volume():
    k=c.WinDLL("kernel32",use_last_error=True)
    k.CreateFileW.restype=w.HANDLE
    k.CreateFileW.argtypes=[w.LPCWSTR,w.DWORD,w.DWORD,c.c_void_p,w.DWORD,w.DWORD,w.HANDLE]
    k.DeviceIoControl.argtypes=[w.HANDLE,w.DWORD,c.c_void_p,w.DWORD,c.c_void_p,w.DWORD,c.POINTER(w.DWORD),c.c_void_p]
    h=k.CreateFileW(r"\\.\E:",0x20,7,None,3,0,None)
    if h==c.c_void_p(-1).value:raise c.WinError(c.get_last_error())
    flags=w.DWORD();sz=w.DWORD()
    try:
        if not k.DeviceIoControl(h,0x90078,None,0,c.byref(flags),4,c.byref(sz),None):raise c.WinError(c.get_last_error())
    finally:k.CloseHandle(w.HANDLE(h))
    need(sz.value==4 and flags.value==0,"E: NTFS dirty")
    return {"dirty":flags.value,"free_bytes":shutil.disk_usage("E:/").free}
def disk():
    baseline=read(OUT/"admission/disk_start.json")["windows"]["Boot"]
    boot=datetime.fromisoformat(baseline.replace("Z","+00:00")).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:23]+"Z"
    selectors=[
        "((Provider[@Name='disk'] or Provider[@Name='Microsoft-Windows-Disk']) and (EventID=51 or EventID=153))",
        "((Provider[@Name='Ntfs'] or Provider[@Name='Microsoft-Windows-Ntfs']) and (EventID=50 or EventID=140))",
        "((Provider[@Name='stornvme'] or Provider[@Name='Microsoft-Windows-StorNVMe']) and (EventID=11 or EventID=129))"
    ]
    query="*[System[TimeCreated[@SystemTime >= '"+boot+"'] and ("+" or ".join(selectors)+")]]"
    ps=("$ErrorActionPreference='Stop'; $q=\""+query+"\"; try {$events=@(Get-WinEvent -LogName System -FilterXPath $q -MaxEvents 10 -ErrorAction Stop | "
        "ForEach-Object { [pscustomobject]@{id=$_.Id;provider=$_.ProviderName;record=$_.RecordId;utc=$_.TimeCreated.ToUniversalTime().ToString('o')} })} "
        "catch {if(($_.FullyQualifiedErrorId -split ',')[0] -ne 'NoMatchingEventsFound'){throw};$events=@()}; ConvertTo-Json -InputObject $events -Compress")
    status=volume()
    events=json.loads(subprocess.check_output(["powershell","-NoProfile","-NonInteractive","-Command",ps],text=True).strip())
    need(not events,"new registered disk error "+repr(events))
    status.update(utc=utc(),events=events)
    return status
