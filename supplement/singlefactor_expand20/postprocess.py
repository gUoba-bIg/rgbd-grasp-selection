"""Wake on the final ledger's directory notification, then prepare final-outcome-only reports."""
from paths import *
import ctypes
from ctypes import wintypes
import subprocess
import time

def final_ledger():
    run=STUDY/"run";ledger=run/"new168_ledger.json"
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.FindFirstChangeNotificationW.argtypes=[wintypes.LPCWSTR,wintypes.BOOL,wintypes.DWORD]
    kernel.FindFirstChangeNotificationW.restype=wintypes.HANDLE
    kernel.FindNextChangeNotification.argtypes=[wintypes.HANDLE]
    kernel.FindCloseChangeNotification.argtypes=[wintypes.HANDLE]
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    handle=kernel.FindFirstChangeNotificationW(str(run),False,1|16)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    try:
        # Job files and checkpoints are in other directories. This waits on native events.
        while not ledger.exists():
            if kernel.WaitForSingleObject(handle,0xffffffff)!=0:raise ctypes.WinError(ctypes.get_last_error())
            if not kernel.FindNextChangeNotification(handle):raise ctypes.WinError(ctypes.get_last_error())
        for _ in range(30):
            try:return read(ledger)
            except (PermissionError,json.JSONDecodeError):time.sleep(.1)
        return read(ledger)
    finally:kernel.FindCloseChangeNotification(handle)

def call(name,args):
    logs=OUT/"admission/report_logs";logs.mkdir(exist_ok=True,parents=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONUTF8="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
    with (logs/(name+".out.log")).open("xb") as so,(logs/(name+".err.log")).open("xb") as se:
        proc=subprocess.run([str(x) for x in args],cwd=OUT,env=env,stdout=so,stderr=se,stdin=subprocess.DEVNULL)
    need(proc.returncode==0,name+" exited "+str(proc.returncode)+"; see "+str(logs))
    print("Returned",name,flush=True)

def main():
    ledger=final_ledger()
    if ledger["status"]!="complete":
        print("STOP",ledger["status"],ledger["first_shared_fault"] or ledger["disk_pause"],flush=True)
        return
    need(len(ledger["returned"])==168 and not ledger["first_shared_fault"],"unfinalized inputs")
    rank=C/"env/rank/Scripts/python.exe"
    src=OUT/"src"
    call("statistics",[rank,"-B","-s","-u","-X","utf8",src/"statistics_expand.py"])
    call("paper",[rank,"-B","-s","-u","-X","utf8",src/"paper.py"])
    node_dir=(src/"node_modules").resolve().parent
    node=node_dir/"bin/node.exe"
    # Set from the installed publication toolchain; executable and packages come from the same bundle.
    marker=Path(os.environ["V1SF_XLSX_MARKER"])
    call("workbook_marker",[node,marker,"--operation-kind","create","--expected-output-count","1","--output-format","xlsx"])
    call("workbook",[node,src/"paper_tables.mjs"])
    print("Tables and figures ready for visual verification.",flush=True)
if __name__=="__main__":main()
