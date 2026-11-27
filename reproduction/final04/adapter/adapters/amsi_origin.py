"""Identity of the single permitted *host* component. No DLL is redistributed and no registry writes occur."""
import os,sys,ctypes,hashlib,json,subprocess,winreg
from ctypes import wintypes
from pathlib import Path
PROVIDER='{2781761E-28E0-4109-99FE-B9D127C57AFE}'
def public_programdata():
    # FOLDERID_ProgramData is machine-wide and independent of the worker's private environment.
    import uuid
    folder_id=(ctypes.c_ubyte*16).from_buffer_copy(uuid.UUID('62AB5D82-FDC1-4DC3-A9DD-070D1D495D97').bytes_le)
    shell=ctypes.WinDLL('shell32');ole=ctypes.WinDLL('ole32')
    shell.SHGetKnownFolderPath.argtypes=[ctypes.c_void_p,wintypes.DWORD,wintypes.HANDLE,ctypes.POINTER(ctypes.c_wchar_p)]
    shell.SHGetKnownFolderPath.restype=ctypes.c_long;ole.CoTaskMemFree.argtypes=[ctypes.c_void_p]
    result=ctypes.c_wchar_p()
    hr=shell.SHGetKnownFolderPath(ctypes.byref(folder_id),0,None,ctypes.byref(result))
    if hr!=0 or not result.value:raise RuntimeError('Windows ProgramData known folder not resolved')
    try:return Path(result.value).resolve(strict=True)
    finally:ole.CoTaskMemFree(ctypes.cast(result,ctypes.c_void_p))
def read64():
    mask=winreg.KEY_READ|winreg.KEY_WOW64_64KEY
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'SOFTWARE\Microsoft\AMSI\Providers',0,mask) as key:
        names=sorted(winreg.EnumKey(key,i) for i in range(winreg.QueryInfoKey(key)[0]))
    if PROVIDER.casefold() not in [n.casefold() for n in names]:raise RuntimeError('unregistered AMSI provider')
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,'SOFTWARE\\Classes\\CLSID\\'+PROVIDER+'\\InprocServer32',0,mask) as key:
        raw,kind=winreg.QueryValueEx(key,None)
    if kind not in (winreg.REG_SZ,winreg.REG_EXPAND_SZ) or not isinstance(raw,str):raise RuntimeError('invalid server registry type')
    text=raw.strip().strip('"')
    prefix='%programdata%'
    if text[:len(prefix)].casefold()==prefix and text[len(prefix):len(prefix)+1]==os.sep:
        path=public_programdata()/text[len(prefix)+1:]
    elif '%' not in text:path=Path(text)
    else:raise RuntimeError('unregistered expandable provider location')
    if not path.is_absolute() or path.name.casefold()!='mpoav.dll' or not path.is_file():raise RuntimeError('invalid 64-bit AMSI server path')
    return dict(view=64,provider=PROVIDER,providers=names,value=raw,registry_type=int(kind),canonical_path=str(path.resolve(strict=True)))
def os_signature(path):
    system=Path(os.environ['SYSTEMROOT']);ps=system/'System32/WindowsPowerShell/v1.0/powershell.exe'
    e=os.environ.copy();e['PROGRAMDATA']=str(public_programdata());e['PATH']=str(system/'System32')+';'+str(system); e.pop('PSMODULEPATH',None)
    cmd=[str(ps),'-NoProfile','-NonInteractive','-File',str(Path(__file__).with_name('amsi_identity.ps1')),'-Component',str(path)]
    return json.loads(subprocess.check_output(cmd,env=e,stderr=subprocess.PIPE,timeout=60))
def match(obs,loaded,old=None):
    r=obs['registry'];s=obs['signature'];actual=Path(loaded)
    if not actual.is_absolute() or not(actual.resolve(strict=True)==Path(r['canonical_path']).resolve(strict=True)==Path(s['actual']).resolve(strict=True)):return False
    if r['view']!=64 or r['provider']!=PROVIDER or s['signature_status']!='Valid' or s['signer_simple']!='Microsoft Windows':return False
    if not s['signer_subject'].startswith('CN=Microsoft Windows,') or not s['version'] or s['bytes']<=0 or len(s['sha256'])!=64:return False
    return old is None or obs==old
def observe():
    r=read64();p=Path(r['canonical_path']);s=os_signature(p)
    with p.open('rb') as f:dig=hashlib.file_digest(f,'sha256').hexdigest()
    obj=dict(registry=r,signature=s)
    if dig!=s['sha256'] or not match(obj,p):raise RuntimeError('host provider integrity/signature mismatch')
    return obj
def get_locked(t):
    old=json.loads((t/'setup_logs/registered_host_provider.json').read_bytes());obs=observe()
    if obs!=old:raise RuntimeError('host provider changed during this registration, no refresh')
    return obs
def loaded_native():
    k=ctypes.WinDLL('kernel32',use_last_error=True);ps=ctypes.WinDLL('psapi',use_last_error=True)
    k.GetCurrentProcess.restype=wintypes.HANDLE;proc=k.GetCurrentProcess()
    ps.EnumProcessModules.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.HMODULE),wintypes.DWORD,ctypes.POINTER(wintypes.DWORD)]
    ps.GetModuleFileNameExW.argtypes=[wintypes.HANDLE,wintypes.HMODULE,wintypes.LPWSTR,wintypes.DWORD]
    mods=(wintypes.HMODULE*8192)();n=wintypes.DWORD()
    if not ps.EnumProcessModules(proc,mods,ctypes.sizeof(mods),ctypes.byref(n)) or n.value>ctypes.sizeof(mods):raise ctypes.WinError()
    paths=[]
    for h in mods[:n.value//ctypes.sizeof(wintypes.HMODULE)]:
        b=ctypes.create_unicode_buffer(32768);count=ps.GetModuleFileNameExW(proc,h,b,len(b))
        if not 0<count<len(b)-1:raise RuntimeError('incomplete loaded module path')
        paths.append(Path(b.value).resolve(strict=True))
    return paths
def inventory(t,base,role):
    import gzip
    provider=get_locked(t);outside=[];all_native={};matches=[]
    with gzip.open(base/'locks/signed_display_driver.jsonl.gz','rt',encoding='utf8') as f:
        driver={}
        for line in f:
            key,_,h=json.loads(line);driver.setdefault(Path(key).name.casefold(),set()).add(h)
    for p in loaded_native():
        if p.is_relative_to(t):origin='cold_installation'
        elif p.is_relative_to(Path(os.environ['SYSTEMROOT']).resolve()):origin='windows_system'
        elif match(provider,p):matches.append(str(p));origin='registered_AMSI_64'
        else:
            with p.open('rb') as f:dig=hashlib.file_digest(f,'sha256').hexdigest()
            if dig in driver.get(p.name.casefold(),set()):origin='sealed_driver'
            else:outside.append(str(p));origin='unregistered'
        all_native[str(p)]=origin
    python={}
    for name,module in list(sys.modules.items()):
        value=getattr(module,'__file__',None)
        if not value or not Path(value).is_file():continue
        p=Path(value).resolve();python[name]=str(p)
        if not p.is_relative_to(t) and not p.is_relative_to(base):outside.append(str(p))
    return dict(role=role,provider=provider,matched_loaded_paths=matches,native_modules=all_native,python_modules=python,outside=outside)
