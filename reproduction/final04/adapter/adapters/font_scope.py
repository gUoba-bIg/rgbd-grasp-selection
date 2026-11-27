"""Isolate font discovery using the locked wheel's supported MPL_IGNORE_SYSTEM_FONTS switch."""
import atexit
import hashlib
import json
import os
import sys
from pathlib import Path

FONT_SUFFIXES=('.ttf','.otf','.ttc','.afm','.pfb')
def install(target, role, config, receipt, arm='full', seed_cache=None):
    target=Path(target).resolve(); config=Path(config).resolve(); receipt=Path(receipt).resolve()
    config.mkdir(parents=True,exist_ok=False)
    os.environ.update(MPLCONFIGDIR=str(config),MPL_IGNORE_SYSTEM_FONTS='1',MPLBACKEND='Agg')
    envroot=target/'env'/role
    wheel=envroot/'Lib/site-packages/matplotlib/mpl-data'
    reads=set(); blocked=[]; registry_queries=[]
    if arm=='subset':
        seed=Path(seed_cache)
        raw=seed.read_bytes(); data=json.loads(raw)
        for field,keep in [('ttflist','DejaVuSans.ttf'),('afmlist','cmr10.afm')]:
            entries=[x for x in data[field] if Path(x['fname']).name==keep]
            if len(entries)!=1 or Path(entries[0]['fname']).is_absolute() or not (wheel/entries[0]['fname']).is_file():raise ValueError(('locked subset',field,entries))
            data[field]=entries
        with (config/seed.name).open('x',encoding='utf8')as f:json.dump(data,f,ensure_ascii=False)
    wheel_prefix=os.path.normcase(str(envroot))+os.sep
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            s=os.fsdecode(args[0])
            # Avoid path resolution on scientific file reads in the timed episode.
            low=s.lower()
            font=low.endswith(FONT_SUFFIXES)
            if 'kingsoft' in low or 'provenance_evidence_appendix' in low:
                blocked.append(s);raise PermissionError('unregistered host font or provenance archive')
            if font:
                full=os.path.normcase(os.path.abspath(s))
                if not full.startswith(wheel_prefix):
                    blocked.append(s);raise PermissionError('font outside isolated wheel installation')
                reads.add(full)
        elif event=='winreg.OpenKey' and len(args)>1 and isinstance(args[1],str) and 'windows nt\\currentversion\\fonts' in args[1].lower():
            registry_queries.append(args[1]);raise PermissionError('host font enumeration blocked')
    sys.addaudithook(audit)
    def finish():
        fm=sys.modules.get('matplotlib.font_manager')
        manager=getattr(fm,'fontManager',None)
        entries=[] if manager is None else [Path(q.fname).resolve() for q in manager.ttflist+manager.afmlist]
        ok=not blocked and not registry_queries and all(q.is_relative_to(wheel) and q.is_file() for q in entries)
        if arm=='subset':ok=ok and sorted(q.name for q in entries)==['DejaVuSans.ttf','cmr10.afm']
        if role=='rank':ok=ok and bool(entries)
        r=dict(role=role,arm=arm,config=str(config),ignore_system_fonts=os.environ.get('MPL_IGNORE_SYSTEM_FONTS'),fontmanager_loaded=manager is not None,fontmanager_path=getattr(fm,'__file__',None),resolved_font_entries=sorted(str(q)for q in entries),python_font_opens=sorted(reads),denied_attempts=blocked,host_font_registry_attempts=registry_queries,accepted=ok)
        if seed_cache:r['locked_cache_input_sha256']=hashlib.sha256(raw).hexdigest()
        with receipt.open('x',encoding='utf8')as f:json.dump(r,f,ensure_ascii=False,indent=2)
        if not ok:sys.stderr.write('Font isolation failed; inspect '+str(receipt)+'\n');sys.stderr.flush();os._exit(45)
    atexit.register(finish)
