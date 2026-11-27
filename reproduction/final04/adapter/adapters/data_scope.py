"""Cold-only path isolation. Capture unique project-input reads without tracing numerical operations."""
import atexit,json,os,sys
from pathlib import Path
def install(target,base,out):
    root=os.path.normcase(str(target.parent.parent.resolve()))+os.sep
    allowed=tuple(os.path.normcase(str(x.resolve()))+os.sep for x in (target,base))
    opened={};blocked=[];seen={}
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        name=os.fsdecode(args[0])
        dest=seen.get(name)
        if dest is None:
            full=os.path.normcase(os.path.abspath(name))
            dest=full if full.startswith(root) else ''
            seen[name]=dest
        if not dest:return
        if not dest.startswith(allowed):
            blocked.append([name,dest]);raise PermissionError('cold run attempted to read or write historical project: '+name)
        mode=args[1]
        if mode is not None and ('r' in mode or '+' in mode):opened[dest]=opened.get(dest,0)+1
    sys.addaudithook(audit)
    def finish():
        with out.open('x',encoding='utf8')as f:json.dump({'unique_open_request_counts':opened,'blocked_old_project_access':blocked,'scope':'Python file opens inside the project. Native engine and extension origins have separate receipts.'},f,ensure_ascii=False)
        if blocked:sys.stderr.flush();os._exit(46)
    atexit.register(finish)
