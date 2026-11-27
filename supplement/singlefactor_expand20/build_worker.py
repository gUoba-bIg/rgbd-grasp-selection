"""Generate a traceable minimal adaptation of the verified V1 supplemental method wrapper."""
from paths import *
import difflib

def build():
    orig=PREVIOUS_CODE/"method_trial.py"
    need(sha(orig)=="cc1865fd7916769550201d8ee36d01930e02ccb4a18a89102e4ae89290c6121b","unknown method wrapper")
    source=orig.read_text("utf8")
    text=source
    def change(a,b):
        nonlocal text
        need(text.count(a)==1,"wrapper patch hunk not unique")
        text=text.replace(a,b)
    change("from common_v1 import *\nfrom runtime_stages import STUDY,native_setup",
           "from paths import *\nfrom runtime_stages import native_setup\nfrom reported_gap import classify_if_known")
    change('need(seal["count"]==60 if cohort=="diagnostic" else 0<=seal["count"]<=72,"budget size")',
           'need(cohort=="robustness" and it["supplement_kind"]=="robustness" and tid in seal["order"] and seal["count"]==168,"expansion authorization")')
    # The earlier supplement also had a diagnostic certificate exception. Exclude this whole path here.
    a=text.index('    if cohort=="diagnostic":\n')
    b=text.index("    return ticket,it,sel,record",a)
    text=text[:a]+text[b:]
    a=text.index('def diagnostic_post(');b=text.index('def run(tid,cohort):',a)
    text=text[:a]+text[b:]
    change('    from diagnostic_gate import install\n','')
    change('    waive=it["supplement_kind"]=="diagnostic_uncertified"\n    counts=install(platform_static) if waive else {}',
           '    waive=False\n    counts={}\n    need(Path(sys.modules["pybullet"].__file__).resolve()==(C/"p/runtime/simulation_v52e/venv/Lib/site-packages/pybullet.cp312-win_amd64.pyd").resolve() and sha(Path(sys.modules["pybullet"].__file__))==ENGINE,"unregistered runtime engine")')
    change('report=diagnostic_post(dest,it,ctx,sr) if waive else post_static(dest,intent=it,links=ctx.links)',
           'report=post_static(dest,intent=it,links=ctx.links)')
    change('                code=dynamic_category(returned,directory=episode,ctx=ctx)',
           '                code=dynamic_category(returned,directory=episode,ctx=ctx)\n                if code is None:\n                    code=classify_if_known(returned,episode,ctx,tid)')
    change('        disk()\n        write("study/trials/', '        write("study/trials/')
    exclusive_bytes(OUT/"src/method_trial_expand.py",text.encode())
    diff="".join(difflib.unified_diff(source.splitlines(True),text.splitlines(True),fromfile="previous_supplement/method_trial.py",tofile="src/method_trial_expand.py"))
    exclusive_bytes(OUT/"admission/method_wrapper.diff",diff.encode())
if __name__=="__main__":build()
