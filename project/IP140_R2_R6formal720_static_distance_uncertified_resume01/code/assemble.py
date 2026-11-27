"""Separate the classification change from path, policy, sensor and execution sources."""
import difflib,sys
from common import *
OLD=ROOT/'IP140_R2_R6formal720_raw_a_c0_paired01'
def main():
    diffs=[];m={}
    def modify(folder,name,changes):
        original=OLD/folder/name;out=HERE/folder/name
        before=original.read_text(encoding='utf8');after=before
        for old,new in changes:
            require(after.count(old)==1,'unexpected context '+name+' '+old[:70])
            after=after.replace(old,new)
        if out.exists():require(sha(out)==sha(original),'pre-assembly local source differed')
        mode='w' if out.exists() else 'x'
        with out.open(mode,encoding='utf8',newline='\n') as f:f.write(after)
        diffs.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='sealed_R6/'+folder+'/'+name,tofile='continuation/'+folder+'/'+name))
        m[out.relative_to(HERE).as_posix()]={'original':sha(original),'continuation':sha(out)}
    modify('implementation','platform_trial.py',[
        ("bind_frozen_libraries()\nfrom platform_trace","bind_frozen_libraries()\nfrom result_classification import post_static,DETAIL\nfrom platform_trace"),
        ("sealed['version']!=VERSION.name","sealed['version']!='"+OLD.name+"'"),
        ("            if not sr['passed']:\n                code=candidate_static_failure(sr,directory=destination/'static_preflight',ctx=ctx)","            checked_report=post_static(destination,intent=intent,links=ctx.links)\n            write(destination/'static_classification_receipt.json',checked_report)\n            if not sr['passed']:\n                if checked_report['detail_class']==DETAIL:\n                    code='path_failure';result['detail_class']=DETAIL\n                else:\n                    code=candidate_static_failure(sr,directory=destination/'static_preflight',ctx=ctx)")])
    modify('code','method_entry.py',[
        ("from platform_trial import frozen_intent,worker,connection,checked_zero","from platform_trial import frozen_intent,worker,connection,checked_zero\n            from result_classification import native_rows"),
        ("                put(dst/'no_candidate_zero_receipt.json'","                empty_native=native_rows(dst/'no_candidate_zero_api',(ctx.robot,ctx.target,ctx.table,client),ctx.links)\n                put(dst/'no_candidate_integrity.json',empty_native)\n                put(dst/'no_candidate_zero_receipt.json'"),
        ("                if r['grasp_success']:","                sr=read(dst/'static_preflight/path_result.json');links={n:v['index'] for n,v in read(dst/'dynamic_setup/robot_link_readback.json').items()}\n                dyn_integrity=native_rows(dst/'dynamic_api',(sr['body_robot'],sr['body_object'],sr['body_table'],sr['client']),links,steps=r['physical_returns'])\n                put(dst/'dynamic_integrity.json',dyn_integrity)\n                if r['grasp_success']:")])
    modify('code','formal_dispatch.py',[
        ("from common import *","from common import *\nOLD=ROOT/'"+OLD.name+"'"),
        ("self.files.update({HERE/k:h for k,h in local.items()})","self.files.update({OLD/k:h for k,h in local.items()})\n        signed=read(HERE/'protocol/release.json')\n        require(signed['replay_passed'] and signed['total']==720 and signed['new_allowed']==712 and signed['old_adjudicated']==7,'incomplete release')\n        self.files.update({HERE/k:h for k,h in signed['files'].items()})\n        self.files[HERE/'protocol/release.json']=sha(HERE/'protocol/release.json')\n        require(sha(OLD/'sealed_intents.json')==sha(HERE/'sealed_intents.json')==signed['parent_intents_sha256'],'original intent seal changed')"),
        ("guard=Guard();d=read(HERE/'preregistration/design.json');q=deque(d['trials']);live={};done=[];faults=[];trail=None","guard=Guard();d=read(HERE/'preregistration/design.json');scope=read(HERE/'protocol/continuation.json');prior=set(scope['adjudicated_unchanged']+scope['supplemental_only']);q=deque(t for t in d['trials'] if t['trial_id'] not in prior);live={};done=[];faults=[];trail=None\n    require([t['trial_id'] for t in q]==scope['allowed_once_in_original_order'] and len(q)==712,'pending identity/order drift')\n    for key in sorted(prior):\n        pr=HERE/'run/jobs/method'/key/'return.json';result=HERE/'trials'/key/'method_result.json';require(read(pr)['method_result_sha256']==sha(result),'previous formal result identity')\n        done.append(dict(trial_id=key,method_result_sha256=sha(result),return_sha256=sha(pr),origin='offline_supplement' if key in scope['supplemental_only'] else 'sealed_prior_result'))"),
        ("require(len(unit_lane)==240,'lane design')","require(len(unit_lane)==240 and unit_lane==read(OLD/'run/method_initial_status.json')['fixed_unit_lane'],'original lane design')"),
        ("pending={lane:deque(t for t in d['trials'] if unit_lane[t['unit_id']]==lane) for lane in lanes}","pending={lane:deque(t for t in q if unit_lane[t['unit_id']]==lane) for lane in lanes}"),
        ("require(all(len(rows)==180 for rows in pending.values()),'balanced independent lanes')","require(sum(map(len,pending.values()))==712,'independent lane membership')")])
    modify('code','analyze.py',[
        ("def category(r,static):","def category(r,static):\n    if r.get('detail_class')=='static_distance_uncertified':return r['detail_class']"),
        ("        ledger.append(row);","        scope=read(HERE/'protocol/continuation.json')\n        row['result_origin']='original_unchanged' if k in scope['adjudicated_unchanged'] else 'offline_supplement_no_execution' if k in scope['supplemental_only'] else 'continuation'\n        ledger.append(row);")])
    modify('code','delivery.py',[
        ("if k[0] in ('preregistration','code','implementation','analysis','checkpoints','audit'):return True","if k[0] in ('preregistration','protocol','history','code','implementation','analysis','checkpoints','audit'):return True"),
        ("'no_candidate_zero_receipt.json','path_result.json','summary.json'","'no_candidate_zero_receipt.json','path_result.json','summary.json','static_classification_receipt.json','dynamic_integrity.json','no_candidate_integrity.json'"),
        ("'## 协议与输入','',","'## 协议与输入','','续接版保留原 R6 的七份已判定回执。R6FORMAL-001 根据零物理步原始证据补充裁定为 static_distance_uncertified，工程门和抓取成功均为 false。原版停批记录保持封存。剩余 712 份意图逐字节继承；每种方法的成功分母仍为最初分派的 240 次。','',"),
        ("zipname='R6formal720_轻量核心审查包.zip'","zipname='R6formal720_续接轻量核心审查包.zip'")])
    originals=['scene_world.py','platform_static.py','platform_trace.py','platform_imports.py']
    for name in originals:require(sha(OLD/'implementation'/name)==sha(HERE/'implementation'/name),'frozen geometry or instrumentation changed')
    with (HERE/'audit/continuation.diff').open('x',encoding='utf8',newline='\n') as f:f.writelines(diffs)
    put(HERE/'audit/adapter_hashes.json',m)
if __name__=='__main__':main()
