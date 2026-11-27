"""v39 evaluation-only completion of frozen v38 final800 models.

No training command exists. v38 source, outcomes, caches, monitoring and weights
remain read-only. A new authorization permits the full160 assessment despite the
preserved historical20-frame validity stop. Scientific thresholds are unchanged.
"""
import sys
import os
import time
import uuid
import shutil
import argparse
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import restore_p5_v38 as parent
OLD_BASE=ROOT/'results/P5/fusion_v38'
OLD_LOG=ROOT/'logs/P5/fusion_v38'
OLD_RUN=ROOT/'experiments/p5_fusion_v38'
BASE=ROOT/'results/P5/completion_v39'
OUT=BASE/'full'
LOG=ROOT/'logs/P5/completion_v39'
PROPOSAL=LOG/'proposal.json'
AUTH=LOG/'authorization.json'
read,sha,stat,need,atomic,once=parent.read,parent.sha,parent.stat,parent.need,parent.atomic,parent.once
# Namespace adaptation is process-local. The original proposal and RUN stay old
# so the inherited parent export_binding retains the exact v38 model identity.
parent.BASE,parent.LOG,parent.RUN=BASE,LOG,OLD_RUN
parent.set_scope('full')
accounting=parent.legacy.accounting
append=parent.legacy.append
_original_decide=parent.decide


def old_monitor_values():
    import numpy as np
    values=[float(np.mean([read(OLD_BASE/'monitor'/f'{arm}_{step:04d}.json')[condition]['scene_macro_f1']
        for arm in ('fixed','quality') for condition in ('clean','joint')])) for step in (640,720,800)]
    return values,max(values)-min(values)


parent.stable_values=old_monitor_values


def verify_old_screen():
    seal=read(OLD_BASE/'screen/exports_complete.json')
    need(seal['source_binding']==parent.export_binding('screen'),'Original screen/final800 identity differs')
    need(seal['method_frames']==280 and seal['parent_cached_frames']==20,'Original screen scope differs')
    current={p.relative_to(ROOT).as_posix():stat(p) for p in (OLD_BASE/'screen/exports').rglob('*') if p.is_file()}
    need(current==seal['inventory'],'Original screen cache inventory/bytes changed')
    decision=read(OLD_BASE/'screen/decision.json')
    need(decision['eligible_full_evaluation'] is False,'Historical v38 validity stop must remain false')
    need(read(OLD_LOG/'cold.json')['passed'] is True,'v38 cold acceptance missing')
    return seal


def new_binding():
    return dict(v39_proposal_sha256=sha(PROPOSAL),full_parent_identity=parent.export_binding('full'),
        v38_screen_seal=stat(OLD_BASE/'screen/exports_complete.json'),
        v38_screen_decision=stat(OLD_BASE/'screen/decision.json'),
        original_screen_eligible_full_evaluation=False,
        authorization='New v39 evaluation-only full160 assessment; no modification of the v38 stop or its policy')


def verify_export_seal(scope):
    if scope=='screen':return verify_old_screen()
    need(scope=='full','Only full160 is a new v39 scope')
    seal=read(OUT/'exports_complete.json')
    need(seal['source_binding']==new_binding(),'v39 full source binding differs')
    need(seal['method_frames']==2240 and seal['parent_cached_frames']==160,'v39 full frame scope differs')
    current={p.relative_to(ROOT).as_posix():stat(p) for p in (OUT/'exports').rglob('*') if p.is_file()}
    need(current==seal['inventory'],'v39 export inventory/bytes differ')
    return seal


parent.verify_export_seal=verify_export_seal


def decide(metrics,scope):
    need(scope=='full','v39 does not reinterpret the old screen')
    result=_original_decide(metrics,scope)
    result['full_validity_passed']=result.pop('eligible_full_evaluation')
    result.update(version='v39',original_screen_eligible_full_evaluation=False,
        original_screen_decision_sha256=sha(OLD_BASE/'screen/decision.json'),
        full_evaluation_basis='New authorized evaluation-only protocol; previous screen failure remains unchanged',
        new_optimizer_steps=0,model_checkpoint_source='Frozen v38 final800 for each arm',
        interpretation='Full160 exposed-development result with original scientific thresholds; not independent testing or evidence that v38 screen passed')
    return result


parent.decide=decide


def proposal():
    old_usage=sum(read(p)['seconds'] for pattern in ('usage_*.json','preflight_usage_*.json') for p in OLD_LOG.glob(pattern))
    final_review=read(OLD_LOG/'root_final_review.json')
    return dict(version='v39',operation='Evaluate unchanged v38 final models on the originally planned full160 scope',
        training_allowed=False,new_optimizer_intents=0,
        source_models={arm:read(OLD_RUN/arm/'final.json') for arm in ('fixed','quality')},
        original_screen_eligible_full_evaluation=False,original_screen_decision=stat(OLD_BASE/'screen/decision.json'),
        authorization_reason='User requests reasoned feasibility assessment; complete missing larger-scope evidence before proposing more training',
        scope=dict(development_scenes=10,frames_per_scene=16,conditions=list(parent.CONDITIONS),
            method_frames=2240,reused_screen_method_frames=280,new_method_frames=1960,
            reused_monitor_frames=320,reused_RGB_parent_frames=160,official_new_AP_blocks=140,
            RGB_parent_AP='Ten original P4 scene blocks reused and independently reconstructed'),
        resources=dict(max_forward_intents=3920,max_optimizer_intents=0,max_official_AP_intents=140,
            wall_seconds=3600,prepare_compute_included=True,gpu_bytes=14*1024**3,rss_bytes=6*1024**3,
            output_bytes=20*1024**3,disk_reserve_bytes=60*1024**3),
        original_scientific_rule=read(OLD_LOG/'proposal.json')['final_rule'],
        historical_accounting=read(OLD_LOG/'cold.json')['accounting'],
        historical_wall=dict(component_usage_seconds=old_usage,root_recorded_compute_seconds=final_review['recorded_compute_seconds'],
            note='Historical records are kept separately; new authorization has its own3600s cap, not a reset or extension of old budget'),
        inherited_sources=['scripts/restore_p5_v38.py','scripts/restore_quality_v37.py','src/modal_fusion_v38.py']+read(OLD_LOG/'proposal.json')['inherited_sources'],
        parent_reference_files=parent.parent_reference_files()+[p.relative_to(ROOT).as_posix() for p in
            [OLD_LOG/'proposal.json',OLD_LOG/'review.json',OLD_LOG/'cold.json',OLD_LOG/'root_final_review.json',
             OLD_BASE/'screen/exports_complete.json',OLD_BASE/'screen/decision.json']]+
            [p.relative_to(ROOT).as_posix() for p in (OLD_BASE/'monitor').glob('*.json')],
        finish='Export, full AP, independent review, real cold entry, evidence-based disposition; then stop. No P6/P10/v40.')


def binding():
    auth=read(AUTH)
    need(auth.get('frozen') is True and auth['proposal_sha256']==sha(PROPOSAL),'Root v39 freeze authorization missing')
    need(auth.get('new_full_evaluation_explicitly_authorized') is True,'New full evaluation authorization flag missing')
    need(auth.get('parent_screen_eligible_unchanged') is False,'Historical screen stop flag must remain false')
    need(auth.get('protected'),'Frozen references missing')
    for path,value in auth['protected'].items():need(stat(ROOT/path)==value,'Protected input/source changed: '+path)
    return auth


class Session(parent.legacy.Session):
    def guard(self):
        self.guard_save_only();self.save()
        need(self.previous+time.monotonic()-self.start<3600,'v39 cumulative3600s cap exhausted')
        need(self.max_rss<6*1024**3 and self.max_gpu<14*1024**3,'v39 memory cap')
        need(shutil.disk_usage(ROOT).free>=60*1024**3,'v39 disk reserve')
        need(not (LOG/'STOP').exists(),'Explicit v39 stop marker')
        used=sum(p.stat().st_size for directory in (LOG,BASE) if directory.exists() for p in directory.rglob('*') if p.is_file())
        need(used<20*1024**3,'v39 output cap')

    def intent(self,family,**values):
        self.guard()
        limit=0 if self.preflight else dict(forward=3920,optimizer=0,official_AP=140)[family]
        need(self.counts[family]<limit,family+' v39 budget exhausted or forbidden')
        self.counts[family]+=1;cid=uuid.uuid4().hex
        append(family+'_intent',self.preflight,call_id=cid,**self.ctx,**values)
        return cid


def prepare():
    if (LOG/'prepare.json').exists():
        need(read(LOG/'prepare.json')['proposal_sha256']==sha(PROPOSAL),'Preparation proposal differs')
        verify_old_screen();return
    s=Session('evaluation_only_preparation',True)
    try:
        verify_old_screen()
        once(PROPOSAL,proposal())
        once(LOG/'prepare.json',dict(passed=True,version='v39',proposal_sha256=sha(PROPOSAL),
            new_full_evaluation_authorization=True,authorization_scope='Full160 using frozen v38 weights, new training forbidden',
            original_screen_eligible_full_evaluation=False,old_screen_decision_sha256=sha(OLD_BASE/'screen/decision.json'),
            new_forward=0,new_optimizer=0,new_official_AP=0,model_identity_checked_without_model_execution=True))
        s.finish()
    except BaseException as error:s.finish('failed',repr(error));raise


def copy_committed(source,destination):
    source,destination=Path(source),Path(destination)
    row=read(source);need(sha(source.with_suffix('.npz'))==row['cache_sha256'],'Reuse source changed')
    target=destination.with_suffix('.npz');destination.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():need(sha(target)==row['cache_sha256'],'Interrupted reused copy differs; preserve it')
    else:shutil.copyfile(source.with_suffix('.npz'),target)
    once(destination,row)


def export(s):
    import torch
    verify_old_screen();records=parent._OLD['development_records']
    for arm in ('fixed','quality'):
        identity=read(OLD_RUN/arm/'final.json')
        need(identity['logical_updates']==800 and sha(OLD_RUN/arm/'final.pt')==identity['sha256'],'v38 final800 differs')
        model=parent.make_model(arm,s)
        model.load_state_dict(torch.load(OLD_RUN/arm/'final.pt',map_location='cpu',weights_only=False)['model'])
        for item in [x for x in parent.legacy.task_sets() if x['arm']==arm]:
            model.ablation=item['ablation']
            for rec in records:
                name=f"scene_{rec['scene']:04d}_{rec['frame']}.json";target=OUT/'exports'/item['name']/name
                if target.exists():need(sha(target.with_suffix('.npz'))==read(target)['cache_sha256'],'v39 cache changed');continue
                old=OLD_BASE/'screen/exports'/item['name']/name
                if old.exists():copy_committed(old,target)
                else:
                    s.ctx=dict(phase='full_export',arm=arm,set=item['name'])
                    arrays,row=parent.predict(s,model,rec,item['condition']);parent.save_case(target,arrays,dict(**row,set=item))
        parent.immutable_parent_states(model)
        del model;torch.cuda.empty_cache()
    item=parent.tasks()[-1]
    for rec in records:
        name=f"scene_{rec['scene']:04d}_{rec['frame']}.json";target=OUT/'exports'/item['name']/name
        if target.exists():need(sha(target.with_suffix('.npz'))==read(target)['cache_sha256'],'RGB cache changed');continue
        old=OLD_BASE/'screen/exports'/item['name']/name
        if old.exists():copy_committed(old,target)
        else:parent.parent_cache('full',rec,item)
    inventory={p.relative_to(ROOT).as_posix():stat(p) for p in (OUT/'exports').rglob('*') if p.is_file()}
    once(OUT/'exports_complete.json',dict(scope='full',method_frames=2240,parent_cached_frames=160,
        new_method_frames=1960,reused_screen_method_frames=280,inventory=inventory,source_binding=new_binding()))


def own_snapshot():
    return {p.relative_to(ROOT).as_posix():stat(p) for p in BASE.rglob('*') if p.is_file() and p != BASE/'closeout.json'}


def old_evidence_snapshot():
    # The accepted v38 output seal already includes weights, all monitors, and
    # old screen results. Its exact identities are rechecked without rewriting.
    receipt=read(OLD_LOG/'review.json');expected=dict(receipt['reviewed_outputs'])
    for extra in [OLD_BASE/'closeout.json',OLD_LOG/'review.json',OLD_LOG/'cold.json',OLD_LOG/'root_final_review.json',OLD_LOG/'proposal.json']:
        expected[extra.relative_to(ROOT).as_posix()]=stat(extra)
    for path,value in expected.items():need(stat(ROOT/path)==value,'Accepted v38 output changed: '+path)
    return expected



def audit_call_scope():
    from collections import Counter
    rows=parent.legacy.events();counts=accounting()
    need(counts['optimizer']['intents']==0,'v39 optimization is forbidden')
    old_keys={(r['scene'],r['frame']) for r in parent._OLD['monitor_records']}
    new_records=[r for r in parent._OLD['development_records'] if (r['scene'],r['frame']) not in old_keys]
    expected={(item['name'],r['scene'],r['frame'],network) for item in parent.legacy.task_sets() for r in new_records for network in ('fusion_wrapper','quality_internal')}
    intents=[r for r in rows if r['kind']=='forward_intent']
    actual=[(r['set'],r['scene'],r['frame'],r['network']) for r in intents]
    need(len(actual)==3920 and len(set(actual))==3920 and set(actual)==expected,'New forward scope differs; no duplicate or inherited predictions may be recounted')
    roles=Counter(r['network'] for r in intents)
    need(dict(roles)=={'fusion_wrapper':1960,'quality_internal':1960},'Network-entry counts differ')
    ap=[r for r in rows if r['kind']=='official_AP_intent']
    expected_ap={(item['name'],scene) for item in parent.legacy.task_sets() for scene in sorted({r['scene'] for r in new_records})}
    actual_ap=[(r['set'],r['scene']) for r in ap]
    need(len(actual_ap)==140 and len(set(actual_ap))==140 and set(actual_ap)==expected_ap,'New official AP scope differs')
    intended={r['call_id']:r for r in rows if r['kind'] in ('forward_intent','official_AP_intent')}
    for r in rows:
        if r['kind']=='forward_returned':
            source=intended[r['call_id']]
            need(all(r.get(k)==source.get(k) for k in ('network','set','scene','frame')),'Known model-return identity differs')
        elif r['kind']=='official_AP_returned':
            source=intended[r['call_id']]
            need(all(r.get(k)==source.get(k) for k in ('set','scene')),'Known AP-return identity differs')
            expected_path=OUT/'AP'/r['set']/f"scene_{r['scene']:04d}.json"
            need(r['path']==expected_path.relative_to(ROOT).as_posix() and sha(expected_path)==r['sha256'],'Known AP-return artifact differs')
    return dict(forward_intents_by_network=dict(roles),forward_known_returns_by_network=dict(Counter(r['network'] for r in rows if r['kind']=='forward_returned')),
        expected_new_cases=1960,official_AP_intents=140,optimizer_intents=0,unknown_calls_preserved=counts)


def review(s):
    before=own_snapshot();old_before=old_evidence_snapshot();counts=accounting();call_scope=audit_call_scope()
    # The inherited monitor reviewer is pure CPU and is explicitly pointed at
    # v38 only during this read-only operation; its Session still writes v39 LOG.
    saved=parent.BASE
    try:
        parent.BASE=OLD_BASE
        monitor_count=parent.review_monitors(s)
    finally:parent.BASE=saved
    result=parent.evaluate(s,'full',True)
    need(own_snapshot()==before,'Read-only v39 review changed paths or bytes')
    for path,value in old_before.items():need(stat(ROOT/path)==value,'Review changed v38 evidence')
    need(accounting()==counts,'Independent review made new research calls')
    need(counts['optimizer']['intents']==0,'v39 optimization is forbidden')
    once(LOG/'review.json',dict(passed=True,engineering_passed=True,scientific_usable=result['scientific_usable'],
        scientific_adoption_supported=result['scientific_adoption_supported'],
        full_original_P5_scientific_passed=result['full_original_P5_scientific_passed'],
        original_screen_eligible_full_evaluation=False,completed_scopes=['new_full160'],
        verified_monitor_frames=monitor_count,verified_method_frames=2240,verified_RGB_frames=160,
        new_forward=0,new_optimizer=0,new_official_AP=0,accounting=counts,call_scope=call_scope,
        reviewed_outputs=before,unchanged_v38_outputs=old_before))


def cold():
    result=read(LOG/'review.json');need(result['passed'],'v39 independent review missing')
    before=accounting();verify_export_seal('full');verify_old_screen()
    need(audit_call_scope()==result['call_scope'],'Cold call-scope audit differs')
    need(before['optimizer']['intents']==0 and before['optimizer']['known_returns']==0,'v39 must never train')
    need(own_snapshot()==result['reviewed_outputs'],'v39 completed artifacts changed')
    for path,value in result['unchanged_v38_outputs'].items():need(stat(ROOT/path)==value,'v38 historical evidence changed')
    need(accounting()==before,'Cold entry made research calls')
    once(LOG/'cold.json',dict(passed=True,engineering_passed=True,scientific_usable=result['scientific_usable'],
        scientific_adoption_supported=result['scientific_adoption_supported'],
        full_original_P5_scientific_passed=result['full_original_P5_scientific_passed'],
        original_screen_eligible_full_evaluation=False,new_forward=0,new_optimizer=0,new_official_AP=0,
        accounting=before,accounting_fully_observed=all(not x['unknown_ids'] for x in before.values()),
        unknown_calls_remain_unknown=True,end_of_authorized_P5_completion=True,next_research_started=False))
    parent.legacy.status('completed_stopped',engineering_passed=True,scientific_usable=result['scientific_usable'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','export','evaluate','review','cold'));args=parser.parse_args()
    from scripts.pipeline import project_lock
    with project_lock(ROOT):
        if args.mode=='prepare':prepare();return
        binding();need(read(LOG/'prepare.json')['passed'],'v39 preparation missing')
        if args.mode=='cold':cold();return
        s=Session(args.mode)
        try:
            if args.mode=='export':export(s)
            elif args.mode=='evaluate':parent.evaluate(s,'full',False)
            else:review(s)
            binding();s.finish()
        except BaseException as error:s.finish('failed',repr(error));parent.legacy.status('failed',mode=args.mode,error=repr(error),accounting=accounting());raise


if __name__=='__main__':main()
