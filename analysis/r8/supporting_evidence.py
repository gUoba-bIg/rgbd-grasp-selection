"""Link derived claims to exact frozen file identities and their source locations."""
import csv, json, hashlib
from pathlib import Path
V=Path(__file__).resolve().parents[1];root=V.parent
R6='IP140_R2_R6formal720_static_distance_uncertified_resume01'
R7='IP140_R2_R7ablation240_mechanism01'
R4='IP140_R2_R4dev72_cross_dim_explicit_resume01'
def h(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def wr(p,j):
    p.write_text(json.dumps(j,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def extract(rel,markers):
    p=root/rel; s=p.read_text('utf-8-sig').splitlines()
    hits={}
    for m in markers:
        found=[k for k,t in enumerate(s,1) if m in t]
        assert found,(rel,m)
        hits[m]=found
    return dict(path=rel,sha256=h(p),lines=hits)
sources={
 'S01':('src/p56_geometry_v51.py',['anchors_per_side','fixed_u_scale_m','def association_by_warp','if target and','def aggregate_contact','collision_proxy=float']),
 'S02':('src/uncertainty_mixture_v43.py',['CANDIDATES = np.array']),
 'S03':(R4+'/implementation/policy_network.py',['from graspnet import','seg=ModalFusionV38','sample','set(o.files)','num_angle=12','offsets','seed']),
 'S04':(R4+'/implementation/policy_rank.py',['def g1_geometry','d=float(c[3])','def decoder_domain','GraspGroup(cand[viable].copy()).nms','top = kept[:20]','seed_predicted_local_instance']),
 'S05':(R4+'/implementation/platform_select.py',['41','never_fallback_after_postlock_safety']),
 'S06':('src/p5_quality_v23.py',['condition==','shift=','rng.normal','v &=','hh,ww']),
 'S07':(R4+'/implementation/platform_evaluator.py',['PHASE_STEPS','aggregate_table_support','slipped','hold']),
 'S08':(R6+'/implementation/result_classification.py',['static_distance_uncertified']),
 'S09':(R6+'/preregistration/design.json',['prior_protocol_deviation','bootstrap_reps','balanced_order']),
 'S10':(R7+'/code/freeze_sample.py',['unit_hash','method_results_accessed_by_sampler','len(keep)==80']),
 'S11':(R7+'/code/score_units.py',['np.ones_like(a)','a,\\x27fixed\\x27,zero','a,\\x27none\\x27,prior','value=float(row[0])']),
 'S12':(R7+'/preregistration/full_C0_reference_index.json',['source_hashes']),
 'S13':('docs/P5完整评价与原方案取舍结果_v39.txt',['门槛']),
 'S14':('docs/P6空间表面概率模型结果_v50.txt',['warp_calibration','warp_multilevel_reliability']),
 'S15':(R6+'/protocol/release.json',['prior_budget_reused','offline_supplements']),
 'S16':(R6+'/analysis/paired_statistics.json',['unknown_or_shared_failure']),
 'S17':(R7+'/analysis/failure_routes.json',['all_four_unique']),
 'S18':(R6+'/implementation/platform_static.py',['static','candidate_id']),
 'S19':(R4+'/implementation/platform_support.py',['aggregate_table_support']),
 'S20':(R4+'/implementation/scene_camera.py',['projection']),
 'S21':(R4+'/implementation/platform_plan.py',['candidate']),
 'S22':(R6+'/analysis/formal_ledger.json',['detail_class']),
 'S23':(R7+'/analysis/ablation_ledger.json',['detail_class']),
 'S24':(R6+'/review_report.md',['R5 observed_contactability','入']),
 'S25':(R7+'/preregistration/design.json',['explicit_80_unit_roster_found','formulas']),
 'S26':(R6+'/preregistration/executing_source.json',[]),
 'S27':('runtime/build/graspnet/utils/loss_utils.py',['torch.cross(axis_x, axis_y, dim=-1)']),
 'S28':(R6+'/preregistration/frozen_inputs.json',[]),
 'S29':(R7+'/preregistration/input_release.json',[]),
 'S30':(R4+'/implementation/platform_trial.py',['static','candidate']),
 'S32':('IP140_R2_smoke22_miss_negative_controls_full_execution01/model/manifest.json',['F_T_G']),
 'S33':('IP140_R2_smoke22_miss_negative_controls_full_execution01/model/ur5e_ip140_r1.urdf',['ip140_left_pad','ip140_right_pad','prismatic']),
 'S31':(R6+'/implementation/platform_trace.py',['physics']),
}
registry={k:extract(rel,[m.replace("\\x27","'") for m in markers]) for k,(rel,markers) in sources.items()}
wr(V/'evidence/method_sources.json',registry)
refs=json.loads((V/'source/references.json').read_text('utf8'))
(V/'supplement/文献核验.md').write_text('# 引用来源核验\n\n正文引文按下列顺序编号。同一文献的会议正式版、作者存档及软件项目分别标明。页码有多版编排的会议文献使用 DOI 对应版本定位。\n\n'+'\n\n'.join(f'**[{i}]** {r[0]}\n\n{r[2]} [核验来源]({r[1]})' for i,r in enumerate(refs,1))+'\n',encoding='utf8')
print(len(registry),'source identities')
