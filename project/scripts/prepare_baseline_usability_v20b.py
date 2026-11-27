"""Freeze one supervised reuse trial and its planning denominator before inference."""
import sys,json,hashlib,ast,shutil
from pathlib import Path
from datetime import datetime
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.pipeline import project_lock
from scripts import run_baseline_usability_v20b as run
import numpy as np
from PIL import Image
from scipy.io import savemat

def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def digest(p):return run.state(ROOT/p)
def write(p,v):run.save(ROOT/p,v)
def text(p,s):
    p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:f.write(s)

with project_lock(ROOT):
    log='logs/P4/baseline_usability_v20b'
    assert not (ROOT/log/'freeze.json').exists(),'Do not prepare an already frozen trial'
    assert not (ROOT/'results/P4/baseline_usability_v20b').exists(),'Prior trial evidence exists'
    now=datetime.now().astimezone().isoformat()
    parent=read('configs/segmentation_baseline_v16.json');training=read('results/P4/segmentation_baseline_v16_training.json')
    assert training['protocol_sha256']==digest('configs/segmentation_baseline_v16.json')['sha256']
    arm=training['arms']['rgbd'];trained=arm['result'];manifest=read(parent['data_manifest'])
    protected={}
    def protect(p,expected=None):
        p=str(Path(p).as_posix());v=digest(p)
        if expected is not None:assert v['sha256']==expected,'Original hash differs: '+p
        protected[p]={k:v[k] for k in ['sha256','bytes']}
    for p,h in dict(parent['source_hashes'],**parent['input_hashes']).items():protect(p,h)
    for p in ['configs/segmentation_baseline_v16.json','results/P4/segmentation_baseline_v16_training.json','results/P4/segmentation_baseline_v16_acceptance.json','results/P4/mask_ap_v17_acceptance.json','runtime/pipeline/state.json','configs/stages.json']:
        protect(p)
    checkpoint=Path(trained['checkpoint_path']).as_posix();protect(checkpoint,arm['checkpoint_sha256'])
    rundir=Path(checkpoint).parent.parent
    for name in ['config','history','initialization','result']:
        protect((rundir/(name+'.json')).as_posix(),arm[name+'_sha256'])
    records=[]
    for source,partition,scenes in [('train','train',parent['train_scenes']),('validation','development',parent['validation_scenes'])]:
        chosen=sorted([r for r in manifest['records'][source] if r['frame']=='0000'],key=lambda r:r['scene'])
        assert [r['scene'] for r in chosen]==sorted(scenes)
        for rec in chosen:
            for kind,p in rec['files'].items():protect((Path(parent['data_root'])/p).as_posix(),rec['sha256'][kind])
            cache=f'results/P4/segmentation_baseline_v16_evaluation/rgbd/frames/{partition}/scene_{rec["scene"]:04d}_{rec["frame"]}/prediction.npz'
            receipt=read(cache+'.sha256.json');protect(cache,receipt['sha256']);protect(cache+'.sha256.json')
            protect((Path(cache).parent/'record.json').as_posix())
            records.append(dict(partition=partition,record=rec,cached_prediction=cache))
    assert len(records)==30
    c=dict(schema='20b',status='frozen',created_at=now,question='Can the accepted R16 RGBD checkpoint be reused as a limited development instance baseline with a deterministic metric-depth interface?',
        scope='Supervised one-round baseline reuse; not repair of R18 weights, held-out validation, full P4 completion, full P6, or method superiority.',
        parent_protocol=parent,checkpoint=checkpoint,checkpoint_config_fingerprint=trained['config_fingerprint'],checkpoint_data_fingerprint=trained['data_fingerprint'],
        selection_rule='Frame 0000 in every existing 20 training and 10 previously exposed development scene. Fixed by identity before any new outcome; no outcome-driven selection.',records=records,
        max_new_forward_calls=30,max_new_training_updates=0,max_new_AP_calls=0,points_per_instance=32,
        gates=dict(each_frame_nonempty=True,each_frame_at_least_one_iou50_match=True,each_frame_matched_instance_has_valid_depth=True,all_predictions_exactly_repeat_frozen_cache=True,projection_tolerance_pixels=1e-6,inverse_formula_tolerance_metres=1e-9),
        missing_depth_policy='Record/exclude invalid-depth pixels; never invent depth. Require at least one matched instance with valid depth per frame.',
        resources=dict(run_seconds=600,verify_seconds=180,cumulative_seconds_cap=780,CPU_threads=4,rss_bytes=4*1024**3,gpu_bytes=2*1024**3,output_bytes=64*1024**2,disk_reserve_bytes=60*1024**3),
        output='results/P4/baseline_usability_v20b',protected_files=protected,
        stop_rule='Exactly one run and one zero-forward full readonly verification. Stop afterwards regardless of scientific outcome; technical failure also stops. No retry, follow-on experiment, training, or automation without new user confirmation.',
        counts_scope='30 new actual returned model calls maximum. Historical calls are not relabelled as new. Historical R18 two unknown outcomes remain unknown. Constructor RNG is technical initialization, not Monte Carlo scientific sampling.')
    # Small synthetic checks only: no model forward, no real-frame evaluation.
    pred=np.array([[1,1,0],[0,2,2]],dtype=np.int32);gt=np.array([[3,3,0],[0,4,4]],dtype=np.int32)
    assert run.pairs(pred,gt)==[[1,3],[2,4]]
    assert len(run.pairs(pred,gt))==run.old.frame_metrics(pred,pred>0,gt)['instances']['tp']
    assert run.pairs(np.zeros_like(pred),gt)==[]
    fdir=ROOT/log/'preflight_fixture';fdir.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.full((720,1280),1500,np.uint16)).save(fdir/'depth.png')
    savemat(fdir/'meta.mat',{'intrinsic_matrix':np.array([[600.,0,640.],[0,600.,360.],[0,0,1.]]),'factor_depth':np.array([[1000.]])})
    fixture=dict(c,parent_protocol=dict(c['parent_protocol'],data_root=log+'/preflight_fixture'))
    small=np.zeros((240,320),np.int32);small[10:20,10:20]=1
    arrays,g=run.geometry(small,{'files':{'depth':'depth.png','meta':'meta.mat'}},fixture)
    assert g['passed'] and g['point_count']==32 and np.all(arrays['depth_m']==1.5)
    assert np.all(arrays['xyz_m'][:,2]==1.5)
    for p in ['scripts/run_baseline_usability_v20b.py','scripts/prepare_baseline_usability_v20b.py']:
        ast.parse((ROOT/p).read_text(encoding='utf-8'));protect(p)
    write(log+'/preflight.json',dict(status='passed',time=now,checks=['source/data/weight/cache identities','fixed 30 unique scene/frame identities','maximum-cardinality matching exact synthetic case','empty matching case','metric-depth projection/inverse/reprojection synthetic case','Python syntax'],new_model_forward_calls=0,new_training_updates=0,fixture_scope='Synthetic preparation only, not formal trial pass'))
    # Fixed planning weights, not elapsed-time measurements. No points for test/script counts.
    packs=[
      ('P0','环境与恢复',3,[('环境和恢复验收',3,'done')],['runtime/pipeline/state.json#P0']),
      ('P1','数据划分与协议',3,[('划分、隔离和协议',3,'done')],['runtime/pipeline/state.json#P1']),
      ('P2','数据与标注、算子环境',6,[('算子与抓取环境',2,'done'),('已有训练开发数据质量',2,'done'),('主计划最终数据覆盖闭环',2,'pending')],['runtime/pipeline/state.json#P2O,P2G7,P2E8']),
      ('P3','抓取基线',8,[('候选、碰撞及官方子集验证',2,'done'),('实例关联与四视角基线',2,'done'),('主计划完整基线评价与闭环',4,'pending')],['results/P3/instance_grasp_acceptance_v11.json','results/P3/grasp_multiview_acceptance_v14r1.json']),
      ('P4','稳定实例基线',10,[('RGB/RGBD固定预算配对训练和实例评价',3,'done'),('官方mask AP评价',1,'done'),('基线身份、用途与有限复用决定',1,'pending_current_trial'),('固定小范围实例到深度接口验证及真实结论',1,'pending_current_trial'),('最终稳定基线覆盖与协议闭环',2,'pending'),('正式抓取链路接入验证',2,'pending')],['results/P4/segmentation_baseline_v16_acceptance.json','results/P4/mask_ap_v17_acceptance.json']),
      ('P5','质量预测和门控融合',10,[('既定配对门控试验及负结果收尾',2,'done'),('有效实例基线下质量预测实现',4,'pending'),('门控有效性对照和取舍',4,'pending')],['results/P4/dualstream_evaluation_v18_closeout.json']),
      ('P6','不确定性几何传播',6,[('合成解析传播及独立数值验证',2,'done'),('真实数据与实例边界接口',2,'pending'),('目标条件下覆盖和误差验收',2,'pending')],['results/P6/geometry_propagation_v19_closeout.json']),
      ('P7','风险模型与概率校准',10,[('局部风险特征和已知诊断证据',1,'done'),('风险模型训练',3,'pending'),('独立校准',3,'pending'),('预设评价和可靠性检查',3,'pending')],['results/P3/grasp_local_risk_acceptance_v15r1.json']),
      ('P8','全链路与消融',8,[('B0至B6全链路对照',4,'pending'),('预设比较和科学关口',4,'pending')],[]),
      ('P9','三随机种子正式实验',8,[('P8通过后的三种子正式运行',6,'pending'),('跨种子稳定性结论',2,'pending')],[]),
      ('P10','独立测试和鲁棒性',8,[('保留测试集评价',4,'pending'),('预设扰动和鲁棒性评价',4,'pending')],[]),
      ('P11','正式统计与图表',8,[('场景层统计、多重比较、区间',4,'pending'),('正式结果表和论文图',4,'pending')],[]),
      ('P12','论文和复现交付',12,[('完整论文、修改和排版',8,'pending'),('复现包、说明和最终检查',4,'pending')],[])]
    items=[]
    for pid,name,w,sub,evidence in packs:
        assert sum(x[1] for x in sub)==w
        for e in evidence:assert (ROOT/e.split('#')[0]).exists(),e
        items.append(dict(id=pid,name=name,weight=w,subdeliverables=[dict(name=n,weight=q,status=s) for n,q,s in sub],evidence=evidence,earned=sum(q for n,q,s in sub if s=='done')))
    earned=sum(i['earned'] for i in items);assert sum(i['weight'] for i in items)==100 and earned==23
    ledger=dict(version='20261027_supervised_v1',created_before_trial_at=now,scope_reference='docs/分阶段训练方案.html',method='Fixed planned deliverable weights, an explicit subjective planning estimate; not exact hours, a measured completion rate, or a confidence interval. Historical 30/40 estimates retired. Scientific failures count as completed investigation work; scientific gates remain independent.',weight_basis='P0-P10 experiments 80 points; statistics/figures 8; paper/reproduction 12. P4/P5/P7 carry greater implementation/training/integration scope; synthetic-only P6 carries 6. No stage/script/test/frame-count denominator.',total_weight=100,experiment_weight=80,earned_points_before_trial=earned,all_percent_before_trial=earned,experiment_percent_before_trial=earned/80*100,packages=items,current_trial_credit_rule='Award the two P4 pending_current_trial points only after a completed 30-frame trial and passing zero-forward full readonly verification with a truthful reuse/reject decision. A scientifically negative outcome still completes that investigation; it does not close P4 or justify P5/P8.',weights_changed_after_outcome=False)
    write('logs/progress_audit/supervised_v1_20261027/weights_before_trial.json',ledger)
    lines=['总进度固定工作包口径（监督模式第一版）',now,'这是公开的计划工作量加权估算，不是精确工时测量或统计置信区间。','以原P0至P12主交付物为100点，实验P0至P10占80点，正式统计图表8点，论文复现12点。','旧30%/40%没有固定分母，作废为历史粗估；与新数值不可作同口径涨跌比较。','计已完成工作，负结果计入，科学可用性单列。不得按脚本/测试/局部帧次计总进度。','本轮开始前：全部任务23%；实验23/80=28.75%（对外约29%）。','本轮仅在30帧完整试验及只读复验通过且形成真实结论后，可记2点：全部25%，实验25/80=31.25%（对外约31%）。','权重按主交付物预计工作复杂度首次明确固定，不声称过去实际耗时比例；以后范围变化必须版本化说明。','']
    for i in items:
        lines.append(f"{i['id']} {i['name']}：已完成{i['earned']}/{i['weight']}点")
        lines.extend(f"  {s['weight']}点 {s['name']}：{s['status']}" for s in i['subdeliverables'])
    text('docs/总进度固定工作包_监督版20261027.txt','\n'.join(lines)+'\n')
    text('docs/第二十轮B基线复用单轮验证方案.txt','\n'.join([
      '第二十轮B：用户监督的单轮基线复用验证',now,
      '目的：恢复一条有实例输出的后续开发基线，验证实例到米制三维点的接口。沿用R16 RGBD最终1600步权重和原解码阈值；R18双流零实例负结果保持原样。',
      '不执行原R20的1280单元诊断/81段调度，不重训，不重算整套官方AP，不改变任何旧冻结输入、源码、数据划分、训练预算。',
      '固定选择：20个原训练场景与10个已暴露开发场景各0000帧，共30帧，选择与当前输出无关。不能作为独立测试或泛化证据。',
      '预设可用门槛：每帧存在实例且有至少一个IoU>=0.5的真值匹配，至少一个匹配实例有有效深度；新实例与前景图逐像素复现原冻结缓存；米制反投影两公式误差<=1e-9m、重投影误差<=1e-6像素。',
      '240x320标签整数扩展3x4到720x1280，保留实例ID；每实例至多32个确定性选取的有效深度像素。无深度的实例记录并排除，不伪造坐标。此检查不证明真实传感器标定、协方差有效性或抓取成功。',
      '预算：最多30次新模型前向、0次训练/反向/官方AP。运行600秒、只读180秒，累计上限780秒；每进程CPU4线程、RSS4GiB、GPU2GiB，结果64MiB、磁盘余量60GiB。资源检查按帧抽样，超时由看门狗强停，非操作系统硬内存配额。',
      '只运行一次，完成后实际执行一次0前向的完整只读复验，然后停止。科学门槛未过仍保留全部结果；技术错误立即停止，不自动修改后重跑。任何重试和下一实验必须用户再次确认。',
      '保护范围：原26项源码绑定、原输入协议、选中120个原数据文件、最终权重与元数据、30个冻结预测及凭据、旧状态和配置；SHA/bytes/mtime前后相同。协议和本轮runner另外冻结绑定。',
      '进度按docs/总进度固定工作包_监督版20261027.txt；方法失败不抹去已完成工作，也不记为完整科学阶段通过。',
      '自动续接任务保持删除，历史恢复说明不构成继续授权。']))
    write('configs/baseline_usability_v20b.json',c)
    write(log+'/freeze.json',dict(created_at=datetime.now().astimezone().isoformat(),protocol_sha256=digest('configs/baseline_usability_v20b.json')['sha256'],runner_sha256=digest('scripts/run_baseline_usability_v20b.py')['sha256'],planning_weights_sha256=digest('logs/progress_audit/supervised_v1_20261027/weights_before_trial.json')['sha256'],plan_sha256=digest('docs/第二十轮B基线复用单轮验证方案.txt')['sha256'],protected_count=len(protected),max_new_forward_calls=30,automatic_retry=False,next_experiment_requires_user_confirmation=True))
    previous=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(previous,ROOT/log/'handoff_before_supervised_trial.txt')
    previous.write_text('当前授权更新：'+now+'\n用户改为亲自监督，仅授权一次R20B旧有效基线复用验证（30帧，0训练），完成及只读复验后自动停止；失败也停止，重试及下一实验必须用户确认。自动任务仍删除。不要按下方历史自动续接授权继续。\n全部任务总进度：固定计划工作包估算23%；实验总进度：28.75%（约29%）。本轮正式运行尚未开始，0/30。权重和计分见docs/总进度固定工作包_监督版20261027.txt。\n\n以下为保留的旧停止状态及历史：\n'+previous.read_text(encoding='utf-8-sig'),encoding='utf-8')
    print(json.dumps(dict(status='frozen_ready_for_single_supervised_run',records=len(records),protected=len(protected),all_percent=23,experiment_percent=28.75,new_forward_calls=0),ensure_ascii=False))
