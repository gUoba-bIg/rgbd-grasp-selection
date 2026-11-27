"""Close P7 and stop, with separate scientific decision and earned-work progress."""
import sys,json,shutil
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p7_common_v31 import *

def run():
 c=verify();review=read(LOG/'readonly.json');cold=read(LOG/'cold.json');need(review['engineering_passed'] and cold['engineering_passed'] and review['files']==cold['files'],'P7 complete review/cold missing')
 for p,s in review['files'].items():need(stat(ROOT/p)==s,'P7 closing file changed')
 from scripts.p23_common_v26 import verify as old_verify
 old_verify()
 events=[json.loads(s) for s in (LOG/'calls.jsonl').read_text(encoding='utf-8').splitlines()];counts={}
 for kind,limit in c['call_limits'].items():
  a=[r['call_id'] for r in events if r['kind']==kind+'_intent'];b=[r['call_id'] for r in events if r['kind']==kind+'_returned'];need(len(a)==len(set(a)) and len(b)==len(set(b)) and set(b)<=set(a) and len(a)<=limit,'Invocation accounting invalid');counts[kind]=dict(attempts=len(a),known_returns=len(b),unknown=sorted(set(a)-set(b)))
 fit=load_sealed(BASE/'fit_result.json');earned=9 if fit['fitted'] else 0;allpct=47+earned;exppct=allpct/80*100
 repairs=[]
 if not fit['fitted']:repairs=['先检查候选池正负样本支持与P3候选有效性；如需新范围，另建冻结方案，不把未知标签当负例。']
 elif not fit['scientific_usable']:
  if not fit['gates']['ranking_gain']:repairs.append('风险模型未优于简单规则：保留较优简单排序；后续仅在确定具体特征缺口后制定一次有预算的新方案。')
  if not fit['gates']['clean_noninferiority']:repairs.append('干净条件退化：检查扰动拟合占比与观测特征漂移；原结果保留，禁止从同一验证结果反复调阈值。')
  if not fit['gates']['calibration_nll'] or not fit['gates']['calibration_ece']:repairs.append('校准不合格：当前概率不能使用；后续须独立新校准数据和预先限定的校准模型，不能在本验证集反复选择。')
 result=dict(at=now(),engineering_passed=True,scientific_usable=fit['scientific_usable'],risk_fitted=fit['fitted'],conditions=review['conditions'],actual_candidates=review['candidates'],compute=counts,all_tasks_percent=allpct,experiment_percent=exppct,earned_work_points=earned,P8_started=False,stopped=True,repair_options_for_future_confirmation=repairs,scope='one seed; fixed 100 scenes x1view x5synthetic conditions; offline geometric event only; no reserved official test or robot success')
 sealed(BASE/'closeout.json',result);once(LOG/'progress.json',dict(before_all=47,before_experiment=58.75,after_all=allpct,after_experiment=exppct,earned_points=earned,reason='Risk fitting, independent calibration, and prespecified complete evaluation delivered' if earned else 'Insufficient support; risk fitting/calibration not delivered',denominators=dict(all=100,experiment=80),not_success_rate=True))
 hand=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(hand,LOG/'handoff_before_closeout.txt');hand.write_text('最新：'+now()+' P7固定范围已收尾并停止；风险训练='+str(fit['fitted'])+'，科学可用='+str(fit['scientific_usable'])+'，全部'+str(allpct)+'%、实验'+str(exppct)+'%。完整只读和实际冷重入通过；不启动P8/P9/周期任务，等用户检查。报告docs/P7固定候选风险与校准结果_v31.txt。\n\n历史：\n'+hand.read_text(encoding='utf-8-sig'),encoding='utf-8')
 report=ROOT/'docs/P7固定候选风险与校准结果_v31.txt';report.write_text('P7固定候选风险与独立校准结果\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\n主指标与概率指标：\n'+json.dumps(fit.get('reports',{}),ensure_ascii=False,indent=2)+'\n\n科学失败不改写为成功。工程验收通过只证明本范围文件、输入、标签和指标一致；本结果不是全保留测试、三种子或机器人成功率。\n',encoding='utf-8')
 ctrl=ROOT/'runtime/pipeline/part_level_supervision.json';shutil.copy2(ctrl,LOG/'supervision_before_P7_closeout.json');row=read(ctrl);row.update(recorded_at=now(),current_execution_state='P7_completed_stopped',all_tasks_percent_planning_estimate=allpct,experiment_percent_planning_estimate=exppct,current_progress_evidence='logs/P7/risk_v31/progress.json',current_execution_evidence='results/P7/risk_v31/closeout.json',user_confirmation_required_after_current_part_before_next_part=True,active_internal_repair=None);atomic(ctrl,row);atomic(ROOT/'runtime/pipeline/risk_v31.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()
