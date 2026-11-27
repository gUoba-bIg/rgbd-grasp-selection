"""Finite P6 source/data freeze and targeted CPU mathematical fixtures."""
import sys,ast
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v24 import *
from scripts.pipeline import project_lock
from src import p6_image_geometry_v24 as g
from src import geometry_propagation_v19 as math19
import numpy as np
from PIL import Image

def tests():
 checks={};K=np.array([[600.,0,630],[0,500,350],[0,0,1.]])
 for z in [.2,1.8]:
  nominal=[17.,633.,z];sg=.003;v=math19.propagate_gaussian(nominal,K,np.diag([0.,0.,sg**2]),[0.,0.,sg]);ray=np.array([(17-630)/600,(633-350)/500,1]);checks['ray_'+str(z)]=bool(np.allclose(v['first_order_covariance'],sg**2*np.outer(ray,ray),rtol=0,atol=1e-20) and v['first_order_spectrum']['numerical_rank']==1)
  dist=math19.support_distance(v['nominal_output']+2*sg*ray,v['nominal_output'],v['first_order_covariance']);checks['rank1distance_'+str(z)]=bool(np.isclose(dist['squared_distance'],4))
 grid=np.arange(720*1280,dtype=np.int32).reshape(720,1280);pil=np.array(Image.fromarray(grid).resize((640,480),Image.Resampling.NEAREST));ys=((2*np.arange(480)+1)*720)//960;xs=((2*np.arange(640)+1)*1280)//1280;checks['pixel_center_mapping']=bool(np.array_equal(pil,grid[np.ix_(ys,xs)]))
 m=np.zeros((5,7),np.int32);m[1:4,2:5]=1
 from scripts.p6_accept_v24 import ref_boundary,ref_select
 checks['boundary']=bool(np.array_equal(g.boundary(m),ref_boundary(m)))
 for n in [0,1,17,4096]:
  mask=(np.arange(71)%3!=0).reshape(1,71);checks['selection_'+str(n)]=bool(np.array_equal(g.select(mask,n),ref_select(mask,n)))
 checks['frozen_old_quality_passed']=read(OLD/'quality_screen.json')['quality_usable'];checks['P5_finite_repair_closed']=read(ROOT/'logs/P5/quality_gate_v23r1/root_final_review.json')['passed'];checks['R19_only_reused']=read(ROOT/'results/P6/geometry_propagation_v19_closeout.json')['full_P6_completed'] is False
 need(all(checks.values()),'P6 fixtures failed '+str(checks));once(LOG/'component_tests.json',dict(passed=True,checks=checks,new_model_calls=0,new_monte_carlo_samples=0));return checks

def prepare():
 need(not CONFIG.exists(),'P6 already frozen');tests();old=read(ROOT/'configs/p5_quality_gate_v23.json');c={k:old[k] for k in ['data_root','train_records','development_records','monitor_records','decoder','seed','quality_screen_epoch','core_conditions','runtime','input_size','split_path']}
 c.update(version=24,created_at=now(),scope='P6 image-linked conditional depth uncertainty, original full main dependency graph not modified',route='docs/P6开发协议与后续路线_v24.txt',quality_checkpoint='experiments/p5_quality_gate_v23/quality_final.pt',fixed_segmentation_checkpoint='experiments/p5_quality_gate_v23/epochs/09_fixed.pt',fixed_segmentation_source='results/P5/quality_gate_v23/exports',quality_sigma_units='metres, conditional zero mean hypothesis',input_coordinate_covariance='diag(0,0,sigma_z^2); fixed pixels and K',output_rank=1,main_points_per_case=4096,boundary_extra_points=1024,resources=dict(seconds=1800,forward_limit=704,corruption_replay_limit=2500,new_output_bytes=4*1024**3,cumulative_output_bytes=5*1024**3,rss_bytes=6*1024**3,gpu_bytes=8*1024**3,disk_reserve_bytes=60*1024**3,max_training=0,max_official_AP=0,max_new_monte_carlo=0,max_engineering_repairs=1),progress=dict(before_all=37,before_experiment=46.25,interface_points=2,complete_error_evaluation_points=2,after_all_if_accepted=41,after_experiment_if_accepted=51.25),protected={})
 old_final=read(ROOT/'logs/P5/quality_gate_v23/final_readonly.json')['files'];paths=set(old['protected'])
 for p in old_final:
  if p.startswith('results/P5/quality_gate_v23/exports/core_fixed_') or p.startswith('results/P5/quality_gate_v23/quality_screen'):paths.add(p)
 paths.update(['configs/p5_quality_gate_v23.json','logs/P5/quality_gate_v23/final_readonly.json','logs/P5/quality_gate_v23/root_final_review.json','results/P5/quality_gate_v23/closeout.json','results/P5/quality_gate_v23/selection.json','logs/P5/quality_gate_v23r1/root_final_review.json','results/P5/quality_gate_v23r1/closeout.json','results/P6/geometry_propagation_v19_closeout.json','results/P6/geometry_propagation_v19_acceptance.json','logs/P6/geometry_propagation_v19/readonly_bridge_v19r1/proof.json','configs/geometry_propagation_v19.json','configs/stages.json','runtime/pipeline/state.json','src/geometry_propagation_v19.py',c['quality_checkpoint'],c['fixed_segmentation_checkpoint'],'docs/P6开发协议与后续路线_v24.txt','logs/P6/image_geometry_v24/component_tests.json'])
 paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'scripts').glob('p6_*_v24.py'));paths.add('src/p6_image_geometry_v24.py')
 for p in sorted(paths):
  v=stat(ROOT/p)
  if p in old_final:need(v==old_final[p],'Previously accepted P5 source changed '+p)
  elif p in old['protected']:need(v==old['protected'][p],'P5 frozen input changed '+p)
  c['protected'][p]=v
 need(read(OLD/'selection.json')['epoch']==9,'Fixed parent not epoch9');need(read(ROOT/'runtime/pipeline/state.json')['stages']['P6E19']['status']=='complete','R19 acceptance missing')
 c['old_P6_output_bytes']=sum(p.stat().st_size for folder in [ROOT/'results/P6',ROOT/'logs/P6'] for p in folder.rglob('*') if p.is_file() and not p.is_relative_to(BASE) and not p.is_relative_to(LOG))
 once(CONFIG,c);once(LOG/'freeze.json',dict(protocol_sha256=sha(CONFIG),protected_files=len(c['protected']),new_research_started=False));status('frozen',case_total=800,new_training=0,route_revision=24)
 handoff=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(handoff,LOG/'handoff_before_P6.txt');oldtext=handoff.read_text(encoding='utf-8-sig');handoff.write_text('最新：'+now()+' 用户已授权进入P6及调整后续路线。v24协议已冻结：普通固定RGBD+质量预测+深度条件三维传播，删除未获支持门控；160帧×5条件共800项，首100项工程预检后补700质量帧，0训练/新AP/MC。独立完整复验、冷重入、汇总后停止，不进入P7，不恢复自动任务。全部37%、实验46.25%，未完成不预记分。正式入口scripts/p6_deliver_v24.py；状态runtime/pipeline/p6_part_v24.json，先核进程和项目锁禁止重复启动。协议和后续对照见docs/P6开发协议与后续路线_v24.txt。原P2/P3依赖仍未闭环，原state.json不改。\n\n历史：\n'+oldtext,encoding='utf-8')
 control=ROOT/'runtime/pipeline/part_level_supervision.json';shutil.copy2(control,LOG/'supervision_before_P6.json');row=read(control);row.update(recorded_at=now(),current_authorized_part='P6图像与实例边界传播开发及后续路线修订',current_execution_state='P6_v24_frozen',current_execution_evidence=CONFIG.relative_to(ROOT).as_posix(),current_runner='scripts/p6_deliver_v24.py',remaining_research_updates_cap=0,waiting_for_user_next_part_confirmation=False,summary_report=None,active_internal_repair=None);atomic(control,row);print('P6 v24 frozen',flush=True)
if __name__=='__main__':
 with project_lock(ROOT):prepare()
