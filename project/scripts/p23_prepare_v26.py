"""Freeze complete pre-test asset acquisition; research model fitting is separate."""
import sys,re,ast
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p23_common_v26 import *
from scripts.pipeline import project_lock

def main():
 need(not CONFIG.exists(),'Data protocol already frozen')
 split=read(ROOT/'data/manifests/scene_splits.json');groups=split['groups'];need(set(groups['train'])|set(groups['validation'])|set(groups['calibration'])==set(range(100)),'Split scope differs');need(sum(len(groups[k]) for k in ['train','validation','calibration'])==100,'Overlapping split')
 c=dict(version=26,created_at=now(),revision='9073d1d4cb6c92272b6aa072545def5101e4fecb',split=groups,camera='realsense',frames=list(range(256)),limits=dict(bytes=120*1024**3,network_bytes=60*1024**3,seconds=21600,workers=4,chunk_bytes=16*1024**2),selected={},archives={},protected={},tolerance_url='https://drive.usercontent.google.com/download?id=1DcjGGhZIJsxd61719N0iWA7L6vNEK0ci&export=download&confirm=t',tolerance_archive_bytes=6083051520)
 for p in sorted((LOG/'indexes').glob('*.json')):
  index=read(p);name=index['archive'];c['archives'][name]={k:index[k] for k in ['url','size','etag']};c['protected'][p.relative_to(ROOT).as_posix()]=stat(p)
  for source,e in index['entries'].items():
   dest=None
   if name.startswith('train_') and (re.match(r'^scene_\d{4}/object_id_list.txt$',source) or '/realsense/' in source):dest='scenes/'+source
   elif name=='collision_label.zip' and int(re.search(r'scene_(\d+)',source)[1])<100:dest=source
   elif name=='grasp_label.zip' and int(re.search(r'(\d+)_labels',source)[1])<=70:dest=source
   elif name=='models.zip' and source.endswith(('/nontextured.ply','/textured.obj','/textured.sdf')) and int(source.split('/')[1])<=70:dest=source
   if dest:c['selected'][dest]=dict(e,archive=name,source=source)
 need(sum(p.endswith('/object_id_list.txt') for p in c['selected'])==100,'Missing scene inventories');need(sum('/rgb/' in p for p in c['selected'])==25600,'Missing full image scope')
 c['expected_output_bytes_without_tolerance']=sum(v['bytes'] for v in c['selected'].values());need(c['expected_output_bytes_without_tolerance']+c['tolerance_archive_bytes']<c['limits']['bytes'],'Selection exceeds disk budget')
 old=read(ROOT/'logs/P6/error_mixture_v25/final_readonly.json')['files'];c['protected'].update(old)
 for p,v in c['protected'].items():need(stat(ROOT/p)==v,'Previously frozen file differs '+p)
 sources=list((ROOT/'scripts').glob('p23_*_v26.py'))+[ROOT/'src/remote_zip.py',ROOT/'docs/P2P3补齐与P7自动衔接_v26.txt',ROOT/'data/manifests/scene_splits.json']
 for p in sources:
  if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
  c['protected'][p.relative_to(ROOT).as_posix()]=stat(p)
 once(CONFIG,c);once(LOG/'data_freeze.json',dict(at=now(),sha256=sha(CONFIG),files=len(c['selected']),bytes=c['expected_output_bytes_without_tolerance'],no_official_test_data=True));status('P2_data_frozen',total_files=len(c['selected']),P7_started=False)
 hand=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(hand,LOG/'handoff_before_v26.txt');hand.write_text('最新：'+now()+' 用户授权补齐P2/P3，验收无问题后自动进入P7，P7结束停止；不恢复周期任务。v26先补原100场景RealSense全256视角及前置标签资产，原70/15/15划分和旧结果不变。原预训练权重不能证明校准隔离，需训练来源仅70场景的新候选模型；P6概率失败保留。当前全部41%、实验51.25%，未预记完成。入口scripts/p23_fetch_v26.py；状态runtime/pipeline/prerequisites_v26.json。先检查真实进程和项目锁禁止重复启动。\n\n历史：\n'+hand.read_text(encoding='utf-8-sig'),encoding='utf-8')
 ctrl=ROOT/'runtime/pipeline/part_level_supervision.json';shutil.copy2(ctrl,LOG/'supervision_before_v26.json');row=read(ctrl);row.update(recorded_at=now(),current_authorized_part='补齐P2/P3并在通过后自动执行P7',current_execution_state='P2_data_frozen_v26',current_execution_evidence=CONFIG.relative_to(ROOT).as_posix(),current_runner='scripts/p23_fetch_v26.py',waiting_for_user_next_part_confirmation=False,user_confirmation_required_after_current_part_before_next_part=False,stop_after_part='P7',automatic_P7_only_after_actual_gates=True,summary_report=None);atomic(ctrl,row)
 print(json.dumps(dict(frozen=True,files=len(c['selected']),bytes=c['expected_output_bytes_without_tolerance']),ensure_ascii=False),flush=True)
if __name__=='__main__':
 with project_lock(ROOT):main()
