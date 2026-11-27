"""Build a manifest-addressed, transferable core; the complete native streams stay on disk."""
from pathlib import PurePosixPath
from collections import Counter,defaultdict
import csv,gzip,io,os,shutil,zipfile
from common import *
def local_files():
    for directory,dirs,files in os.walk(HERE,followlinks=False):
        q=Path(directory);dirs[:]=sorted(n for n in dirs if n not in ('workspaces','delivery','__pycache__','codec_calibration') and not (q/n).is_symlink() and not (q/n).is_junction())
        for name in sorted(files):
            p=q/name
            if name in ('SHA256SUMS.txt','local_evidence_index.csv.gz') or name.endswith(('.pyc','.writing')):continue
            yield p
def small(p):
    rel=p.relative_to(HERE);k=rel.parts;n=p.name
    if k[0] in ('preregistration','protocol','history','code','implementation','analysis','checkpoints','audit'):return True
    if k[0]=='scenes':return n in ('receipt.json','settled_state.json')
    if k[0]=='units':return len(k)==3 and n in ('network.json','raw_decoder.npy','candidate_domain_ledger.json','pool.npz','orders.json','selection.json','ranking_A.json','ranking_C0.json','ranking_Raw.json','forward_intent.json','platform_input.json')
    if k[0]=='trials':
        if n in ('intent.json','method_result.json','worker_open.json','safety_receipt.json','no_candidate_zero_receipt.json','path_result.json','summary.json','static_classification_receipt.json','dynamic_integrity.json','no_candidate_integrity.json'):return True
        return 'episode' in k and n=='result.json'
    if k[0]=='run':return len(k)==2 or n in ('dispatch.json','return.json','launch.json','exception.json')
    return len(k)==1
def ftxt(path,text):
    with path.open('x',encoding='utf8',newline='\n') as f:f.write(text)
def terminal_rows():
    dest=HERE/'analysis/final_hold_observations.csv.gz';cats={};reasons=defaultdict(Counter)
    fields=['trial_id','method','step','lift_m','effective_left_target_contact','effective_right_target_contact','aggregate_table_support','table_normal_force_sum_N','slipped','engineering_violation_count']
    with gzip.open(dest,'xt',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for p in sorted((HERE/'trials').glob('*/episode/final_480_steps.json')):
            r=read(p);m=read(p.parent.parent/'intent.json')['method']
            rows=r['observed_rows'];cats[p.parent.parent.name]=len(rows)
            for v in rows:
                w.writerow(dict(trial_id=p.parent.parent.name,method=m,engineering_violation_count=len(v['engineering_violations']),**{k:v[k] for k in fields[2:-1]}))
    for r in read(HERE/'analysis/formal_ledger.json')['rows']:
        if r.get('failure_category')=='no_executable_candidate':
            counts=Counter(r['failure_detail']);reasons[r['method']].update(set(counts));reasons['candidate_events_'+r['method']].update(counts)
    put(HERE/'analysis/terminal_and_E_coverage.json',dict(terminal_window_rows_by_trial=cats,reason_counts_across_all_empty_E_method_trials={k:dict(v) for k,v in reasons.items()},reason_overlap='A trial can contain multiple candidate-level reasons; dedicated primary failure categories remain exclusive.'))
def make_report():
    r=read(HERE/'analysis/paired_statistics.json');o=read(HERE/'analysis/outcome_breakdown.json');d=read(HERE/'preregistration/design.json');ledger=read(HERE/'analysis/formal_ledger.json')
    budgets=r['assigned_per_method'];known=r['adjudicated']
    s=['# R6 正式试次审查报告','',f'已预登记 720 个方法试次，已判定 {known}，未分派 {ledger["not_dispatched"]}，已分派但未取得可信结果 {r["unknown_or_shared_failure"]}。','',
       '## 协议与输入','','续接版保留原 R6 的七份已判定回执。R6FORMAL-001 根据零物理步原始证据补充裁定为 static_distance_uncertified，工程门和抓取成功均为 false。原版停批记录保持封存。剩余 712 份意图逐字节继承；每种方法的成功分母仍为最初分派的 240 次。','',
       'R4 开发阶段 C0−A 为 0 个百分点，原扩大门未通过。本轮依据单独授权开展正式验证。R4 开发数据独立封存。R5 observed_contactability 接口准入未通过，正式评分链使用 R4 的原始 Raw、A、C0。','',
       '三个物体各使用四个登记 XY 位置。每个物体与位置组合的十个重复按预登记种子从 −45° 至 +45° 的十个方位角区间抽取，初始高度、滚转、俯仰及物理参数沿用冻结配置。每个基础场景渲染一次，clean、joint 各自形成一个配对单元。旧 E 使用仅含机器人及台面的同一套静态运动学筛选。每个单元内的方法按冻结排列串行启动，以独立仿真连接执行。','',
       '四个 CPU 工作目录和一个 GPU 工作目录独立。完整原始引擎调用、路径快照及动力学逐步遥测留在本地。核心包包含试次、候选、评分、首败、正式统计和原始证据哈希索引。','',
       '## 正式运行结果','']
    if r['complete']:
        s+=['| 方法 | 成功/全部分派 | 成功率 | 95% Wilson 区间 |','|---|---:|---:|---:|']
        for m,v in r['overall']['rates'].items():s.append(f'| {m} | {v["success"]}/{v["assigned"]} | {100*v["rate"]:.2f}% | {100*v["wilson95"][0]:.2f}% 至 {100*v["wilson95"][1]:.2f}% |')
        s+=['','| 配对比较 | 成功率差（百分点） | 配对分层重抽样 95% 区间（百分点） | 前者独有成功 | 后者独有成功 |','|---|---:|---:|---:|---:|']
        for name,v in r['overall']['paired'].items():
            lo,hi=v['cluster_CI95_percentage_points'];s.append(f'| {name} | {v["difference_percentage_points"]:+.2f} | {lo:+.2f} 至 {hi:+.2f} | {v["hi_only"]} | {v["lo_only"]} |')
        sc=r['overall']['selection'];cand=sc['counts'].get('different_candidates',0)
        s+=['',f'A、C0 实际选择了不同执行候选的单元为 {cand}/240。两种方法共同弃选 {sc["counts"].get("both_abstain",0)} 个单元。','',
            '| 分层 | 单方法分母 | Raw 成功 | A 成功 | C0 成功 | C0−A（百分点） |','|---|---:|---:|---:|---:|---:|']
        for k,v in r['strata'].items():s.append(f'| {k} | {v["assigned_units"]} | {v["rates"]["Raw"]["success"]} | {v["rates"]["A"]["success"]} | {v["rates"]["C0"]["success"]} | {v["paired"]["C0-A"]["difference_percentage_points"]:+.2f} |')
        s+=['','成功要求末 480 个返回步均有双侧有效接触，抬升至少 0.10 m，且无台面聚合支撑、滑落或禁止碰撞。主配对差按物体和位置分层，成组重抽同一重复下的两个观测条件，共 20,000 次。区间适用于这三个固定开发物体和四个位置的正式重复。配对不一致很少时，经验重抽样区间可能退化，不能据此证明方法等效。']
    else:
        s+=['尚无可用于报告完整正式成功率及配对差的 720 个可判定结果。未分派和结果未知均在账本单列。','']
        for name in ('PREP_STOP','METHOD_STOP','SEAL_STOP','CONTROLLER_STOP'):
            q=HERE/'run'/(name+'.json')
            if q.is_file():
                note=read(q);s.append(f'共享阻塞见 run/{name}.json。首项记录为 '+str(note.get('error',''))[:900]+'。')
    s+=['',f'取得可判定动力学结果的试次 {o["dynamics_entered"]} 个，对应 {o["returned_dynamics_steps"]} 个引擎返回步。场景采集的沉降单列。','',
        '逐项结果见 analysis/formal_ledger.json。配对、分层和失配对见 analysis/paired_statistics.json。失败归类及未运行或未知状态见 analysis/outcome_breakdown.json，耗时见 analysis/timing.json。','',
        '240 回合消融未启动。','']
    ftxt(HERE/'review_report.md','\n'.join(s))
def main():
    terminal_rows();make_report()
    delivery=HERE/'delivery';delivery.mkdir(exist_ok=True)
    index=HERE/'analysis/local_evidence_index.csv.gz';manifest=HERE/'SHA256SUMS.txt'
    hashes={};included={};total=0
    with gzip.open(index,'xt',encoding='utf8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['relative_path','bytes','sha256','in_transferable_core','storage'])
        for p in local_files():
            rel=p.relative_to(HERE).as_posix();h=sha(p);size=p.stat().st_size;total+=size;hashes[rel]=h;include=small(p)
            writer.writerow([rel,size,h,int(include),'core_and_local' if include else 'complete_local_native_evidence'])
            if include:included[rel]=h
    hashes[index.relative_to(HERE).as_posix()]=sha(index);included[index.relative_to(HERE).as_posix()]=sha(index)
    ftxt(manifest,''.join(v+'  '+k+'\n' for k,v in sorted(hashes.items())))
    # Inside the ZIP, LOCAL_SHA256SUMS covers the full local evidence. SHA256SUMS covers only this archive.
    zipname='R6formal720_续接轻量核心审查包.zip';path=delivery/zipname
    zhash=dict(included);zhash['LOCAL_SHA256SUMS.txt']=sha(manifest)
    internal=''.join(v+'  '+k+'\n' for k,v in sorted(zhash.items()))
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
        for key,h in sorted(included.items()):
            require(not PurePosixPath(key).is_absolute() and '..' not in PurePosixPath(key).parts,'unsafe package name')
            z.write(HERE/key,key)
        z.write(manifest,'LOCAL_SHA256SUMS.txt');z.writestr('SHA256SUMS.txt',internal.encode('utf8'))
    with zipfile.ZipFile(path) as z:
        names=z.namelist();require(len(set(names))==len(names)==len(zhash)+1,'archive names')
        require(z.read('SHA256SUMS.txt')==internal.encode('utf8'),'archive manifest')
        for name,h in zhash.items():
            with z.open(name) as src:require(hashlib.file_digest(src,'sha256').hexdigest()==h,'archive verification '+name)
    h=sha(path);ftxt(delivery/(zipname+'.sha256'),h+'  '+zipname+'\n')
    put(delivery/'verification.json',dict(utc=now(),core_path=path.relative_to(HERE).as_posix(),core_bytes=path.stat().st_size,core_sha256=h,archive_members=len(zhash)+1,all_included_member_hashes_verified=True,local_file_count=len(hashes),local_evidence_bytes=total,local_sha256_manifest=sha(manifest),index_sha256=sha(index),exclusions='Workspace input junctions, transient lane cache, calibration codec scratch copies and the archive itself. Frozen external assets and models are referenced by hash only.'))
if __name__=='__main__':main()
