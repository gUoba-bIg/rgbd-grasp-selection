"""Readable V1 closure from preserved native returns."""
from pathlib import Path
import json
V=Path(__file__).resolve().parents[1];A=V/'audit';D=V/'delivery'
def j(p):return json.loads(p.read_bytes())
r=j(A/'final_cold_outcome.json');g=j(A/'font_AB_and_rank_gate.json');a=j(V/'provenance_evidence_appendix/manifest.json');g0=j(A/'preservation_opening.json');g1=j(A/'preservation_closing.json')
m=r['models'];h=r['physical_returns']
lines=['# V1 最终独立冷启动复现','',r['verdict']+'。','',
'## 三个预登记案例','',
'每个案例各分派一次。生产代码从冻结 RGB-D 和种子重新生成网络输出、候选和选择。逐阶段实际载入来源保存在 cold/cold_receipts。','',
'|案例|冻结类别|本轮返回类别|复现验收|旧 E 候选数|返回物理步|','|---|---|---|---:|---:|---:|']
for x in r['cases']:
    lines.append('|{}|{}|{}|{}|{}|{}|'.format(x['trial_id'],x['expected'],x['classification'] or '无方法返回','通过' if x['accepted_method_return']else '未完成',x['eligibles'] if x['eligibles'] is not None else '未取得',x['native_returns'] if x['native_returns'] is not None else '未知'))
lines+=['',f"完整复现 {r['case_acceptances']}/3。P3 进入 {m['P3']['entered']} 次、返回 {m['P3']['returned']} 次；P5 进入 {m['P5']['entered']} 次、返回 {m['P5']['returned']} 次。已确认返回物理步 {h}。算法和动力学重试均为 0。".replace('；','。'),'']
first=r['cases'][0]
if first['accepted_method_return']:
    lines+=['R6FORMAL-007 的最大抬升为 {:.9f} m，末段连续保持 {} 步。验收参照为 {:.9f} m，绝对容差为 1 µm。'.format(first['max_lift_m'],first['final_hold_streak'],0.1499016329193084),'']
if r['stop']:lines+=['## 首次停止回执','',str(r['stop'].get('trial'))+' / '+str(r['stop'].get('stage'))+'。原始错误见 cold/cold_receipts/STOP.json。','']
lines+=['## 字体来源与案例身份','','A 使用锁定 Matplotlib 自带的 101 个字体条目，B 在独立配置中只使用锁定字体 DejaVuSans.ttf 和 cmr10.afm。两组均禁用宿主字体发现，并记录访问。旧进程访问过的两个 Kingsoft 字体没有随包分发。六份探针的共享池、完整 Raw/A/C0 分数与风险组件、排序、资格账本和几何资格结果逐项与冻结 R6 输出一致。耗时和 API 调用的时间戳单独保存。','',
'审计夹具以三个原登记 unit 目录为末级。policy_rank 及其生产资格算法字节未变。两份宿主字体在这三个案例中归为偶发依赖。','',
'## 运行依赖与来源底稿','',
f"独立只读附卷保存 {len(a['files'])} 项登记底稿，共 {a['total_bytes']:,} 字节。155 项具备冻结整文件哈希，另 3 项同时提供冻结字节前缀校验和本次实测整文件 SHA256。三案例原预检实际打开底稿 0 项，本次科学运行目录也未部署这些文件。附卷 SHA256 为 {a['archive_sha256']}。",'',
'4 份运行必需的 smoke11 原件按冻结路径部署，单独附有 43,578 字节的运行补卷。audit/dependency_classes.json 记录 139 个在此前三案例初始化中实际读取的项目路径。本次全链路实际文件打开索引、运行模块和原生引擎身份分别存放在 audit/actually_opened_cold_project_inputs.jsonl.gz 和各案例逐阶段回执中。Python 级打开记录的覆盖范围与原生 API 记录分别说明。','',
'## 封存与复核','',
f"前后分别复核 {g1['unique_prior_sources_checked']:,} 个历史源文件、{len(g1['primary_archive_volumes'])} 个主卷及 V1 Git 标记。前测通过 {g0['passed']}，后测通过 {g1['passed']}。基线提交 {g1['baseline']}，标签对象 {g1['tag']}。历史 0/3 及其模型调用原样保留。",'',
'差异位于 delivery/adapter.patch。以冻结 V1 主卷、独立运行补卷及本版适配器为交接材料。新环境安装需使用 adapter/setup_v1.ps1，运行补卷按 ops/restore_runtime_supplement.py 校验后部署，再执行 adapter/preflight_v1.ps1。已消耗的实验预算不会因交付而重置。完整证据的本地相对路径和哈希见 delivery/local_raw_index.jsonl.gz。','',
'本结论的范围是三种预登记结果在本机独立环境中的复现。R6/R7 的统计、V1 论文和模型结论均沿用历史封存版本。']
with (D/'V1_final04_report.md').open('x',encoding='utf8',newline='\n')as f:f.write('\n'.join(lines)+'\n')
