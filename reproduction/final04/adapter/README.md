# V1 双语论文与 R6/R7 可复现封存

本目录是新建封存控制仓库。上游检出的提交、工作树差异以及项目检查时缺少历史仓库的原始回执见 metadata/git_before.json。新标签 paper_v1_r6r7_frozen 的提交和创建时间以 Git 对象及仓库外的 tag_identity.json 为准。ZIP 条目采用规范化的包头时间；原研究和本次封存日期分别来自原始回执和新提交。

## 恢复布局

将 v1_control.zip 解压为 v1_baseline 目录；将各数据分卷放到同级 v1_archive。各 ZIP 均可单独读取，不是分片压缩。只读副本在 v1_archive_readonly，独立的 SHA256 清单在 external_checksums。Windows 只读属性、独立物理文件、逐件哈希与 Git 标签分别核验。研究资料的长期离线保存可使用目录整体复制。原作者目录与原 Python 环境无需存在。

建议在有足够磁盘的 Windows 11 x64、同型号 NVIDIA GPU、同一驱动与同一架构上复现，见 metadata/platform_capture.json。CUDA 运行库在已锁定的 Torch 安装包内；系统驱动由宿主提供。所有实验操作在新目录中执行。setup 使用实际随附的 Python、离线 wheels 和绑定校验生成 vision、rank、physics 三个独立环境。快照和带哈希安装锁分别保存。原始提交若无记录则按缺失登记。

在包含本目录及各档案的父目录运行 PowerShell，如 D:\repro 。Target 选择未用目录，避免在历史路径或已有输出上运行。

```powershell
& .\v1_baseline\verify_v1.ps1 -Archives .\v1_archive -Deep
& .\v1_baseline\setup_v1.ps1 -Target .\cold01 -Archives .\v1_archive
& .\v1_baseline\preflight_v1.ps1 -Target .\cold01
& .\v1_baseline\run_smoke_v1.ps1 -Target .\cold01
& .\v1_baseline\summarize_v1.ps1 -Target .\cold01
```

## 审查入口

- metadata/cold_preregistration.json 固定三个案例、输入、来源、预期类别和数值界限，以及模型前向、物理步和包装修复预算。
- metadata/stage_seeds.json 和 stage_seed_sources.json 给出各阶段原始工作种子。模型采样及退化还使用独立的场景种子。metadata/smoke_expected.json 仅在新评分完成后核证。
- execution_project.zip 封存运行源码、配置、模型、资产、配准先验、定制引擎、上游构建与修改、观察、稳定状态及意图。native_build_inputs.zip 与 offline_wheels.zip 补足构建输入和离线安装件。锁中的每行给出原相对路径、字节数和 SHA256。
- source_R6_initial.zip 和 source_R6_final.zip 分别保存最初停止证据及最终正式证据。source_R7_final.zip 保存消融结果。source_R8_paper.zip 与 source_bilingual_final.zip 分别保留初稿证据链和分开的中、英文成稿。
- 冷启动以原原始 RGB、深度缓冲和相机矩阵为输入，用冻结模型实际前向。对比输入张量和解码行、全部排序、候选池、E 与轨迹，再进入冻结的零步或动力学入口。评分或引擎日志的耗时变化会改变文件哈希；适配器只允许更换新工作目录中与此相关的回执绑定，并保留原意图和新旧映射。任何算法、执行或判定差异均导致停止。
- 每个角色的模块和原生库实际加载路径、模型调用、首次失败、物理步和输出均在 Target 下留存。没有完成的案例保持未执行。冷启动是独立环境中的开发复现，不提供新论文效果估计。

补充卷 source_supplement.zip 保存既有清单之外的目录文件和排版核验图。signed_display_driver.zip 提供本机已签名显示驱动的实际安装文件，setup 只核验运行环境。
