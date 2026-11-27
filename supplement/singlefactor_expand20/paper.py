"""Manuscript tables, figures and a readable report for the sealed, selected cohort."""
from paths import *
from ranking_analysis import csv_bytes

LABELS={"all":"合并","depth_only":"深度高斯噪声","shift_only":"水平配准偏移","missing_only":"15% 随机缺深","block_only":"矩形遮挡"}
ENG={"all":"Pooled","depth_only":"Depth noise","shift_only":"Registration shift","missing_only":"Missing depth","block_only":"Rectangular mask"}
COLORS={"Raw":"#31557d","A":"#ae5b25","C0":"#197568"}
EXACT={"success":"成功","empty_pool":"共享候选池为空","empty_old_E":"非空候选池、旧 E 为空","static_distance_uncertified":"入口未认证",
       "other_static_rejection":"其他静态拒绝","dynamic_hard_gate":"动力学硬门","forbidden_collision":"动力学禁止碰撞","slip":"滑落",
       "closure_failure":"未形成有效闭合","insufficient_lift":"抬升不足","hold_failure":"保持失败","timeout":"超时"}
def textfile(name,content):return exclusive_bytes(OUT/name,content.encode("utf8"))
def ratio(x):
    return "—" if x["value"] is None else f'{100*x["value"]:.2f}%'
def ci(x,scale=100):
    if x["ci95"] is None:return "—"
    return f'{scale*x["ci95"][0]:.1f} 至 {scale*x["ci95"][1]:.1f}' if scale==100 else f'{x["ci95"][0]:.3f} 至 {x["ci95"][1]:.3f}'
def figure_setup():
    os.environ.update(MPLCONFIGDIR=str(OUT/"figure_cache"),MPLBACKEND="Agg",MPL_IGNORE_SYSTEM_FONTS="1")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.5,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"axes.axisbelow":True})
    return plt
def savefig(fig,name,plt):
    for ext in ["pdf","png"]:
        f=OUT/"figures"/(name+"."+ext);need(not f.exists(),"figure already exists")
        fig.savefig(f,dpi=210,bbox_inches="tight",facecolor="white")
    plt.close(fig)
def figures(group,pairs,ranking):
    import numpy as np
    import matplotlib.ticker as mt
    plt=figure_setup()
    order=["depth_only","shift_only","missing_only","block_only","all"]
    def panel(ax,data,metric,title,ylabel):
        for m,off,symbol in zip(METHODS,[-.19,0,.19],["o","s","^"]):
            for i,c in enumerate(order):
                v=data[c,m][metric];y=v["value"]
                if y is None:continue
                band=v["ci95"];xx=i+off
                if band:ax.vlines(xx,band[0],band[1],color=COLORS[m],linewidth=1.4,alpha=.8)
                ax.plot(xx,y,marker=symbol,linestyle="none",color=COLORS[m],markeredgecolor="white",markeredgewidth=.4,markersize=6,label=m if i==0 else None)
        ax.axvline(3.5,ls=":",lw=.9,color="#a0aab2")
        ax.set_xticks(range(5),["Depth\nnoise","Registration\nshift","Missing\ndepth","Rectangular\nmask","Pooled"])
        ax.set_title(title,loc="left",fontsize=11,fontweight="bold",pad=9);ax.set_ylabel(ylabel)
        ax.set_ylim(-.035,1.035);ax.yaxis.set_major_formatter(mt.PercentFormatter(1,decimals=0));ax.grid(axis="y",alpha=.18)
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.8),layout="constrained")
    panel(axes[0],group,"grasp_success","A   Grasp success","All assigned attempts")
    panel(axes[1],group,"static_admission_all","B   Static admission","All assigned attempts")
    axes[0].legend(frameon=False,loc="upper left",ncol=3,handletextpad=.15,columnspacing=.9)
    savefig(fig,"Fig1_execution",plt)
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.8),layout="constrained")
    panel(axes[0],ranking,"pre_E_top1_change","A   Top-1 change relative to clean","Share of assigned units")
    panel(axes[1],ranking,"top3_jaccard","B   Top-3 overlap with clean","Mean Jaccard overlap")
    axes[1].yaxis.set_major_formatter(mt.FormatStrFormatter("%.1f"))
    axes[1].legend(frameon=False,ncol=3,loc="upper right",handletextpad=.15,columnspacing=.9)
    savefig(fig,"Fig2_ranking_stability",plt)
    fig,axes=plt.subplots(1,3,figsize=(10.4,3.65),sharey=True,layout="constrained")
    contrasts=[("A","Raw"),("C0","Raw"),("C0","A")]
    bound=max(5,max(abs(t*100) for r in pairs.values() for t in (r["difference"]["ci95"] or [r["difference"]["value"]])))
    lim=max(5,5*np.ceil((bound+1)/5))
    for i,(a,b) in enumerate(contrasts):
        ax=axes[i];ax.axvline(0,color="#73808a",ls="--",lw=1)
        for j,c in enumerate(order):
            x=pairs[c,a,b]["difference"];val=100*x["value"];col=COLORS[a] if c!="all" else "#213548"
            if x["ci95"]:
                lo,hi=(100*k for k in x["ci95"]);ax.hlines(j,lo,hi,color=col,lw=1.5);ax.plot([lo,hi],[j,j],ls="none",marker="|",color=col)
            ax.plot(val,j,"D" if c=="all" else "o",color=col,ms=5)
        ax.axhline(3.5,color="#cad1d4",lw=.8);ax.set_xlim(-lim,lim);ax.grid(axis="x",alpha=.15);ax.set_title(a+" − "+b,fontweight="bold",fontsize=11)
        ax.set_xlabel("Percentage points")
    axes[0].set_yticks(range(5),[ENG[c] for c in order]);axes[0].invert_yaxis()
    savefig(fig,"Fig3_paired_differences",plt)

def main(text_only=False):
    summary=read(OUT/"statistics/execution240.json")
    ranks=read(OUT/"statistics/rank_stability.json")["summaries"]
    data=read(OUT/"statistics/paired_effects.json")["groups"]
    book=read(OUT/"ledger240.json");window=read(OUT/"admission/terminal_window_verification.json")
    group={(r["condition"],r["method"]):r for r in summary["groups"]}
    pairs={(r["condition"],r["first"],r["second"]):r for r in data}
    ranking={(r["condition"],r["method"]):r for r in ranks if r["population"]=="execution_sample"}
    if not text_only:
        write("statistics/outcome_dictionary.json",EXACT)
    capture=[];flatpair=[];scoretable=[];detail=[]
    for c in (*FACTORS,"all"):
        for m in METHODS:
            r=group[c,m];ci1=r["grasp_success"]["ci95"];ci2=r["static_admission_all"]["ci95"];ci3=r["conditional_success"]["ci95"]
            capture.append(dict(condition=c,method=m,n=r["n"],base_scenes=r["base_scenes"],success=r["success"],rate=r["grasp_success"]["value"],
             lower=ci1[0] if ci1 else None,upper=ci1[1] if ci1 else None,admitted=r["static_admitted"],admission_rate=r["static_admission_all"]["value"],
             admission_lower=ci2[0] if ci2 else None,admission_upper=ci2[1] if ci2 else None,dynamics=r["entered_dynamics"],conditional_success=r["conditional_success"]["value"],
             conditional_lower=ci3[0] if ci3 else None,conditional_upper=ci3[1] if ci3 else None))
            detail.append(dict(condition=c,method=m,n=r["n"],**r["outcomes"]))
        for a,b in (("A","Raw"),("C0","Raw"),("C0","A")):
            r=pairs[c,a,b];diff=r["difference"]
            flatpair.append(dict(condition=c,contrast=a+"-"+b,n=r["n"],clusters=r["scene_clusters"],first_only=r["first_only_success"],second_only=r["second_only_success"],
             both_success=r["both_success"],both_failure=r["both_failure"],different_candidates=r["different_candidates"],risk_difference=diff["value"],
             lower=diff["ci95"][0] if diff["ci95"] else None,upper=diff["ci95"][1] if diff["ci95"] else None,
             first_only_success_rate=r["first_only_success_rate"]["value"],first_only_success_lower=r["first_only_success_rate"]["ci95"][0],
             first_only_success_upper=r["first_only_success_rate"]["ci95"][1],first_only_failure_rate=r["first_only_failure_rate"]["value"],
             first_only_failure_lower=r["first_only_failure_rate"]["ci95"][0],first_only_failure_upper=r["first_only_failure_rate"]["ci95"][1]))
    for r in ranks:
        item={k:r[k] for k in ("population","condition","method","units","distinct_scenes")}
        for short,long in (("top1","pre_E_top1_change"),("locked_top1","post_E_top1_change"),("jaccard","top3_jaccard"),("kendall_tau_b","kendall_tau_b")):
            v=r[long];item[short]=v["value"];item[short+"_valid"]=int(v["denominator"])
            item[short+"_lower"]=v["ci95"][0] if v["ci95"] else None;item[short+"_upper"]=v["ci95"][1] if v["ci95"] else None
        scoretable.append(item)
    if not text_only:
        for name,rows in [("table1_execution",capture),("table2_paired",flatpair),("table3_ranking",scoretable),("table4_outcomes",detail)]:
            exclusive_bytes(OUT/("statistics/"+name+".csv"),csv_bytes(rows))
        figures(group,pairs,ranking)
    a=["# V1 单因素鲁棒性扩充实验", "", "《基于RGB-D实例分割与几何不确定性建模的六自由度机器人抓取选择研究》","",
       "## 样本与执行","",
       f"120 个 V1 基础场景各有四种单因素观测。对每种干预，只有至少一种方法相对同场景 clean 的筛选前 Top-1 几何身份或有无发生变化时，单元才进入抽样框。深度噪声、配准偏移、随机缺深、矩形遮挡的入围单元依次为 119、102、108、105。按原单元哈希顺序，各取前 20 个，共 80 个观测单元，覆盖 {book['unique_base_scenes']} 个不同基础场景。",
       "", "每个条件的前 6 个单元与上次封存逐项一致。72 份历史结果直接沿用，另外 56 个单元的 168 份方法意图各分派一次。每组在每个条件中的分母为 20，四条件合并为 80。全部汇总均限于这个预选子集。",
       "", "原始 RGB-D、条件种子、P3/P5 输出、候选池、旧 E、三组排序与执行参数沿用 V1。四种干预依次采用 5 mm 标准差的轴向高斯噪声、网络网格 +3 像素水平错位、15% 随机深度缺失及 120×160 像素矩形遮挡。方法锁定各自首个满足旧 E 的候选，再经过独立的完整静态路径检查。",
       "", "## 全部尝试与动力学","", "| 干预 | 方法 | 成功/全部 | 成功率（95% 区间） | 静态准入/全部 | 静态准入率 | 动力学次数 | 动力学成功率（95% 区间） |",
       "|---|---|---:|---:|---:|---:|---:|---:|"]
    for c in (*FACTORS,"all"):
        for m in METHODS:
            r=group[c,m]
            a.append(f'| {LABELS[c]} | {m} | {r["success"]}/{r["n"]} | {ratio(r["grasp_success"])}（{ci(r["grasp_success"])}） | {r["static_admitted"]}/{r["n"]} | {ratio(r["static_admission_all"])} | {r["entered_dynamics"]} | {ratio(r["conditional_success"])}（{ci(r["conditional_success"])}） |')
    a+=["", f'合计 {book["physics_total"]} 个试次进入动力学，其中本轮新增 {book["physics_new"]} 个。原生返回物理步累计 {book["native_returns_total"]:,} 个，本轮新增 {book["native_returns_new"]:,} 个。对完整执行的 {window["full_windows"]} 个试次，以末 480 步的物体高度、双侧有效接触、台面总支撑和滑落字段复核了终判。',
       "", "## 相对 clean 的排序稳定性","", "表中为实际抽取的 80 个观测单元。Top-1 对应旧 E 筛选前的最高分候选。各干预独立生成候选池，因此跨观测的首选变化同时反映候选生成和排序两步。采用冻结的 NMS 几何邻域及一对一匹配建立跨观测候选对应。两侧 Top-3 都为空时 Jaccard 缺失。至少两个匹配候选且分数顺序可比较时才计算 Kendall τ-b。括号给出 τ-b 的有效单元数。全部 120 场景的结果及各区间见表格数据。",
       "", "| 干预 | 方法 | Top-1 变化/全部 | 变化率 | 平均 Top-3 Jaccard | 平均 Kendall τ-b（有效单元数） |","|---|---|---:|---:|---:|---:|"]
    for c in (*FACTORS,"all"):
        for m in METHODS:
            r=ranking[c,m];v=r["pre_E_top1_change"];t=r["kendall_tau_b"];j=r["top3_jaccard"]
            jt="—" if j["value"] is None else f'{j["value"]:.3f}'
            tt="—" if t["value"] is None else f'{t["value"]:.3f}'
            a.append(f'| {LABELS[c]} | {m} | {int(v["numerator"])}/{int(v["denominator"])} | {ratio(v)} | {jt} | {tt}（{int(t["denominator"])}） |')
    a+=["","## 配对成功率差","", "同一单元内比较前者减后者。独有成功表示前者成功、后者失败；独有失败表示前者失败、后者成功。分母始终保留该层全部配对单元。",
       "", "| 干预 | 配对比较 | 配对数 | 独有成功 | 独有失败 | 差值（百分点） | 95% 区间（百分点） |","|---|---|---:|---:|---:|---:|---:|"]
    for r in flatpair:
        a.append(f'| {LABELS[r["condition"]]} | {r["contrast"].replace("-"," − ")} | {r["n"]} | {r["first_only"]} | {r["second_only"]} | {100*r["risk_difference"]:+.1f} | {100*r["lower"]:.1f} 至 {100*r["upper"]:.1f} |')
    a+=["","95% 区间使用预登记随机种子 1262275604，按基础场景有放回重采样 20,000 次。每次保留同一基础场景的全部已抽干预和方法。比率用重采样中的计数之比。表 2 的 CSV 同时列出前者独有成功与独有失败比例的聚类区间。有效场景不足两个时区间留空。没有不一致配对时，经验重采样区间可能退化到一个点，小样本仍保留外推不确定性。",
       "", "## 失败去向","", "| 互斥终点 | Raw | A | C0 | 合计 |","|---|---:|---:|---:|---:|"]
    for k,name in EXACT.items():
        vals=[group["all",m]["outcomes"][k] for m in METHODS]
        a.append("| "+" | ".join([name,*map(str,vals),str(sum(vals))])+" |")
    a+=["", "共享候选池为空与非空候选池经旧 E 筛选后为空分列，两项相加即为没有旧 E 候选的试次总数。其他静态拒绝包括在冻结目标几何上发现的接近、碰撞或路径违规。动力学禁止碰撞只计入实际物理执行中的对应终点。分层的完整去向见表 4，逐试次标签和首败保存在运行账本。","", "## 解释",""]
    pooled=[pairs["all",x,y]["difference"] for x,y in (("A","Raw"),("C0","Raw"),("C0","A"))]
    if all(r["ci95"][0]<=0<=r["ci95"][1] for r in pooled):
        a+=["本次固定样本未观察到明确执行收益。"]
    else:
        a+=["各比较的样本方向及不确定性见配对效应表。估计对象为预先抽中的 Top-1 变化单元。"]
    a+=["","该样本特意富集首选候选发生变化的单元。所得执行率只描述该抽样条件，不能外推到全部 120 个基础场景。条件成功率另受静态筛选影响。R6 正式试次与本补充单独统计。",
       "", "## 图注","",
       "图 1　四种干预及合并样本的全部尝试成功率与静态准入率。每种方法在各干预分派 20 次，合并 80 次。点为观测比例，线为基础场景聚类 95% 区间。",
       "", "图 2　抽中单元相对同场景 clean 的筛选前 Top-1 变化率与 Top-3 Jaccard。候选对应采用固定几何邻域和一对一匹配。区间按基础场景聚类。",
       "", "图 3　三个方法对的配对成功率差，以百分点表示。正值表示前者成功更多，虚线标示零，菱形对应合并估计。横线为基础场景聚类 95% 区间。",
       "", "## 复核与归档","",
       "对封存来源、1201 份允许读取的抽样输入、历史 72 份意图及结果链和运行目录进行独立核验。初次包装检查将三案例冷启动后的登记文件与安装前归档作了比较。依据旧回执，从原件逐字节重建了三个冷启动意图及其总清单，确认差异仅来自当时登记的回执哈希。原正式文件、模型、物理参数和方法源码均通过核验。新试次分派前已封存 240 份意图。",
       "", "完整账本见 ledger240.json，原始静态和动力学回执的大文件留在本地。交付包附代码差异、证据位置及 SHA256 清单。"]
    name="V1单因素扩充实验报告.md"
    content="\n".join(a)+"\n"
    if text_only:
        # This option refreshes only the Markdown presentation of the sealed counts.
        staged=OUT/(name+".new")
        exclusive_bytes(staged,content.encode("utf8"))
        staged.replace(OUT/name)
    else:
        textfile(name,content)
if __name__=="__main__":main(text_only=sys.argv[1:]==["--text-only"])
