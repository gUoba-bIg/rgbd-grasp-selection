"""First-party figures from completed receipts, with cluster intervals and explicit denominators."""
from pathlib import Path
import os
from common_v1 import OUT,read,need
os.environ["MPLCONFIGDIR"]=str(OUT/"plot_runtime")
os.environ["MPL_IGNORE_SYSTEM_FONTS"]="1"
os.environ["MPLBACKEND"]="Agg"
import matplotlib.pyplot as plt
import numpy as np

COLORS={"Raw":"#354e71","A":"#ad5c26","C0":"#217e71","C0_no_instance":"#815589","C0_depth_only":"#4b85a9","C0_warp_only":"#8d733a"}
SHORT={"C0_no_instance":"No instance","C0_depth_only":"Depth only","C0_warp_only":"Warp only"}
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"white","savefig.facecolor":"white","pdf.fonttype":42})
def save(fig,name):
    d=OUT/"figures";d.mkdir(exist_ok=True)
    for suffix in (".png",".pdf"):
        dest=d/(name+suffix);need(not dest.exists(),"existing figure")
        fig.savefig(dest,dpi=200,bbox_inches="tight")
    plt.close(fig)
def historic():
    pts=read(OUT/"analysis/risk_coverage.json")
    fig,axs=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    for row,methods in enumerate([["Raw","A","C0"],["C0_no_instance","C0_depth_only","C0_warp_only"]]):
        for col,field in enumerate(["observed_score","risk_proxy"]):
            ax=axs[row,col]
            for m in methods:
                sel=[q for q in pts if q["method"]==m and q["criterion"]==field and q["threshold"] is not None]
                if not sel:continue
                x=[q["coverage"]["value"] for q in sel];y=[q["empirical_failure_risk"]["value"] for q in sel]
                ax.step(x,y,where="post",color=COLORS[m],lw=1.6,label=SHORT.get(m,m))
                ci=[q["empirical_failure_risk"]["ci95"] or [np.nan,np.nan] for q in sel]
                ax.fill_between(x,[q[0] for q in ci],[q[1] for q in ci],color=COLORS[m],alpha=.08,step="post")
            ax.set_title(("R6, 240 per method" if row==0 else "R7, 80 per ablation")+" · "+("score" if col==0 else "collision proxy"))
            ax.grid(axis="y",alpha=.18);ax.legend(frameon=False,fontsize=8,loc="lower right");ax.set_ylim(-.02,1.03);ax.set_xlim(-.01,1.01)
            if col==0:ax.set_ylabel("Observed failure fraction\namong retained attempts")
            if row==1:ax.set_xlabel("Coverage among all assigned attempts")
    fig.tight_layout();save(fig,"Fig1_historical_risk_coverage")
    rr=read(OUT/"analysis/method_rates.json")
    rr=[r for r in rr if r["stratum"]=="all" and r["population"]!="common_80_each"]
    fig,axs=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
    for ax,group in zip(axs,["R6_240_each","R7_80_each"]):
        ms=[r for r in rr if r["population"]==group]
        x=np.arange(len(ms))
        series=["selected","static_admission","all_attempt_success"]
        for i,(metric,col) in enumerate(zip(series,["#9ba6ae","#5085a0","#1a695e"])):
            vs=np.array([r[metric]["value"] for r in ms]);cis=np.array([r[metric]["ci95"] for r in ms])
            ax.bar(x+(i-1)*.25,vs,width=.23,color=col,label=["V1 selection","Static admission","Grasp success"][i])
            ax.errorbar(x+(i-1)*.25,vs,yerr=[np.maximum(0,vs-cis[:,0]),np.maximum(0,cis[:,1]-vs)],fmt="none",ecolor="#35454b",lw=1,capsize=2)
        ax.set_xticks(x,[SHORT.get(r["method"],r["method"]) for r in ms]);ax.set_ylim(0,1.03)
        ax.set_title("R6 · 240 attempts per method" if group.startswith("R6") else "R7 · 80 attempts per ablation");ax.grid(axis="y",alpha=.17)
    axs[0].set_ylabel("Fraction of all assigned attempts");axs[1].legend(frameon=False,loc="center",bbox_to_anchor=(.54,-.22),ncol=3,fontsize=8)
    fig.tight_layout();save(fig,"Fig2_historical_selection_to_execution")
def prospective():
    rows=read(OUT/"robustness/rank_summary.json");modes=["depth_only","shift_only","missing_only","block_only","joint"]
    fig,axs=plt.subplots(1,2,figsize=(11.6,4),sharex=True,sharey=True)
    for ax,metric,title in zip(axs,["top1_change","top3_jaccard"],["Top-1 pose or presence change","Top-3 Jaccard overlap"]):
        for m,j in zip(["Raw","A","C0"],[-.16,0,.16]):
            arr=[next(r[metric] for r in rows if r["method"]==m and r["condition"]==mode) for mode in modes]
            y=np.array([x["value"] for x in arr]);ci=np.array([x["ci95"] for x in arr])
            ax.errorbar(np.arange(5)+j,y,yerr=[np.maximum(0.,y-ci[:,0]),np.maximum(0.,ci[:,1]-y)],fmt="o",markersize=4,color=COLORS[m],capsize=3,label=m)
        ax.set_title(title);ax.set_xticks(np.arange(5),["Noise","Shift","Missing","Block","Joint"]);ax.set_ylim(-.04,1.04);ax.grid(axis="y",alpha=.18)
    axs[0].set_ylabel("Fraction / average overlap across 120 scenes");axs[1].legend(frameon=False,loc="best");fig.tight_layout();save(fig,"Fig3_six_condition_ranking")
if __name__=="__main__":
    import sys
    {"historic":historic,"prospective":prospective}[sys.argv[1]]()
