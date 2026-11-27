"""Standalone development figures from frozen accepted P5 results."""
import sys,json
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p5_common_v23r1 import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def full():
    need(read(BASE/'evaluation/readonly.json')['passed'],'Complete readonly acceptance required')
    out=BASE/'figures';out.mkdir(parents=True,exist_ok=True)
    if (out/'receipt.json').exists():need(all(stat(ROOT/p)==v for p,v in read(out/'receipt.json')['files'].items()),'Figure changed');return
    conditions=['clean','noise5','missing15','shift3','joint'];colors=['#436A9A','#C36B30'];fig,axes=plt.subplots(2,2,figsize=(12,8.2),constrained_layout=True);x=np.arange(5)
    for j,arm in enumerate(['fixed','quality']):
        rows=[read(BASE/f'evaluation/sets/core_{arm}_learned_{d}.json') for d in conditions]
        axes[0,0].bar(x+(j-.5)*.35,[r['scene_macro_AP'] for r in rows],width=.35,color=colors[j],label=arm)
        axes[0,1].plot(x,[r['scene_macro_f1'] for r in rows],'-o',color=colors[j],label=arm)
    for ax,ylabel in [(axes[0,0],'Scene-macro mask AP'),(axes[0,1],'Scene-macro instance F1 @ IoU .50')]:ax.set_xticks(x,conditions);ax.set_ylabel(ylabel);ax.set_ylim(bottom=0);ax.legend();ax.grid(axis='y',alpha=.2)
    dec=read(BASE/'evaluation/decision.json');g=dec['by_scene_degraded_gain'];axes[1,0].bar(list(g),[100*v for v in g.values()],color=[colors[1] if v>0 else colors[0] for v in g.values()]);axes[1,0].axhline(0,color='black',lw=.8);axes[1,0].set_xlabel('Previously exposed development scene');axes[1,0].set_ylabel('Mean degraded AP difference (percentage points)')
    ab=['learned','unit','mean'];v=[read(BASE/('evaluation/sets/'+('core_quality_learned_joint' if a=='learned' else 'ablation_quality_'+a+'_joint')+'.json'))['scene_macro_AP'] for a in ab];axes[1,1].bar(ab,v,color=[colors[1],'#777777','#B2B2B2']);axes[1,1].set_ylabel('Joint-degradation mask AP');axes[1,1].set_title('Same quality-trained checkpoint; gate ablation')
    axes[0,0].set_ylim(0,.28);axes[0,1].set_ylim(0,.7);axes[1,1].set_ylim(0,.25)
    fig.suptitle('P5: depth quality and spatial gating — development evidence',fontsize=15);fig.savefig(out/'p5_development_summary.png',dpi=170);fig.savefig(out/'p5_development_summary.pdf');plt.close(fig)
    graded=['noise1','noise3','missing5','missing30','shift1','shift5','block'];fig,ax=plt.subplots(figsize=(9,4.4),constrained_layout=True)
    for j,arm in enumerate(['fixed','quality']):
        vals=[read(BASE/f'evaluation/sets/graded_{arm}_learned_{d}.json')['scene_macro_f1'] for d in graded];ax.plot(range(7),vals,'-o',label=arm,color=colors[j])
    ax.set_xticks(range(7),graded);ax.set_ylim(0,.7);ax.set_ylabel('Scene-macro instance F1');ax.legend();ax.set_title('Graded perturbations: fixed 20-frame development subset (no full-set AP)');ax.grid(axis='y',alpha=.2);fig.savefig(out/'p5_graded_subset.png',dpi=170);fig.savefig(out/'p5_graded_subset.pdf');plt.close(fig)
    once(out/'receipt.json',dict(files={p.relative_to(ROOT).as_posix():stat(p) for p in out.iterdir() if p.is_file()},source_sha256=sha(Path(__file__)),new_model_calls=0,scope='Descriptive development figures; no independent test or statistical significance claim'))
def run():
    need(read(LOG/'screen_readonly.json')['passed'],'Screen readonly required')
    need(stat(Path(__file__))==read(LOG/'delivery_freeze.json')['protected']['scripts/p5_figures_v23r1.py'],'Figure source changed')
    if not (BASE/'finite_stop.json').exists():full();return
    out=BASE/'figures';out.mkdir(parents=True,exist_ok=True);conditions=['clean','noise5','missing15','shift3','joint'];x=np.arange(5);fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    values={a:[read(BASE/f'screen/screen_{a}_learned_{d}.json')['scene_macro_f1'] for d in conditions] for a in ['fixed','quality']}
    for j,a in enumerate(['fixed','quality']):axes[0].bar(x+(j-.5)*.35,values[a],.35,label=a)
    axes[0].set_xticks(x,conditions);axes[0].set_ylim(0,.7);axes[0].set_ylabel('Scene-macro instance F1 @ IoU .50');axes[0].legend()
    gain=read(BASE/'first_five_gate.json')['by_condition_F1_gain'];axes[1].bar(list(gain),[100*v for v in gain.values()],color=['#C36B30' if v>0 else '#436A9A' for v in gain.values()]);axes[1].axhline(0,color='black',lw=.8);axes[1].set_ylabel('Gated minus fixed F1 (percentage points)');axes[1].margins(y=.2)
    fig.suptitle('P5 single repair: epoch5 continuation screen, fixed20 development frames\nFull160-frame mask AP was not run; no generalization claim',fontsize=12);fig.savefig(out/'finite_screen.png',dpi=170);fig.savefig(out/'finite_screen.pdf');plt.close(fig)
    once(out/'receipt.json',dict(files={p.relative_to(ROOT).as_posix():stat(p) for p in out.iterdir() if p.is_file()},source_sha256=sha(Path(__file__)),new_model_calls=0,scope='Prespecified finite continuation screen, not full AP evaluation'))
if __name__=='__main__':run()
