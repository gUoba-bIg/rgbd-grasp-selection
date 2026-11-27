"""Static P4 scientific figures from accepted, sealed reports; no recomputation."""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
os.environ['MPLCONFIGDIR']=str(ROOT/'runtime/tmp/p4_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
BASE=ROOT/'results/P4/full_baseline_v22r2'
OUT=BASE/'summary_figures'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    close=read(BASE/'closeout.json');assert close['usable']
    assert not OUT.exists(),'Do not overwrite already delivered scientific figures'
    OUT.mkdir(parents=True)
    selection=read(BASE/'selection.json');metrics=read(BASE/'evaluation/metrics.json');epochs=range(1,selection['last_epoch']+1)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    colors={'rgb':'#3465a4','rgbd':'#c05b1b'}
    points={};sources={}
    for arm in ['rgb','rgbd']:
        reports=[]
        for epoch in epochs:
            path=(ROOT/'results/P4/full_baseline_v22r1' if epoch<=30 else BASE)/f'epochs/{epoch:02d}_{arm}.json';reports.append(read(path));sources[path.relative_to(ROOT).as_posix()]=digest(path)
        points[arm]=dict(monitor_f1=[r['scene_macro_f1'] for r in reports],monitor_fg_iou=[r['scene_macro_foreground_iou'] for r in reports])
        for ax,key in [(axes[0,0],'monitor_f1'),(axes[0,1],'monitor_fg_iou')]:ax.plot(list(epochs),points[arm][key],label=arm.upper(),color=colors[arm],linewidth=1.7)
    for ax in axes[0]:ax.axvline(selection['selected_epoch'],color='#555555',ls='--',lw=1,label='Selected paired epoch');ax.axvspan(30.5,35.5,color='#999999',alpha=.12);ax.set_xlabel('Training epoch (31-35: fixed low-LR repair)');ax.set_ylim(0,1);ax.legend(loc='lower right',fontsize=8);ax.grid(alpha=.2)
    axes[0,0].set_title('Development monitor: instance F1 at IoU 0.5')
    axes[0,1].set_title('Development monitor: foreground IoU')
    scenes=sorted(k for k in metrics['by_scene']['rgb'] if k!='pooled');x=np.arange(len(scenes));width=.36
    for arm,offset in [('rgb',-width/2),('rgbd',width/2)]:
        ap=[metrics['by_scene'][arm][s]['AP'] for s in scenes];points[arm]['scene_ap']=ap
        axes[1,0].bar(x+offset,ap,width,label=arm.upper(),color=colors[arm])
        path=BASE/(arm+'_full_development.json');report=read(path);sources[path.relative_to(ROOT).as_posix()]=digest(path)
        f1=[r['f1'] for r in report['scenes']];points[arm]['full_scene_f1']=f1
        axes[1,1].bar(x+offset,f1,width,label=arm.upper(),color=colors[arm])
    for ax in axes[1]:ax.set_xticks(x,[s[6:] for s in scenes],rotation=45);ax.set_xlabel('Development scene');ax.set_ylim(0,1);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    axes[1,0].set_title('Full development visible-mask AP (COCO .50:.95)')
    axes[1,1].set_title('Full development instance F1 at IoU 0.5')
    fig.suptitle('P4 ordinary RGB/RGBD development baselines | one seed, reused development scenes',fontsize=12)
    for ext in ['png','svg']:fig.savefig(OUT/('p4_development_summary.'+ext),dpi=180)
    plt.close(fig)
    for path in [BASE/'selection.json',BASE/'evaluation/metrics.json',BASE/'closeout.json']:sources[path.relative_to(ROOT).as_posix()]=digest(path)
    payload=dict(source_files=sources,epochs=list(epochs),selected_epoch=selection['selected_epoch'],scenes=scenes,plotted_values=points,scope='Descriptive development results, no independent generalization or method-gain inference',new_model_calls=0,new_training_updates=0,new_metrics_computed=False)
    (OUT/'plot_sources.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps({p.name:digest(p) for p in OUT.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf-8')
    print(str(OUT/'p4_development_summary.png'))
if __name__=='__main__':main()
