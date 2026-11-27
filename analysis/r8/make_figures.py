"""Generate publication figures exclusively from R8's reconstructed tables."""
import os, csv
from pathlib import Path
V=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(V/'qa'/'mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.font_manager import FontProperties
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
plt.rcParams.update({'font.family':font,'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.7,
                     'axes.edgecolor':'#6a7378','xtick.color':'#34404a','ytick.color':'#34404a',
                     'text.color':'#17252c','axes.labelcolor':'#17252c','savefig.facecolor':'white'})
def load(name):
    with (V/'tables'/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
rates=load('rates.csv');pairs=load('paired_effects.csv')
def save(fig,name):
    for ext in ('pdf','png'):
        fig.savefig(V/'figures'/(name+'.'+ext),dpi=320)
    plt.close(fig)
def grid(ax):
    ax.set_axisbelow(True);ax.grid(axis='x',color='#dce2e4',linewidth=.6)
r=[x for x in rates if x['phase']=='R6' and x['stratum']=='all']
fig,axes=plt.subplots(1,2,figsize=(7.05,2.85),gridspec_kw={'width_ratios':[1.05,1.1]})
fig.subplots_adjust(left=.085,right=.97,top=.81,bottom=.25,wspace=.42)
ax=axes[0];grid(ax)
for j,(z,col) in enumerate(zip(r,['#334e68','#137575','#ac521e'])):
    y=2-j;a,l,h=[float(z[k]) for k in ('rate_pct','wilson_low_pct','wilson_high_pct')]
    ax.errorbar(a,y,xerr=[[a-l],[h-a]],fmt='o',ms=6,color=col,capsize=3,lw=1.5)
    ax.text(a,y+.16,f"{int(z['success'])}/240",ha='center',va='bottom',fontsize=8)
ax.set(yticks=[2,1,0],yticklabels=['Raw','A','C0'],ylim=(-.45,2.55),xlim=(0,25.5),xlabel='全部尝试成功率（%）');ax.xaxis.set_major_locator(MultipleLocator(5))
ax.set_title('A   单组估计与 95% Wilson 区间',loc='left',fontsize=10,pad=15)
ax=axes[1];grid(ax);x=np.arange(3)
cat=['资格集为空','静态未获准入','进入动力学'];cols=['#d7dfe3','#88a4b2','#156c71'];left=np.zeros(3)
for lab,col,k in zip(cat,cols,['empty','static_rejected','dynamics_entered']):
    a=np.array([int(z[k]) for z in r]);ax.barh(2-x,a,left=left,height=.50,color=col,label=lab)
    for jj,num in enumerate(a):ax.text(left[jj]+num/2,2-jj,str(num),ha='center',va='center',fontsize=7.8,color='white' if k=='dynamics_entered' else '#132c38')
    left+=a
ax.set(yticks=[2,1,0],yticklabels=['Raw','A','C0'],xlim=(0,240),ylim=(-.45,2.55),xlabel='方法试次数');ax.set_xticks([0,60,120,180,240]);ax.set_title('B   全部分派试次的执行去向',loc='left',fontsize=10,pad=15)
ax.legend(loc='upper center',bbox_to_anchor=(.50,-.27),frameon=False,ncol=3,fontsize=6.8,handlelength=.9,columnspacing=.7)
save(fig,'Fig1_R6_success_and_admission')
fig,ax=plt.subplots(figsize=(7.05,3.40));fig.subplots_adjust(left=.27,right=.77,bottom=.18,top=.91);grid(ax)
ax.axvline(0,ls='--',color='#66757a',lw=1.1)
labels=['C0 − A','A − Raw','C0 − Raw','去实例关联 − C0','仅深度误差 − C0','仅配准误差 − C0']
rows=[z for z in pairs if z['stratum']=='all'];ys=[5.7,4.7,3.7,2.1,1.1,.1]
for y,z,label in zip(ys,rows,labels):
    val,lo,hi=[float(z[k]) for k in ('delta_pp','CI_low_pp','CI_high_pp')]
    color='#146d70' if z['phase']=='R6' else '#a55124'
    ax.errorbar(val,y,xerr=[[val-lo],[hi-val]],fmt='o' if hi!=lo else 'D',markersize=5,capsize=3,
                color=color,mfc=color if hi!=lo else 'white',elinewidth=1.6)
    ax.text(1.04,y,f'{val:+.2f}  [{lo:.2f}, {hi:.2f}]'.replace('+0.00','0.00'),transform=ax.get_yaxis_transform(),fontsize=8,va='center')
ax.axhline(2.9,color='#d7dee0',lw=.8);ax.set(yticks=ys,yticklabels=labels,xlim=(-10,10),ylim=(-.5,6.2),xlabel='全部尝试成功率的配对差（百分点）')
ax.set_xticks([-10,-5,0,5,10]);ax.text(1.04,1.02,'差值  [95% 区间]',transform=ax.transAxes,fontsize=8)
save(fig,'Fig3_paired_differences')
fig,ax=plt.subplots(figsize=(7.05,3.1));fig.subplots_adjust(left=.18,right=.96,bottom=.11,top=.74)
arms=['Raw','A','C0'];strata=['clean','joint','014','017','021']
nums=np.array([[int(next(z['success'] for z in rates if (z['phase'],z['method'],z['stratum'])==('R6',m,s))) for s in strata] for m in arms]);den=[120,120,80,80,80];p=100*nums/np.array(den)
from matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('teal',['#f0f5f5','#b4d7d0','#45958c','#165950'])
ax.imshow(p,cmap=cmap,vmin=0,vmax=35,aspect='auto');ax.set(xticks=np.arange(5),xticklabels=['clean','joint','物体 014','物体 017','物体 021'],yticks=np.arange(3),yticklabels=arms);ax.xaxis.tick_top();ax.tick_params(length=0,pad=7)
for i in range(3):
    for j in range(5):
        ax.text(j,i,f'{nums[i,j]}/{den[j]}  ({p[i,j]:.1f}%)',ha='center',va='center',fontsize=8.7,color='white' if p[i,j]>=22 else '#16363a')
ax.axvline(1.5,color='white',linewidth=5)
for sp in ax.spines.values():sp.set_visible(False)
ax.text(.2,1.24,'观测条件',ha='center',transform=ax.transAxes,fontsize=10)
ax.text(.7,1.24,'物体',ha='center',transform=ax.transAxes,fontsize=10)
save(fig,'Fig2_R6_stratified')
