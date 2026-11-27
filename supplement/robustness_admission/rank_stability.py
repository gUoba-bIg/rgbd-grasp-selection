"""Compare independent prediction pools without treating per-unit indices as identities."""
from collections import Counter
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau
from grasp_nms import nms_grasp
from common_v1 import METHODS, need, read

def load_unit(unit_dir):
    o=read(unit_dir/"orders.json")
    sel=read(unit_dir/"selection.json")
    with np.load(unit_dir/"pool.npz",allow_pickle=False) as p:
        rows=p["candidate_camera"].copy()
    need(len(rows)==len(sel["eligibility"])==o["pool_count"],"inconsistent V1 pool")
    need(np.isfinite(rows).all(),"nonfinite pose")
    return rows,o["scores"],o["orders_by_pool_index"],sel["selected"]

def pair_matches(old,new):
    if not len(old) or not len(new):
        return {},[]
    keep=np.zeros((len(old),len(new)),bool)
    distances=np.zeros((len(old),len(new)),float)
    receipts=[]
    for i,a in enumerate(old):
        for j,b in enumerate(new):
            dist=float(np.linalg.norm(a[13:16].astype(float)-b[13:16].astype(float)))
            if dist>0.03000001:
                continue
            pair=np.stack((a,b)).copy()
            raw=nms_grasp(pair,.03,np.pi/6)
            if len(raw)==2:continue
            need(len(raw)==1,"unknown two-pose NMS return")
            ra=a[4:13].reshape(3,3).astype(float)
            rb=b[4:13].reshape(3,3).astype(float)
            cosines=[(np.trace(ra.T@rb@d)-1)/2 for d in (np.eye(3),np.diag([1.,-1.,-1.]))]
            angle=float(np.arccos(np.clip(max(cosines),-1.,1.)))
            distances[i,j]=dist/.03+angle/(np.pi/6)
            keep[i,j]=True
            receipts.append(dict(clean_index=i,other_index=j,translation_m=dist,angle_for_tie_break_rad=angle,
                exact_pose_and_dimensions=bool(np.array_equal(a[1:16],b[1:16]))))
    # Each feasible edge costs <10. Dummy assignments cost 1,000, dominating every
    # feasible cost in a Top-20 pool. Unusable edges cost more than dummy slots.
    costs=np.column_stack((np.where(keep,distances,1e6),np.full((len(old),len(old)),1000.)))
    ii,jj=linear_sum_assignment(costs)
    mapping={int(i):int(j) for i,j in zip(ii,jj) if j<len(new) and keep[i,j]}
    for r in receipts:r["assigned"]=mapping.get(r["clean_index"])==r["other_index"]
    return mapping,receipts

def compare(a,b,base_scene,condition):
    A,ascores,aranks,aselected=a;B,bscores,branks,bselected=b
    mp,log=pair_matches(A,B);results=[]
    def status(i,j):
        if i is None or j is None:
            if i is None and j is None:return "both_absent"
            return "presence_changed"
        return "pose_matched" if mp.get(i)==j else "pose_changed"
    for m in METHODS:
        aa,bb=aranks[m],branks[m]
        rankpairs=sorted(mp.items())
        ra={v:i for i,v in enumerate(aa)};rb={v:i for i,v in enumerate(bb)}
        ta=tb=None
        if len(rankpairs)>=2:
            u=[ascores[m][i] for i,j in rankpairs];v=[bscores[m][j] for i,j in rankpairs]
            if len(set(u))>1 and len(set(v))>1:ta=float(kendalltau(u,v).statistic)
            tb=float(kendalltau([ra[i] for i,j in rankpairs],[rb[j] for i,j in rankpairs]).statistic)
        first=aa[0] if aa else None; second=bb[0] if bb else None
        row=dict(scene=base_scene,condition=condition,method=m,clean_pool=len(A),degraded_pool=len(B),
            matched=len(mp),score_tau_b=ta,stable_order_tau=tb,top1=status(first,second),
            top1_exact_pose_and_dimensions=(bool(np.array_equal(A[first,1:16],B[second,1:16])) if first is not None and second is not None else None),
            locked=status(aselected[m],bselected[m]),clean_lock=aselected[m],degraded_lock=bselected[m])
        for k in (1,3,5,10,20):
            x,y=set(aa[:k]),set(bb[:k])
            ct=sum(i in x and j in y for i,j in mp.items())
            union=len(x)+len(y)-ct
            row["top%d_intersection"%k]=ct
            row["top%d_jaccard"%k]=ct/union if union else None
            row["top%d_overlap_coefficient"%k]=ct/min(len(x),len(y)) if min(len(x),len(y)) else None
        results.append(row)
    return results,log

def self_test():
    ident=np.eye(3).ravel()
    def row(x,score=1):
        return [score,.08,.02,.02,*ident,x,0.,.6,-1.]
    a=np.array([row(0),row(.12),row(.24)],dtype=np.float32);b=a[[2,0,1]].copy();b[1,0]=.8
    m,_=pair_matches(a,b);need(m=={0:1,1:2,2:0},"source indices may change across observations")
    need(pair_matches(a[:0],a)==({},[]),"empty identity")
    m,_=pair_matches(a,np.array([row(.6)],dtype=np.float32));need(not m,"far pose")
    m,_=pair_matches(a[:1],np.array([row(.0001),row(.0002)],dtype=np.float32))
    need(m=={0:0},"one-to-one maximum cardinality/minimum distance")
    aa=(a,{k:[0.,0.,0.] for k in METHODS},{k:[0,1,2] for k in METHODS},{k:0 for k in METHODS})
    bb=(b,{k:[0.,0.,0.] for k in METHODS},{k:[1,2,0] for k in METHODS},{k:1 for k in METHODS})
    rr,_=compare(aa,bb,"fixture","near")
    need(all(x["score_tau_b"] is None and x["stable_order_tau"]==1 and x["locked"]=="pose_matched" and x["top3_jaccard"]==1 for x in rr),"ties and permutation")
    return {"tested":["pool permutation","empty","unmatched","one-to-one","all-tie tau","stable order","top-K"],"passed":True}
