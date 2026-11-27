"""Fixed six-cell selective correspondence rule; no fitted parameters or I/O.

The old global affine map is reused on local peaks as an uncalibrated frozen
mapping. A keep action can be either an identified-zero prediction or unknown;
these states are never conflated. No reference depth/warp/GT enters prediction.
"""
import numpy as np
from src import correspondence_v40 as old


def supported(mask):
    cells=np.flatnonzero(mask)
    return bool(len(cells)>=4 and len(set((cells//3).tolist()))==2 and len(set((cells%3).tolist()))>=2)


def decide(local_uv,local_usable,global_uv,global_usable):
    local=np.asarray(local_uv,np.int64); valid=np.asarray(local_usable,bool); global_uv=np.asarray(global_uv,np.int64)
    old.need(local.shape==(6,2) and valid.shape==(6,) and global_uv.shape==(2,),'Invalid selective-decision shapes')
    zero=valid & np.all(local==0,axis=1)
    move=valid & np.any(local!=0,axis=1) & (np.max(np.abs(local-global_uv),axis=1)<=1) & bool(np.any(global_uv!=0))
    old.need(not np.any(zero & move),'Zero/move supporting cells must be disjoint')
    z=bool(global_usable and supported(zero)); m=bool(global_usable and supported(move))
    old.need(not (z and m),'Two disjoint sets cannot both have four of six cells')
    if z:
        status='supported_zero'; action=[0,0]; usable=True; reason='none'
    elif m:
        status='supported_move'; action=global_uv.tolist(); usable=True; reason='none'
    else:
        status='unknown_abstain'; action=[0,0]; usable=False
        reason='global_unusable' if not global_usable else ('fewer_than_four_usable_cells' if valid.sum()<4 else 'no_four_cell_consensus')
    return dict(status=status,action_uv=action,usable=usable,unknown_reason=reason,
        zero_support_cells=np.flatnonzero(zero).tolist(),move_support_cells=np.flatnonzero(move).tolist(),
        zero_supported=z,move_supported=m,valid_cells=int(valid.sum()),
        original_xyz_overwritten=False,zero_status_is_prediction_not_truth=True)


def estimate_local(rgb,observed_depth,observed_valid,model,global_uv,global_usable):
    idx,rx,ry,rw,zx,zy,zw=old.edge_arrays(rgb,observed_depth,observed_valid)
    h,w=observed_depth.shape; v,u=np.divmod(idx,w)
    cell_id=(u*3//w)+3*(v*2//h)
    rows=[]; raw=[]; usability=[]
    for cell in range(6):
        selected=idx[cell_id==cell]; vv,uu=np.divmod(selected,w)
        scores=np.zeros(81,np.float64)
        for j,(dx,dy) in enumerate(old.CANDIDATES):
            xx,yy=uu-dx,vv-dy; inside=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
            if inside.any():
                ii=selected[inside]
                dot=np.abs(zx.ravel()[ii]*rx[yy[inside],xx[inside]]+zy.ravel()[ii]*ry[yy[inside],xx[inside]])
                scores[j]=np.sum(zw.ravel()[ii]*rw[yy[inside],xx[inside]]*dot)/len(selected)
        ordered=np.sort(scores); gap=float(ordered[-1]-ordered[-2]); best=int(scores.argmax())
        ok=len(selected)>=32 and gap>1e-12
        peak=old.CANDIDATES[best].tolist() if ok else [0,0]
        rows.append(dict(cell=cell,row=cell//3,column=cell%3,edge_count=len(selected),scores=scores.tolist(),
            raw_uv=peak,usable=bool(ok),peak_margin=gap,best_score=float(scores[best]),
            rejection='none' if ok else ('fewer_than32_edges' if len(selected)<32 else 'nonunique_peak')))
        raw.append(peak); usability.append(ok)
    continuous,corrected=old.predict(model,np.asarray(raw),np.asarray(usability))
    for i,row in enumerate(rows):
        row['continuous_uv']=continuous[i].tolist(); row['corrected_uv']=corrected[i].tolist()
    return dict(cells=rows,edge_count=len(idx),edge_indices=idx,
        decision=decide(corrected,usability,global_uv,global_usable),global_corrected_uv=np.asarray(global_uv).tolist(),
        candidate_scores_written=486,nonempty_candidate_dot_products=81*sum(r['edge_count']>0 for r in rows))
