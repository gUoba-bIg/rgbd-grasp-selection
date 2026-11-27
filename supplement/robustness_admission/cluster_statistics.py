"""Scene-cluster ratio intervals. Undefined denominators stay undefined."""
from functools import lru_cache
import numpy as np
from common_v1 import need

SEED=1262275604
REPS=20000
@lru_cache(maxsize=32)
def weights(scenes):
    n=len(scenes)
    rng=np.random.default_rng(SEED)
    return rng.multinomial(n,np.repeat(1/n,n),size=REPS).astype(np.int16)

def estimate(rows,numerator,denominator,scene="scene"):
    grouped={}
    for r in rows:
        n,d=numerator(r),denominator(r)
        if n is None or d is None:continue
        need(np.isfinite(n) and np.isfinite(d) and d>=0,"invalid cluster counts")
        name=r[scene]
        a,b=grouped.get(name,(0.,0.))
        grouped[name]=(a+n,b+d)
    names=tuple(sorted(grouped))
    if not names:return dict(value=None,ci95=None,clusters=0,denominator=0.,numerator=0.,valid_draws=0)
    a,b=np.array([grouped[k] for k in names]).T
    result=dict(value=float(a.sum()/b.sum()) if b.sum() else None,ci95=None,
                clusters=len(names),informative_clusters=int((b>0).sum()),denominator=float(b.sum()),numerator=float(a.sum()),valid_draws=0)
    if len(names)<2 or (b>0).sum()<2:return result
    w=weights(names)
    den=w@b;valid=den>0
    if valid.any():
        ratios=(w@a)[valid]/den[valid]
        result.update(ci95=np.quantile(ratios,[.025,.975]).tolist(),valid_draws=int(valid.sum()))
    return result
