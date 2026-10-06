"""Transport 'survival': road moai = failures, delivered ahu moai = right-censored observations."""
import json
from common import *

EDGES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 16.0])


def censor_set(db, drop_fragments=False, exclude=()):
    """Delivered statues: ahu records of Rano Raraku tuff not located at Rano Raraku itself.
    Records already counted as failures (the ahu statue at Ahu Tongariki added to D84) are excluded."""
    a = db[(db.LOCATION_TYPE == 'AHU') & db.dist_km.notna() & (db.MATERIAL_TYPE == 'Raraku Tuff')]
    a = a[~a.LOCATION_NAME.astype(str).str.strip().str.upper().eq('RANO RARAKU')]
    a = a[~a.OBJECTID.isin(list(exclude))]
    if drop_fragments:
        a = a[a.STATE_FOUND.astype(str) != 'Fragment']
    return a.dist_km.values


def binned_hazard(ev, ce, edges=EDGES):
    allx = np.r_[ev, ce]; out = []
    for a, b in zip(edges[:-1], edges[1:]):
        d = np.sum((ev >= a) & (ev < b))
        expo = np.clip(np.minimum(allx, b) - a, 0, None).sum()   # statue-km travelled inside [a, b)
        out.append((a, b, d, expo, d / expo if expo > 0 else np.nan))
    return pd.DataFrame(out, columns=['a', 'b', 'events', 'statue_km', 'hazard'])


def km(ev, ce):
    t = np.r_[ev, ce]; e = np.r_[np.ones(len(ev)), np.zeros(len(ce))]
    o = np.lexsort((-e, t)); t, e = t[o], e[o]; n = len(t); S = 1.0; ts = [0.0]; Ss = [1.0]
    for i in range(n):
        if e[i] == 1:
            S *= 1 - 1 / (n - i); ts.append(t[i]); Ss.append(S)
    return np.array(ts), np.array(Ss)


def run(B=2000):
    rng = np.random.default_rng(7)
    ds, m, db = datasets(); hz, summ, kms = [], [], []
    for key in ['D84', 'D62', 'D56']:
        excl = m.OBJECTID.dropna().astype(int).tolist() if key == 'D84' else []
        ce = censor_set(db, exclude=excl); ce2 = censor_set(db, True, exclude=excl)
        ev = ds[key]; h = binned_hazard(ev, ce)
        H = np.array([binned_hazard(rng.choice(ev, len(ev)), rng.choice(ce, len(ce))).hazard.values for _ in range(B)])
        h['lo'] = np.nanpercentile(H, 2.5, 0); h['hi'] = np.nanpercentile(H, 97.5, 0); h['dataset'] = key
        hz.append(h)
        t, S = km(ev, ce); kms.append(pd.DataFrame(dict(dataset=key, t=t, S=S)))
        near, far = h[h.b <= 2], h[h.a >= 2]
        hn = near.events.sum() / near.statue_km.sum(); hf = far.events.sum() / far.statue_km.sum()
        h2 = binned_hazard(ev, ce2); n2, f2 = h2[h2.b <= 2], h2[h2.a >= 2]
        summ.append(dict(dataset=key, n_fail=len(ev), n_cens=len(ce), frac_fail=len(ev) / (len(ev) + len(ce)),
                         S2=S[t <= 2][-1], S10=S[t <= 10][-1], h_near=hn, h_far=hf, ratio=hn / hf, n_cens_fragfree=len(ce2),
                         h_near_ff=n2.events.sum() / n2.statue_km.sum(), h_far_ff=f2.events.sum() / f2.statue_km.sum()))
    pd.concat(hz).to_csv(rpath('hazard_bins.csv'), index=False)
    pd.concat(kms).to_csv(rpath('km_curves.csv'), index=False)
    sdf = pd.DataFrame(summ); sdf.to_csv(rpath('survival_summary.csv'), index=False)
    ce = censor_set(db)
    extra = dict(n_cens=len(ce), median_cens=float(np.median(ce)), mean_cens=float(ce.mean()),
                 p_deliver_median_if_const_0p5=float(np.exp(-0.5 * np.median(ce))), p_deliver_10km_if_const_0p5=float(np.exp(-5)))
    json.dump(extra, open(rpath('survival_extra.json'), 'w'), indent=1)
    print(sdf.round(4).to_string(index=False)); print(extra)
