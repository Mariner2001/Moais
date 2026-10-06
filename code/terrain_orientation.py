"""Orientation-slope association (Table 1 of Lipo & Hunt) and comparison of its margins with road terrain."""
import json
from common import *
from scipy import stats


def shares(df, thr):
    d = df.Distance.values; e = df.Elevation.values; o = np.argsort(d); d, e = d[o], e[o]
    s = 100 * np.diff(e) / np.diff(d); w = np.diff(d)
    dn = s < -thr; up = s > thr; fl = ~(dn | up)
    return np.array([w[dn].sum(), w[fl].sum(), w[up].sum()]) / w.sum()


def run():
    sl = pd.read_csv(dpath('road_slope_profiles.csv')); t1 = pd.read_csv(dpath('table1_orientation_counts.csv')).set_index('position')
    obs = t1[['downhill', 'flat', 'uphill']].sum().values.astype(float)
    rows = []
    for road in ['raraku_southcoast', 'southcoast']:
        for thr in [0.5, 1.0, 2.0]:
            sh = shares(sl[sl.road == road], thr); ex = obs.sum() * sh; chi2 = float(((obs - ex) ** 2 / ex).sum())
            rows.append(dict(road=road, thr=thr, desc=sh[0], flat=sh[1], asc=sh[2], e_desc=ex[0], e_flat=ex[1], e_asc=ex[2], chi2=chi2, p=stats.chi2.sf(chi2, 2)))
    df = pd.DataFrame(rows)
    # flat and sloping sections separately: few statues are classified as lying on flat ground
    n_slope = obs[0] + obs[2]
    df['p_flat_le_obs'] = [stats.binom.cdf(int(obs[1]), int(obs.sum()), f) for f in df.flat]
    df['asc_share_sloping'] = df.asc / (df.asc + df.desc)
    df['e_asc_sloping'] = n_slope * df.asc_share_sloping
    df['p_binom_sloping'] = [stats.binomtest(int(obs[2]), int(n_slope), q).pvalue for q in df.asc_share_sloping]
    df.to_csv(rpath('terrain_chi2.csv'), index=False)
    # terrain shares stated by Lipo & Hunt (2025, Fig. 6A) for the Rano Raraku-south coast road
    pub = np.array([0.45, 0.34, 0.21]); ex = obs.sum() * pub; chi_pub = float(((obs - ex) ** 2 / ex).sum())
    q = pub[2] / (pub[0] + pub[2])
    pubres = dict(shares=pub.tolist(), expected=ex.tolist(), chi2=chi_pub, p=float(stats.chi2.sf(chi_pub, 2)),
                  asc_share_sloping=float(q), e_asc_sloping=float(n_slope * q), p_binom_sloping=float(stats.binomtest(int(obs[2]), int(n_slope), q).pvalue))
    json.dump(pubres, open(rpath('terrain_published_shares.json'), 'w'), indent=1)
    tab = t1[['downhill', 'flat', 'uphill']].values
    chi2, p, dof, _ = stats.chi2_contingency(tab, correction=False); V = np.sqrt(chi2 / (tab.sum() * (min(tab.shape) - 1)))
    fd, fu = t1.loc['face_down'], t1.loc['face_up']
    t22 = np.array([[fd.downhill, fd.uphill], [fu.downhill, fu.uphill]]); orr, pf = stats.fisher_exact(t22)
    pdn = fd.downhill / (fd.downhill + fu.downhill); pup = fd.uphill / (fd.uphill + fu.uphill)
    lg = lambda q: np.log(q / (1 - q))
    res = dict(observed=obs.tolist(), chi2_assoc=chi2, p_assoc=p, dof=int(dof), cramer_V=V, OR_2x2=orr, p_fisher_2x2=pf,
               p_fd_down=pdn, p_fd_up=pup, logit_a=(lg(pdn) + lg(pup)) / 2, logit_b=(lg(pdn) - lg(pup)) / 2)
    json.dump({k: (float(v) if not isinstance(v, list) else v) for k, v in res.items()}, open(rpath('orientation.json'), 'w'), indent=1)
    print(df.round(3).to_string(index=False)); print(res)
