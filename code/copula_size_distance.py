"""Rank and copula analysis of statue size vs distance from the quarry (data of Lipo & Hunt 2025, Fig. 13, n = 46).
Convention: U = size rank, V = distance rank. The transport-failure claim ('large statues fail near the quarry')
concerns the corner (U -> 1, V -> 0); its tail coefficient is lambda_10 = lim P(V <= q | U > 1 - q)."""
import json
from common import *
from scipy import stats, optimize
from scipy.integrate import quad
from scipy.stats import hypergeom

rng = np.random.default_rng(13)


def ll_gauss(u, v, r):
    x, y = stats.norm.ppf(u), stats.norm.ppf(v)
    return np.sum(-0.5 * np.log(1 - r ** 2) - (r ** 2 * (x ** 2 + y ** 2) - 2 * r * x * y) / (2 * (1 - r ** 2)))


def ll_frank(u, v, t):
    if abs(t) < 1e-8:
        return 0.0
    num = np.log(abs(t)) + np.log(abs(1 - np.exp(-t))) - t * (u + v)
    den = 2 * np.log(np.abs((1 - np.exp(-t)) - (1 - np.exp(-t * u)) * (1 - np.exp(-t * v))))
    return np.sum(num - den)


def ll_clayton(u, v, t):
    return np.sum(np.log1p(t) - (1 + t) * (np.log(u) + np.log(v)) - (2 + 1 / t) * np.log(u ** (-t) + v ** (-t) - 1))


def ll_gumbel(u, v, t):
    # c = C (uv)^-1 (xy)^(t-1) A^(1/t - 2) (A^(1/t) + t - 1),  x = -ln u, y = -ln v, A = x^t + y^t
    x, y = -np.log(u), -np.log(v); A = x ** t + y ** t
    return np.sum(-A ** (1 / t) - np.log(u) - np.log(v) + (t - 1) * (np.log(x) + np.log(y)) + (1 / t - 2) * np.log(A) + np.log(A ** (1 / t) + t - 1))


ROT = {0: lambda u, v: (u, v), 90: lambda u, v: (1 - u, v), 180: lambda u, v: (1 - u, 1 - v), 270: lambda u, v: (u, 1 - v)}


def frank_tau(t):
    if abs(t) < 1e-8:
        return 0.0
    a = abs(t); D1 = quad(lambda s: s / np.expm1(s), 0, a)[0] / a
    tau = 1 - 4 / a * (1 - D1)
    return tau if t > 0 else -tau


def fit_all(u, v):
    res = [dict(family='Independence', rot='', k=0, ll=0.0, par=np.nan, tau=0.0, lam10=0.0, boundary=False)]
    o = optimize.minimize_scalar(lambda r: -ll_gauss(u, v, r), bounds=(-0.99, 0.99), method='bounded')
    res.append(dict(family='Gaussian', rot='', k=1, ll=-o.fun, par=o.x, tau=2 / np.pi * np.arcsin(o.x), lam10=0.0, boundary=False))
    o = optimize.minimize_scalar(lambda t: -ll_frank(u, v, t), bounds=(-30, 30), method='bounded')
    res.append(dict(family='Frank', rot='', k=1, ll=-o.fun, par=o.x, tau=frank_tau(o.x), lam10=0.0, boundary=False))
    for fam, llf, lo, hi in [('Clayton', ll_clayton, 1e-4, 30), ('Gumbel', ll_gumbel, 1.0001, 20)]:
        for rot in [0, 90, 180, 270]:
            uu, vv = ROT[rot](u, v)
            o = optimize.minimize_scalar(lambda t: -llf(uu, vv, t), bounds=(lo, hi), method='bounded'); t = o.x
            tau = t / (t + 2) if fam == 'Clayton' else 1 - 1 / t
            if rot in (90, 270):
                tau = -tau
            if fam == 'Clayton':
                lam = 2 ** (-1 / t) if rot == 90 else 0.0          # lower-tail corner mapped to (U=1, V=0)
            else:
                lam = 2 - 2 ** (1 / t) if rot == 270 else 0.0      # upper-tail corner mapped to (U=1, V=0)
            bnd = (t < 1e-3) if fam == 'Clayton' else (t < 1.001)
            res.append(dict(family=fam, rot=str(rot), k=1, ll=-o.fun, par=t, tau=tau, lam10=lam, boundary=bnd))
    df = pd.DataFrame(res); df['AIC'] = -2 * df.ll + 2 * df.k
    df['dAIC_indep'] = df.AIC - df.loc[df.family == 'Independence', 'AIC'].values[0]
    df['dAIC_best'] = df.AIC - df.AIC.min(); w = np.exp(-0.5 * df.dAIC_best); df['weight'] = w / w.sum()
    return df


def cvm_indep(u, v, B=9999):
    n = len(u)

    def S(u, v):
        Cn = np.array([np.mean((u <= u[i]) & (v <= v[i])) for i in range(n)])
        return np.sum((Cn - u * v) ** 2)
    s0 = S(u, v); cnt = sum(S(u, rng.permutation(v)) >= s0 for _ in range(B))
    return s0, (cnt + 1) / (B + 1)


def pobs(x):
    return stats.rankdata(x) / (len(x) + 1)


def analyse(dist, size, label, B=9999):
    u, v = pobs(size), pobs(dist); n = len(u)
    tau, pt = stats.kendalltau(size, dist); rho, pr = stats.spearmanr(size, dist); r, pp = stats.pearsonr(dist, size)
    bt = [stats.kendalltau(*(lambda i: (size[i], dist[i]))(rng.integers(0, n, n)))[0] for _ in range(2000)]
    s0, pc = cvm_indep(u, v, B)
    a = int(np.sum((u > 0.75) & (v <= 0.25))); b_ = int(np.sum((u > 0.75) & (v > 0.25)))
    c = int(np.sum((u <= 0.75) & (v <= 0.25))); d = int(np.sum((u <= 0.75) & (v > 0.25)))
    pcorner = stats.fisher_exact([[a, b_], [c, d]], alternative='greater')[1]
    out = dict(sample=label, n=n, r=r, p_r=pp, tau=tau, p_tau=pt, tau_lo=np.percentile(bt, 2.5), tau_hi=np.percentile(bt, 97.5),
               rho=rho, p_rho=pr, cvm=s0, p_cvm=pc, corner_k=a, corner_n=a + b_, corner_p=pcorner)
    df = fit_all(u, v); df.insert(0, 'sample', label)
    print(f"\n== {label}: n={n} r={r:.3f} (p={pp:.3f}) tau={tau:.3f} (p={pt:.3f}; CI {out['tau_lo']:.2f}..{out['tau_hi']:.2f}) "
          f"rho={rho:.3f} (p={pr:.3f}) CvM={s0:.4f} (p={pc:.3f}) corner {a}/{a + b_} (p={pcorner:.3f})")
    print(df[['family', 'rot', 'par', 'tau', 'lam10', 'll', 'dAIC_indep', 'boundary']].round(3).to_string(index=False))
    return out, df


def power(n, taus, B=1500, family='Gaussian'):
    out = []
    for tau in taus:
        hits = 0
        for _ in range(B):
            if family == 'Gaussian':
                rr = np.sin(np.pi * tau / 2); z = rng.multivariate_normal([0, 0], [[1, rr], [rr, 1]], n); a, b = z[:, 0], z[:, 1]
            else:   # Clayton rotated by 90 degrees: tail dependence in the (U=1, V=0) corner
                th = 2 * abs(tau) / (1 - abs(tau)); w1 = rng.uniform(size=n); w2 = rng.uniform(size=n)
                vv = (w1 ** (-th) * (w2 ** (-th / (1 + th)) - 1) + 1) ** (-1 / th)
                a, b = 1 - w1, vv
            hits += stats.kendalltau(a, b)[1] < 0.05
        out.append(hits / B)
    return np.array(out)


def run():
    f = pd.read_csv(dpath('figure13_size_distance_published46.csv'))
    dist = f.distance_from_quarry_km.values; size = f.total_length_cm.values.astype(float)
    db = load_db(); cand = db[db.LOCATION_TYPE.isin(['ROAD', 'QUARRY NOT BEDROCK'])].copy()
    cand['L'] = pd.to_numeric(cand.TOTAL_LENGTH_cm, errors='coerce')
    cls, frag, oid = [], [], []
    for d, L in zip(dist, size):
        c = cand[cand.L == L]
        c = c.iloc[[np.argmin(np.abs(c.dist_km.values - d))]]
        cls.append(c.LOCATION_TYPE.values[0]); frag.append(str(c.NUMBER_OF_FRAGMENTS.values[0])); oid.append(int(c.OBJECTID.values[0]))
    f['class'], f['n_fragments'], f['OBJECTID'] = cls, frag, oid
    f.to_csv(dpath('figure13_annotated.csv'), index=False)
    print('composition:', f['class'].value_counts().to_dict(), f['n_fragments'].value_counts().to_dict())
    r1, d1 = analyse(dist, size, 'all46')
    keep = f['class'].values == 'ROAD'
    r2, d2 = analyse(dist[keep], size[keep], 'road')
    keep2 = keep & (f.n_fragments.values == '1')
    r3, d3 = analyse(dist[keep2], size[keep2], 'road_single', B=4999)
    pd.DataFrame([r1, r2, r3]).to_csv(rpath('rank_tests.csv'), index=False)
    pd.concat([d1, d2, d3]).to_csv(rpath('copula_fits.csv'), index=False)
    big, far, w2, w15 = size > 800, dist > 4.5, dist <= 2, dist < 1.5
    N = len(size)
    hg = dict(n_big=int(big.sum()), n_far=int(far.sum()), overlap=int((big & far).sum()),
              p_no_overlap=float(hypergeom(N, far.sum(), big.sum()).pmf(0)),
              n_within2=int(w2.sum()), big_within2=int((big & w2).sum()), p_big_within2=float(hypergeom(N, w2.sum(), big.sum()).sf((big & w2).sum() - 1)),
              n_within15=int(w15.sum()), big_within15=int((big & w15).sum()), p_big_within15=float(hypergeom(N, w15.sum(), big.sum()).sf((big & w15).sum() - 1)),
              road_n_big=int((big & keep).sum()), road_n_far=int((far & keep).sum()),
              road_p_no_overlap=float(hypergeom(int(keep.sum()), int((far & keep).sum()), int((big & keep).sum())).pmf(0)),
              big_from_qnb=int((big & ~keep).sum()), big_154=bool(154 in f.loc[big, 'OBJECTID'].values))
    json.dump(hg, open(rpath('hypergeom.json'), 'w'), indent=1); print(hg)
    taus = np.array([-0.05, -0.1, -0.15, -0.2, -0.25, -0.3, -0.35, -0.4, -0.5])
    pw = pd.DataFrame(dict(tau=taus, gauss=power(46, taus, 1500, 'Gaussian'), clayton90=power(46, taus, 1500, 'Clayton90')))
    pw.to_csv(rpath('power_n46.csv'), index=False); print(pw.round(3).to_string(index=False))
