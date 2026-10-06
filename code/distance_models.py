"""Likelihood-based comparison of distance-to-quarry distributions (raw distances, optional left truncation)."""
from common import *
from scipy import stats, optimize

B_ISLAND = 15.3   # max straight-line distance of any ahu moai from the quarry centroid (km)
MODELS = ['Exponential', 'Weibull', 'Gamma', 'Lognormal', 'Half-normal', 'Two-exponential mixture', 'Exponential + uniform']


def nll_factory(name, x, t=0.0):
    n = len(x)
    tr = lambda logpdf, logsf_t: -(logpdf.sum() - n * logsf_t)
    if name == 'Exponential':
        return (lambda p: tr(stats.expon.logpdf(x, scale=np.exp(p[0])), stats.expon.logsf(t, scale=np.exp(p[0])))), [np.log(x.mean() - t + 1e-9)], 1
    if name == 'Weibull':
        return (lambda p: tr(stats.weibull_min.logpdf(x, np.exp(p[0]), scale=np.exp(p[1])), stats.weibull_min.logsf(t, np.exp(p[0]), scale=np.exp(p[1])))), [0.0, np.log(x.mean())], 2
    if name == 'Gamma':
        return (lambda p: tr(stats.gamma.logpdf(x, np.exp(p[0]), scale=np.exp(p[1])), stats.gamma.logsf(t, np.exp(p[0]), scale=np.exp(p[1])))), [0.0, np.log(x.mean())], 2
    if name == 'Lognormal':
        return (lambda p: tr(stats.lognorm.logpdf(x, np.exp(p[1]), scale=np.exp(p[0])), stats.lognorm.logsf(t, np.exp(p[1]), scale=np.exp(p[0])))), [np.log(np.median(x)), 0.0], 2
    if name == 'Half-normal':
        return (lambda p: tr(stats.halfnorm.logpdf(x, scale=np.exp(p[0])), stats.halfnorm.logsf(t, scale=np.exp(p[0])))), [np.log(np.sqrt((x ** 2).mean()))], 1
    if name == 'Two-exponential mixture':
        def f(p):
            w = 1 / (1 + np.exp(-p[0])); s1 = np.exp(p[1]); s2 = s1 + np.exp(p[2])
            pdf = w * stats.expon.pdf(x, scale=s1) + (1 - w) * stats.expon.pdf(x, scale=s2)
            sf = w * stats.expon.sf(t, scale=s1) + (1 - w) * stats.expon.sf(t, scale=s2)
            return -(np.log(pdf + 1e-300).sum() - n * np.log(sf))
        return f, [0.0, np.log(0.5), np.log(3.0)], 3
    if name == 'Exponential + uniform':
        def f(p):
            w = 1 / (1 + np.exp(-p[0])); s = np.exp(p[1])
            pdf = w * stats.expon.pdf(x, scale=s) + (1 - w) * stats.uniform.pdf(x, 0, B_ISLAND)
            sf = w * stats.expon.sf(t, scale=s) + (1 - w) * stats.uniform.sf(t, 0, B_ISLAND)
            return -(np.log(pdf + 1e-300).sum() - n * np.log(sf))
        return f, [1.0, np.log(1.5)], 2
    raise ValueError(name)


def fit(name, x, t=0.0):
    f, p0, k = nll_factory(name, x, t)
    best = None
    starts = [np.array(p0)] + [np.array(p0) + np.random.default_rng(i).normal(0, 0.7, len(p0)) for i in range(12)]
    for s in starts:
        try:
            r = optimize.minimize(f, s, method='Nelder-Mead', options=dict(maxiter=40000, xatol=1e-9, fatol=1e-9))
            if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
                best = r
        except Exception:
            pass
    return best.fun, best.x, k


def describe(name, p):
    if name == 'Exponential': return f'mean={np.exp(p[0]):.2f}'
    if name in ('Weibull', 'Gamma'): return f'shape={np.exp(p[0]):.2f}, scale={np.exp(p[1]):.2f}'
    if name == 'Lognormal': return f'median={np.exp(p[0]):.2f}, sigma={np.exp(p[1]):.2f}'
    if name == 'Half-normal': return f'sigma={np.exp(p[0]):.2f}'
    if name == 'Two-exponential mixture':
        w = 1 / (1 + np.exp(-p[0])); s1 = np.exp(p[1]); return f'w={w:.2f}, means={s1:.2f}/{s1 + np.exp(p[2]):.2f}'
    if name == 'Exponential + uniform':
        return f'w_exp={1 / (1 + np.exp(-p[0])):.2f}, mean={np.exp(p[1]):.2f}'


def run():
    ds, m, db = datasets()
    rows = []
    for key, t in [('D84', 0.0), ('D62', 0.0), ('D56', 0.0), ('D84', 0.6), ('D62', 0.6)]:
        x = np.sort(ds[key]); x = x[x > t] if t > 0 else x
        for nm in MODELS:
            nll, p, k = fit(nm, x, t)
            row = dict(dataset=key, trunc=t, n=len(x), model=nm, k=k, nll=nll, AIC=2 * nll + 2 * k, desc=describe(nm, p))
            row.update({f'p{i}': v for i, v in enumerate(p)})
            rows.append(row)
    df = pd.DataFrame(rows)
    g = df.groupby(['dataset', 'trunc'])
    df['dAIC'] = df.AIC - g.AIC.transform('min')
    df['weight'] = np.exp(-0.5 * df.dAIC); df['weight'] = df.weight / df.groupby(['dataset', 'trunc']).weight.transform('sum')
    df.to_csv(rpath('distance_models.csv'), index=False)
    for (key, t), s in df.groupby(['dataset', 'trunc'], sort=False):
        print(f'== {key} trunc={t} n={s.n.iloc[0]}')
        for _, r in s.sort_values('AIC').iterrows():
            print(f'   {r.model:26s} AIC={r.AIC:7.2f} dAIC={r.dAIC:6.2f} w={r.weight:.3f}  {r.desc}')
