"""Reproduce the Supplementary S1 statistics of Lipo & Hunt (2025) and quantify the AIC scale artefact."""
import json
from common import *
from scipy import stats, optimize

PUBLISHED = dict(CV=2.696, KS_p=9.65e-20, NN=0.804, viewshed=1.054, VMR=8.97, R2=0.524)
SUPP_TEXT = dict(CV=1.749, KS_p=np.nan, NN=0.820, viewshed=0.871, VMR=6.41, R2=0.641)
CRIT = dict(CV=(r'$<0.5$', r'$>1.0$'), KS_p=(r'$>0.05$', r'$<0.05$'), NN=(r'$>1.2$', r'$\leq 1.2$'),
            viewshed=(r'$>1.5$', r'$\approx 1.0$'), VMR=(r'$<1.5$', r'$>3.0$'), R2=('poor fit', '0.4--0.7'))
LABEL = dict(CV='CV of spacings', KS_p=r'KS $p$ vs.\ uniform', NN='Nearest-neighbour ratio (1-D)',
             viewshed='Viewshed density ratio', VMR='Variance/mean of bin counts', R2=r'$R^2$ of log-linear fit')


def s1_stats(x):
    d = np.sort(np.asarray(x, float)); n = len(d)
    sp = np.diff(d); cv = sp.std(ddof=1) / sp.mean()
    # asymptotic distribution, as used by R's ks.test when observations are tied (as here)
    ks = stats.kstest(d, 'uniform', args=(0, d.max()), method='asymp')
    nn = np.array([np.min(np.abs(np.delete(d, i) - d[i])) for i in range(n)])
    nnr = nn.mean() / ((d.max() - d.min()) / (2 * n))
    vw = 3.5; inw = np.sum((d >= vw - 1) & (d <= vw + 1)); vr = (inw / 2) / (n / d.max())
    br = np.arange(0, np.ceil(d.max()) + 0.5, 0.5); cnt, _ = np.histogram(d, bins=br); mids = br[:-1] + 0.25
    nz = cnt > 0; sl = stats.linregress(mids[nz], np.log(cnt[nz]))
    return dict(n=n, CV=cv, KS_p=ks.pvalue, NN=nnr, viewshed=vr, VMR=cnt.var(ddof=1) / cnt.mean(), R2=sl.rvalue ** 2)


def aic_gauss(rss, n, k):   # R's AIC for lm/nls: -2 logLik + 2k with sigma^2 = RSS/n
    return n * np.log(2 * np.pi * rss / n) + n + 2 * k


def published_aic(x):
    d = np.sort(np.asarray(x, float)); n = len(d); rank = np.arange(1, n + 1); cp = rank / n
    out = {}
    X = np.c_[np.ones(n), d]; b = np.linalg.lstsq(X, rank, rcond=None)[0]
    out['Linear'] = aic_gauss(((rank - X @ b) ** 2).sum(), n, 3)
    f = lambda l: ((cp - (1 - np.exp(-l * d))) ** 2).sum()
    l1 = optimize.minimize_scalar(f, bounds=(1e-3, 10), method='bounded').x
    out['Exponential_pub'] = aic_gauss(f(l1), n, 2)
    g = lambda p: ((rank - p[0] * d ** p[1]) ** 2).sum()
    pw = optimize.minimize(g, [10, 0.5], method='Nelder-Mead', options=dict(maxiter=20000, xatol=1e-10, fatol=1e-10)).x
    out['Power law'] = aic_gauss(g(pw), n, 3)
    X = np.c_[np.ones(n), np.log(d + 0.1)]; b = np.linalg.lstsq(X, rank, rcond=None)[0]
    out['Logarithmic'] = aic_gauss(((rank - X @ b) ** 2).sum(), n, 3)
    for deg, nm in [(2, 'Quadratic'), (3, 'Cubic')]:
        X = np.vander(d, deg + 1); b = np.linalg.lstsq(X, rank, rcond=None)[0]
        out[nm] = aic_gauss(((rank - X @ b) ** 2).sum(), n, deg + 2)
    h = lambda l: ((rank - n * (1 - np.exp(-l * d))) ** 2).sum()
    l2 = optimize.minimize_scalar(h, bounds=(1e-3, 10), method='bounded').x
    out['Exponential_common'] = aic_gauss(h(l2), n, 2)
    return out, 2 * n * np.log(n), l1, l2


def weights(a):
    a = np.asarray(a, float); w = np.exp(-0.5 * (a - a.min())); return w / w.sum()


def run():
    ds, m, db = datasets()
    a, b = s1_stats(ds['D84']), s1_stats(ds['D62'])
    keys = ['CV', 'KS_p', 'NN', 'viewshed', 'VMR', 'R2']
    pd.DataFrame([dict(statistic=k, label=LABEL[k], published=PUBLISHED[k], supp_text=SUPP_TEXT[k], D84=a[k], D62=b[k],
                       crit_ceremonial=CRIT[k][0], crit_failure=CRIT[k][1]) for k in keys]).to_csv(rpath('s1_stats.csv'), index=False)
    res, off, l1, l2 = published_aic(ds['D84'])
    names = ['Linear', 'Exponential', 'Power law', 'Logarithmic', 'Quadratic', 'Cubic']
    pub = [res['Exponential_pub'] if nm == 'Exponential' else res[nm] for nm in names]
    com = [res['Exponential_common'] if nm == 'Exponential' else res[nm] for nm in names]
    df = pd.DataFrame(dict(model=names, response_pub=['rank/$n$' if nm == 'Exponential' else 'rank' for nm in names],
                           AIC_pub=pub, w_pub=weights(pub), AIC_common=com, w_common=weights(com)))
    df.to_csv(rpath('aic_artifact.csv'), index=False)
    json.dump(dict(offset=off, lambda_pub=l1, lambda_common=l2), open(rpath('aic_artifact.json'), 'w'), indent=1)
    print('S1 reproduced on D84:', {k: float(f'{a[k]:.4g}') for k in keys})
    print('S1 on D62:          ', {k: float(f'{b[k]:.4g}') for k in keys})
    print(df.round(4).to_string(index=False)); print('offset 2 n ln n =', round(off, 2))
