"""Goodness of fit of the exponential law: parametric-bootstrap Anderson-Darling and KS against the authors' fixed rate."""
from common import *
from scipy import stats


def ad_exp(x):
    lam = 1 / x.mean(); z = np.sort(1 - np.exp(-lam * x)); n = len(z); i = np.arange(1, n + 1)
    return -n - np.mean((2 * i - 1) * (np.log(z) + np.log(1 - z[::-1])))


def run(B=5000):
    rng = np.random.default_rng(2026)
    ds, m, db = datasets(); rows = []
    for key in ['D84', 'D62', 'D56']:
        x = ds[key]; a0 = ad_exp(x); n = len(x)
        cnt = sum(ad_exp(rng.exponential(x.mean(), n)) >= a0 for _ in range(B))
        ks = stats.kstest(x, 'expon', args=(0, 2.0))
        rows.append(dict(dataset=key, AD=a0, p_AD=(cnt + 1) / (B + 1), KS_D=ks.statistic, p_KS_fixed=ks.pvalue))
    df = pd.DataFrame(rows); df.to_csv(rpath('gof.csv'), index=False); print(df.round(4).to_string(index=False))
