"""Corrected Fig. 2 contrast (Van Tilburg 1986 data): reproduction, rank test and size (allometry) adjustment."""
import json
from common import *
from scipy import stats


def ols(X, y):
    b = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X @ b; s2 = r @ r / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X))); return b, se


def run():
    v = pd.read_csv(dpath('vantilburg1986_extract.csv'))
    v = v[(v.base_width > 0) & (v.shoulder_width > 0)].copy(); v['SB'] = v.shoulder_width / v.base_width; v['BS'] = 1 / v.SB
    road, ahu = v[v.location.isin([6, 7])], v[v.location == 8]
    t = stats.ttest_ind(ahu.SB, road.SB, equal_var=False); mw = stats.mannwhitneyu(ahu.SB, road.SB)
    pa, pr = v[v.location.between(1, 6)], v[v.location == 8]
    tp = stats.ttest_ind(pa.BS, pr.BS, equal_var=False)
    e = pd.concat([road.assign(road=1), ahu.assign(road=0)]); e = e[e.length > 0]
    b, se = ols(np.c_[np.ones(len(e)), np.log(e.length), e.road], np.log(e.SB.values))
    out = dict(n_road=len(road), mean_road=road.SB.mean(), sd_road=road.SB.std(), n_ahu=len(ahu), mean_ahu=ahu.SB.mean(), sd_ahu=ahu.SB.std(),
               welch_t=t.statistic, welch_df=t.df, welch_p=t.pvalue, mw_p=mw.pvalue,
               published_coding_n=[len(pa), len(pr)], published_coding_t=tp.statistic, published_coding_df=tp.df, published_coding_p=tp.pvalue,
               ancova_n=len(e), ancova_logL=b[1], ancova_logL_se=se[1], ancova_road=b[2], ancova_road_se=se[2], ancova_road_t=b[2] / se[2],
               mean_length_loc6=v[v.location == 6].length.mean(), mean_length_loc7=v[v.location == 7].length.mean(), mean_length_loc8=ahu.length.mean())
    out = {k: (float(x) if not isinstance(x, list) else x) for k, x in out.items()}
    json.dump(out, open(rpath('morphometrics.json'), 'w'), indent=1)
    for k, x in out.items():
        print(f'{k:22s} {x}')
