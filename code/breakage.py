"""Are near-quarry road moai broken more often? (test of the 'hidden structural flaw' explanation)

Each road moai of the source database is classified from three fields (number of fragments, state found,
general condition). One record is kept per location (records 411 and 701 share their coordinates).
Records described only as a 'single fragment' are ambiguous; the main analysis treats them as indeterminate
and the sensitivity analysis treats them as broken or as intact."""
import json
from common import *
from scipy import stats
from scipy.stats.contingency import odds_ratio
import statsmodels.api as sm


def classify(r, single='unknown'):
    s = str(r.STATE_FOUND).lower(); gc = str(r.GENERAL_CONDITION).lower()
    try:
        nf = float(r.NUMBER_OF_FRAGMENTS)
    except (TypeError, ValueError):
        nf = np.nan
    if (nf >= 2) or ('fragment' in s) or ('broken' in gc) or ('head only' in gc):
        return 'broken'
    if ('single fragmant' in gc) or ('single fragment' in gc):     # spelling as in the database
        return single
    if 'intact' in gc:
        return 'intact'
    if (nf == 1) or (s == 'complete'):
        return 'intact'
    return 'unknown'


def test(k, cut):
    a, b = k[k.dist_km < cut], k[k.dist_km >= cut]
    tab = np.array([[a.y.sum(), len(a) - a.y.sum()], [b.y.sum(), len(b) - b.y.sum()]])
    p = stats.fisher_exact(tab)[1]; o = odds_ratio(tab, kind='conditional'); ci = o.confidence_interval(0.95)
    return dict(cut=cut, broken_near=int(a.y.sum()), n_near=len(a), broken_far=int(b.y.sum()), n_far=len(b),
                OR=o.statistic, OR_lo=ci.low, OR_hi=ci.high, p=p)


def road_records():
    db = load_db()
    road = db[(db.LOCATION_TYPE == 'ROAD') & db.dist_km.notna()].sort_values('OBJECTID')
    return road.drop_duplicates(['lat', 'lon'], keep='first').copy()     # drops 701 (same coordinates as 411)


def run():
    road = road_records(); rows = []; summ = {}
    for single in ['unknown', 'broken', 'intact']:
        road['cond'] = road.apply(lambda r: classify(r, single), axis=1)
        if single == 'unknown':
            road[['OBJECTID', 'dist_km', 'STATE_FOUND', 'NUMBER_OF_FRAGMENTS', 'GENERAL_CONDITION', 'cond']].sort_values('dist_km').to_csv(rpath('road_breakage_classified.csv'), index=False)
            summ.update(n_records=len(road), n_intact=int((road.cond == 'intact').sum()), n_broken=int((road.cond == 'broken').sum()),
                        n_unknown=int((road.cond == 'unknown').sum()))
        k = road[road.cond != 'unknown'].copy(); k['y'] = (k.cond == 'broken').astype(int)
        k2 = k[~k.OBJECTID.isin(PROBABLE_DUPLICATES)]
        for smp, kk in [('all', k), ('no_duplicates', k2)]:
            for c in (1.0, 2.0):
                rows.append(dict(single_fragment=single, sample=smp, **test(kk, c)))
        if single == 'unknown':
            fit = sm.Logit(k.y.values, sm.add_constant(np.log(k.dist_km.values))).fit(disp=0)
            summ.update(logit_slope=float(fit.params[1]), logit_se=float(fit.bse[1]), logit_p=float(fit.pvalues[1]))
    df = pd.DataFrame(rows); df.to_csv(rpath('breakage_tests.csv'), index=False)
    summ.update(OR2_range=[float(df[df.cut == 2].OR.min()), float(df[df.cut == 2].OR.max())],
                OR1_range=[float(df[df.cut == 1].OR.min()), float(df[df.cut == 1].OR.max())],
                p_min_all_variants=float(df.p.min()))
    json.dump(summ, open(rpath('breakage_summary.json'), 'w'), indent=1)
    print(df.round(3).to_string(index=False)); print(summ)
