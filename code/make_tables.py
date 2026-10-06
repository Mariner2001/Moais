"""LaTeX table bodies (tabular environments only) generated from results/.
Their content is embedded verbatim in main.tex."""
import json
from common import *


def num(x, nd=2, plus=False):
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return '--'
    s = f'{x:.{nd}f}'
    if float(s) == 0:
        s = s.lstrip('-')
    if plus and float(s) > 0:
        s = '+' + s
    return f'${s}$'


def pval(p, nd=3, md=1):
    if p is None or not np.isfinite(p):
        return '--'
    if p < 10 ** (-nd):
        e = int(np.floor(np.log10(p))); mnt = p / 10 ** e
        if round(mnt, md) >= 10:
            mnt /= 10; e += 1
        return rf'${mnt:.{md}f}\times10^{{{e}}}$'
    return f'${p:.{nd}f}$'


def write(name, body):
    open(os.path.join(TAB, name), 'w').write(body)


def tab_audit():
    ds, m, db = datasets(); rows = []
    for k in DS_KEYS:
        x = np.sort(ds[k])
        rows.append(rf"{DS_LABEL[k]} & {len(x)} & {100 * np.mean(x <= 0.6):.1f}\% & {100 * np.mean(x <= 1):.1f}\% & {100 * np.mean(x <= 2):.1f}\% & {np.median(x):.2f} & {x.mean():.2f}\\")
    write('tab_audit.tex', '\\begin{tabular}{lrrrrrr}\\toprule\n'
          r'& & \multicolumn{3}{c}{Share of statues within} & \multicolumn{2}{c}{Distance (km)}\\' + '\n'
          r'\cmidrule(lr){3-5}\cmidrule(lr){6-7}' + '\n'
          r'Dataset & $n$ & 0.6\,km & 1\,km & 2\,km & Median & Mean\\\midrule' + '\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


S1_LABEL = dict(CV='Coefficient of variation of gaps',
                KS_p=r'KS test against uniformity, $p$',
                NN='Nearest-neighbour ratio',
                viewshed='Viewshed density ratio',
                VMR='Variance/mean of bin counts',
                R2=r'$R^2$ of log-linear fit')
S1_PUB = dict(CV='2.696', KS_p=r'$9.65\times10^{-20}$', NN='0.804', viewshed='1.054', VMR='8.97', R2='0.524')       # Table S1
S1_TXT = dict(CV='1.749', KS_p=r'$<0.001$', NN='0.820', viewshed='0.871', VMR='6.41', R2='0.641')                   # text of the SI
S1_CRIT = dict(CV=(r'$<0.5$', r'$>1.0$'), KS_p=(r'$>0.05$', r'$<0.05$'), NN=(r'$>1.2$', r'$\leq1.2$'),
               viewshed=(r'$>1.5$', r'$\approx1.0$'), VMR=(r'$<1.5$', r'$>3.0$'), R2=('poor fit', '0.4--0.7'))


def tab_s1():
    s = pd.read_csv(rpath('s1_stats.csv')).set_index('statistic'); rows = []
    for k in ['CV', 'KS_p', 'NN', 'viewshed', 'VMR', 'R2']:
        r = s.loc[k]
        if k == 'KS_p':
            f = lambda x: pval(x, md=2)
        elif k == 'VMR':
            f = lambda x: f'{x:.2f}'
        else:
            f = lambda x: f'{x:.3f}'
        rows.append(f"{S1_LABEL[k]} & {S1_PUB[k]} & {S1_TXT[k]} & {f(r.D84)} & {f(r.D62)} & {S1_CRIT[k][0]} & {S1_CRIT[k][1]}\\\\")
    write('tab_s1.tex', '\\begin{tabular}{lcccccc}\\toprule\n'
          r"& \multicolumn{2}{c}{Published value} & \multicolumn{2}{c}{Recomputed} & \multicolumn{2}{c}{Criterion in Table S1}\\" + '\n'
          r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}' + '\n'
          r"Statistic & Table S1 & SI text & D84 & D62 & Ceremonial & Failure\\\midrule" + '\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


def tab_aic():
    a = pd.read_csv(rpath('aic_artifact.csv')); rows = []
    names = {'Linear': 'Linear (``uniform\'\')'}
    for _, r in a.iterrows():
        rows.append(f"{names.get(r.model, r.model)} & {r.response_pub} & {num(r.AIC_pub, 2)} & {r.w_pub:.3f} & rank & {num(r.AIC_common, 2)} & {r.w_common:.3f}\\\\")
    write('tab_aic.tex', '\\begin{tabular}{llrrlrr}\\toprule\n'
          r'& \multicolumn{3}{c}{Published script} & \multicolumn{3}{c}{All models on the same response}\\' + '\n'
          r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}' + '\n'
          r'Model & Response & AIC & Weight & Response & AIC & Weight\\\midrule' + '\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


LAW_FORM = {'Exponential': 'steady decline from zero', 'Weibull': 'includes the exponential',
            'Gamma': 'includes the exponential', 'Lognormal': 'skewed, peak away from zero',
            'Half-normal': 'bell-shaped decline', 'Two-exponential mixture': 'two exponentials',
            'Exponential + uniform': 'decline plus even scatter'}


def tab_lik():
    d = pd.read_csv(rpath('distance_models.csv')); g = pd.read_csv(rpath('gof.csv')).set_index('dataset')
    cols = [('D84', 0.0), ('D62', 0.0), ('D56', 0.0), ('D84', 0.6), ('D62', 0.6)]
    rows = []
    for mname in LAW_FORM:
        cells = []
        for k, t in cols:
            r = d[(d.dataset == k) & (d.trunc == t) & (d.model == mname)].iloc[0]
            cells.append(r'$\mathbf{0.0}$' if r.dAIC < 1e-9 else num(r.dAIC, 1))
        rows.append(f'{mname} & {LAW_FORM[mname]} & ' + ' & '.join(cells) + r'\\')
    ns = [d[(d.dataset == k) & (d.trunc == t)].n.iloc[0] for k, t in cols]
    head = (r'& & \multicolumn{3}{c}{All distances} & \multicolumn{2}{c}{Beyond 0.6\,km}\\' + '\n'
            r'\cmidrule(lr){3-5}\cmidrule(lr){6-7}' + '\n'
            r'Law & Shape & ' + ' & '.join(k for k, t in cols) + r'\\' + '\n'
            r'& & ' + ' & '.join(f'$n={n}$' for n in ns) + r'\\\midrule')
    gof = (r'\midrule' + '\n' + r'\multicolumn{2}{l}{Exponential, fitted rate: Anderson--Darling $p$} & '
           + ' & '.join(pval(g.loc[k, 'p_AD']) for k in ['D84', 'D62', 'D56']) + r' & -- & --\\' + '\n'
           + r'\multicolumn{2}{l}{Exponential, $\lambda=0.5$\,km$^{-1}$: Kolmogorov--Smirnov $p$} & '
           + ' & '.join(pval(g.loc[k, 'p_KS_fixed']) for k in ['D84', 'D62', 'D56']) + r' & -- & --\\')
    write('tab_lik.tex', '\\begin{tabular}{llrrrrr}\\toprule\n' + head + '\n' + '\n'.join(rows) + '\n' + gof + '\n\\bottomrule\n\\end{tabular}\n')


def tab_hazard():
    h = pd.read_csv(rpath('hazard_bins.csv')); s = pd.read_csv(rpath('survival_summary.csv')).set_index('dataset'); rows = []
    for a, b in zip([0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 10], [0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 12]):
        cells = []
        for k in ['D84', 'D62']:
            r = h[(h.dataset == k) & (h.a == a)].iloc[0]
            cells.append(f"{int(r.events)} & {r.statue_km:.0f} & {100 * r.hazard:.1f} ({100 * r.lo:.1f}--{100 * r.hi:.1f})")
        rows.append(f"{a:g}--{b:g} & " + ' & '.join(cells) + r'\\')
    mc = lambda txt: r'\multicolumn{3}{c}{' + txt + '}'
    summ = (r'\midrule' + '\n'
            + r'Rate, 0--2\,km & ' + mc(f"{100 * s.loc['D84', 'h_near']:.1f}") + ' & ' + mc(f"{100 * s.loc['D62', 'h_near']:.1f}") + r'\\' + '\n'
            + r'Rate, $>2$\,km & ' + mc(f"{100 * s.loc['D84', 'h_far']:.1f}") + ' & ' + mc(f"{100 * s.loc['D62', 'h_far']:.1f}") + r'\\' + '\n'
            + r'Ratio & ' + mc(f"{s.loc['D84', 'ratio']:.1f}") + ' & ' + mc(f"{s.loc['D62', 'ratio']:.1f}") + r'\\' + '\n'
            + r'Completing 10\,km$^{a}$ & ' + mc(f"{s.loc['D84', 'S10']:.2f}") + ' & ' + mc(f"{s.loc['D62', 'S10']:.2f}") + r'\\')
    write('tab_hazard.tex', '\\begin{tabular}{lrrcrrc}\\toprule\n'
          r'& \multicolumn{3}{c}{D84 (published set)} & \multicolumn{3}{c}{D62 (database road class)}\\' + '\n'
          r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}' + '\n'
          r'Distance (km) & Failures & Statue-km & Rate (95\% CI) & Failures & Statue-km & Rate (95\% CI)\\\midrule' + '\n'
          + '\n'.join(rows) + '\n' + summ + '\n\\bottomrule\n'
          + r'\multicolumn{7}{l}{\footnotesize $^{a}$ Estimated share of statues completing a 10\,km journey (Kaplan--Meier).}\\' + '\n\\end{tabular}\n')


def tab_rank():
    r = pd.read_csv(rpath('rank_tests.csv')).set_index('sample')
    cols = [('all46', 'Published sample'), ('road', 'Road \\emph{moai}'), ('road_single', 'Road, one piece')]
    head = (r'Statistic & ' + ' & '.join(lab for k, lab in cols) + r'\\' + '\n'
            + r'& ' + ' & '.join(f"$n={int(r.loc[k, 'n'])}$" for k, lab in cols) + r'\\\midrule')
    lines = [
        (r'Pearson $r$ ($p$)', lambda x: f"{num(x.r, 2)} ({pval(x.p_r, 2)})"),
        (r"Kendall's $\tau$ ($p$)", lambda x: f"{num(x.tau, 2)} ({pval(x.p_tau, 2)})"),
        (r'95\% bootstrap interval for $\tau$', lambda x: f"{num(x.tau_lo, 2)} to {num(x.tau_hi, 2)}"),
        (r"Spearman's $\rho$ ($p$)", lambda x: f"{num(x.rho, 2)} ({pval(x.p_rho, 2)})"),
        (r"Rank test of independence ($p$)", lambda x: f"{pval(x.p_cvm, 2)}"),
        (r'Longest quarter in nearest quarter ($p$)', lambda x: f"{int(x.corner_k)} of {int(x.corner_n)} ({pval(x.corner_p, 2)})"),
    ]
    rows = [lab + ' & ' + ' & '.join(f(r.loc[k]) for k, _ in cols) + r'\\' for lab, f in lines]
    write('tab_rank.tex', '\\begin{tabular}{lccc}\\toprule\n' + head + '\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


def tab_copula():
    c = pd.read_csv(rpath('copula_fits.csv')); rows = []
    fams = [('Gaussian', ''), ('Frank', ''), ('Clayton', '0'), ('Clayton', '90'), ('Clayton', '180'), ('Clayton', '270'),
            ('Gumbel', '0'), ('Gumbel', '90'), ('Gumbel', '180'), ('Gumbel', '270')]
    c['rot'] = c['rot'].fillna('').astype(str).str.replace('.0', '', regex=False)
    for fam, rot in fams:
        cells = []
        for smp in ['all46', 'road']:
            r = c[(c['sample'] == smp) & (c.family == fam) & (c['rot'] == rot)].iloc[0]
            if bool(r.boundary):
                cells.append(r'\multicolumn{3}{c}{at independence boundary} & ' + num(r.dAIC_indep, 2, plus=True))
            else:
                cells.append(f"{num(r.par, 3)} & {num(r.tau, 3)} & {num(r.lam10, 3)} & {num(r.dAIC_indep, 2, plus=True)}")
        lab = fam + (rf", ${rot}^\circ$" if rot != '' else '')
        rows.append(lab + ' & ' + ' & '.join(cells) + r'\\')
    write('tab_copula.tex', '\\begin{tabular}{lrrrrrrrr}\\toprule\n'
          r'& \multicolumn{4}{c}{Published sample ($n=46$)} & \multicolumn{4}{c}{Road \emph{moai} only ($n=38$)}\\' + '\n'
          r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}' + '\n'
          r'Family & $\hat\theta$ & $\tau(\hat\theta)$ & $\lambda_{10}$ & $\Delta$AIC & $\hat\theta$ & $\tau(\hat\theta)$ & $\lambda_{10}$ & $\Delta$AIC\\\midrule' + '\n'
          + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


def pshort(p):
    return r'$<0.001$' if p < 0.001 else f'${p:.3f}$'


def tab_terrain():
    t = pd.read_csv(rpath('terrain_chi2.csv')); o = json.load(open(rpath('orientation.json')))
    pub = json.load(open(rpath('terrain_published_shares.json'))); ob = o['observed']; rows = []
    name = {'raraku_southcoast': 'Rano Raraku--coast', 'southcoast': 'South coast'}
    for _, r in t.iterrows():
        rows.append(f"{name[r.road]} & {r.thr:g}\\% & {100 * r.desc:.1f}/{100 * r.flat:.1f}/{100 * r.asc:.1f} & "
                    f"{r.chi2:.1f} & {pshort(r.p)} & {pshort(r.p_flat_le_obs)} & "
                    f"{r.e_asc_sloping:.1f} & {pshort(r.p_binom_sloping)}\\\\")
    rows.append(r"Rano Raraku--coast$^{a}$ & -- & 45.0/34.0/21.0 & "
                + f"{pub['chi2']:.1f} & {pshort(pub['p'])} & -- & "
                + f"{pub['e_asc_sloping']:.1f} & {pshort(pub['p_binom_sloping'])}\\\\")
    write('tab_terrain.tex', '\\begin{tabular}{lrcccccc}\\toprule\n'
          r'& Flat if & Terrain (\%) & \multicolumn{2}{c}{All 51 statues} & Flat & \multicolumn{2}{c}{Ascending, of 48}\\' + '\n'
          r'\cmidrule(lr){4-5}\cmidrule(lr){7-8}' + '\n'
          r'Road profile & $|$slope$|\leq$ & desc./flat/asc. & $\chi^2_2$ & $p$ & $p$ & expected & $p$\\\midrule' + '\n'
          + '\n'.join(rows) + '\n\\bottomrule\n'
          + r'\multicolumn{8}{l}{\footnotesize $^{a}$ Terrain shares stated by Lipo and Hunt (2025a, Section 1.2).}\\' + '\n\\end{tabular}\n')


def tab_mech():
    j = json.load(open(rpath('mechanics.json')))
    rows = [
        (r'Volume of statue 12-220-01 (m$^3$)', r'--', f"{j['volume_m3']:.1f} (3D mesh)"),
        (r'Mass at 1.86\,g\,cm$^{-3}$ (t)', f"{j['eq1_mass_W283']:.1f} or {j['eq1b_mass_W283']:.1f} (Eq.~1)$^{{a}}$", f"{j['mass_t']:.1f} (3D mesh)"),
        (r'Height of the centre of mass (m; share of height)', r'2.98; 0.406', f"{j['com_height_m']:.2f}; {j['com_frac']:.3f}"),
        (r'Offset of the centre of mass from the base centre (cm)', r"34.0 ``backward'', 23.2 lateral (Fig.~4)", f"{abs(j['off_foreaft_cm']):.1f} fore--aft, {abs(j['off_lateral_cm']):.1f} lateral"),
        (r'Distance from the centre of mass to the fore and aft base edges (m)', r"``close to or even slightly beyond'' the front edge (text); 1.43 to the front edge (Fig.~4)", f"{j['edge_zmin_m']:.2f} and {j['edge_zmax_m']:.2f}"),
        (r'Further tilt needed to topple forwards or backwards ($^\circ$)', r'$\approx0$ (implied by the text)', f"{j['tip_foreaft_deg'][0]:.1f} and {j['tip_foreaft_deg'][1]:.1f}"),
        (r"``Forward lean'' ($^\circ$)", r'6.5', f"{j['authors_lean_deg']:.1f}; size of the offset only, direction not determined"),
        (r'Scale of the 4.35\,t replica and implied height (m)', r'not reported', f"{j['replica_scale']:.2f}; {j['replica_height_m']:.1f}"),
        (r'Mass of a similar replica 3.0\,m tall (t)', r'4.35 (replica)', f"{j['mass_3m_replica_t']:.1f}"),
        (r'Crew for the full-size statue, scaled from the 18-person crew', r'35--50 (statues of 6--8\,m)', f"about {round(j['crew_froude'], -1):.0f}"),
        (r'Free-rocking period at 2, 5 and 10$^\circ$ (s)', f"{j['T_simple_R']:.2f} at any amplitude (simple pendulum)", ', '.join(f'{x:.2f}' for x in j['T_proto_2_5_10'])),
        (r'Speed implied by 0.89\,m steps and 2.2--3.8\,s cycles (km/h)', r'0.31 (mean)', f"{j['v_range_kmh'][0]:.2f}--{j['v_range_kmh'][1]:.2f}"),
        (r'Time for 11\,000 rocking cycles (h)', r'32', f"{j['hours_11000_cycles'][0]:.1f}--{j['hours_11000_cycles'][1]:.1f}"),
        (r'Power for a 20\,t statue at 0.31\,km/h, friction coefficient 0.4 (kW)', r'6.2', f"{j['power_20t_kW']:.2f}"),
        (r'Crew needed at 75--100\,W per person', r'15--20', f"{j['crew_at_75_100W'][0]:.0f}--{j['crew_at_75_100W'][1]:.0f}"),
    ]
    write('tab_mech.tex', '\\begin{tabular}{p{6.0cm}p{4.9cm}p{3.9cm}}\\toprule\n'
          r'Quantity & Lipo and Hunt (2025a) & Recomputed\\\midrule' + '\n' + '\n'.join(f'{a} & {b} & {c}\\\\' for a, b, c in rows)
          + '\n\\bottomrule\n'
          + r'\multicolumn{3}{p{15.2cm}}{\footnotesize $^{a}$ For a base width of 2.83\,m: '
          + f"{j['eq1_mass_W283']:.1f}\\,t if the depth is taken as 50\\% of the average width, {j['eq1b_mass_W283']:.1f}\\,t if as 50\\% of the base width."
          + r'}\\' + '\n\\end{tabular}\n')


def tab_records():
    m = pd.read_csv(dpath('road84_annotated.csv')); db = load_db().set_index('OBJECTID')
    q = m[m.LOCATION_TYPE == 'QUARRY']; a = m[m.LOCATION_TYPE == 'AHU']
    rows = [rf"Quarry statue & {len(q)} & {', '.join(str(int(i)) for i in sorted(q.OBJECTID))} & {q.distance_from_quarry_km.min():.2f}--{q.distance_from_quarry_km.max():.2f} & Rano Raraku\\"]
    for _, r in a.sort_values('distance_from_quarry_km').iterrows():
        rows.append(rf"\emph{{Ahu}} statue & 1 & {int(r.OBJECTID)} & {r.distance_from_quarry_km:.2f} & {str(r.LOCATION_NAME).strip().title()}\\")
    write('tab_added.tex', '\\begin{tabular}{llp{5.4cm}rl}\\toprule\n'
          r'Class in the database & $n$ & OBJECTID & Distance (km) & Location name\\\midrule' + '\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')
    rows = []
    for x, y, sep in DUPLICATE_PAIRS:
        rx, ry = db.loc[x], db.loc[y]
        mark = '$^{a}$' if x == 63 else ''
        sx, sy = rx.SOURCE.title().replace('&', r'\&'), ry.SOURCE.title().replace('&', r'\&')
        rows.append(rf"{x} ({sx}){mark} & {y} ({sy}) & {sep:.1f} & {ry.dist_km:.2f}\\")
    write('tab_dups.tex', '\\begin{tabular}{llrr}\\toprule\n'
          r'Record removed (survey) & Record kept (survey) & Separation (m) & Distance (km)\\\midrule' + '\n' + '\n'.join(rows)
          + '\n\\bottomrule\n' + r'\multicolumn{4}{l}{\footnotesize $^{a}$ Noted in the database as a highly probable match.}\\'
          + '\n\\end{tabular}\n')


def run():
    for f in (tab_audit, tab_s1, tab_aic, tab_lik, tab_hazard, tab_rank, tab_copula, tab_terrain, tab_mech, tab_records):
        f()
    print('tables written to', TAB, sorted(os.listdir(TAB)))


if __name__ == '__main__':
    run()
