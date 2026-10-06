"""Figures of the paper (vector PDF, with PNG previews)."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from scipy import stats

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10', 'DejaVu Serif'], 'mathtext.fontset': 'cm',
                     'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False, 'font.size': 8.5,
                     'axes.labelsize': 8.5, 'axes.titlesize': 8.5, 'legend.fontsize': 6.8, 'xtick.labelsize': 7.5,
                     'ytick.labelsize': 7.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03})
BLUE, ORANGE, RED, GREY = '#2f5d8a', '#e08a2b', '#b03a2e', '#7f7f7f'
W = 6.3


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf'), metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(os.path.join(FIG, name + '.png'), dpi=200); plt.close(fig)


def fig_distances():
    ds, m, db = datasets(); dm = pd.read_csv(rpath('distance_models.csv'))
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.55))
    bins = np.arange(0, 12.5 + 1e-9, 0.5); d = m.distance_from_quarry_km.values; c = m.LOCATION_TYPE.values
    parts = [(d[c == 'ROAD'], f"classified as road moai ({(c == 'ROAD').sum()})", BLUE),
             (d[c == 'QUARRY'], f"added; classified as quarry statues ({(c == 'QUARRY').sum()})", ORANGE),
             (d[c == 'AHU'], f"added; classified as ahu statues ({(c == 'AHU').sum()})", RED)]
    ax[0].hist([p[0] for p in parts], bins=bins, stacked=True, color=[p[2] for p in parts], label=[p[1] for p in parts],
               edgecolor='white', linewidth=0.4)
    ax[0].axvline(0.6, color='k', ls=':', lw=0.8)
    ax[0].set_xlim(0, 12.5); ax[0].set_xlabel('Distance from quarry centroid (km)'); ax[0].set_ylabel('Number of statues')
    ax[0].legend(frameon=False, loc='upper right'); ax[0].set_title('(a) Composition of the published 84-statue set', loc='left')
    x = np.linspace(0, 12.5, 600)
    for key, col, ls, lab in [('D84', 'k', '-', 'D84, published set (n = 84)'), ('D62', BLUE, '-', 'D62, road class of the database (n = 62)'),
                              ('D56', BLUE, (0, (3, 2)), 'D56, without probable duplicates (n = 56)')]:
        y = np.sort(ds[key]); ax[1].step(np.r_[0, y], np.r_[0, np.arange(1, len(y) + 1) / len(y)], where='post', color=col, ls=ls, lw=1.0, label=lab)
    ax[1].plot(x, 1 - np.exp(-0.5 * x), color=RED, ls=':', lw=1.4, label=r"published model: exponential, $\lambda = 0.5$ km$^{-1}$")
    r = dm[(dm.dataset == 'D62') & (dm.trunc == 0) & (dm.model == 'Lognormal')].iloc[0]
    ax[1].plot(x, stats.lognorm.cdf(x, np.exp(r.p1), scale=np.exp(r.p0)), color=BLUE, ls='-.', lw=0.8, label='lognormal law fitted to D62')
    ax[1].set_xlim(0, 12.5); ax[1].set_ylim(0, 1.02); ax[1].set_xlabel('Distance from quarry centroid (km)'); ax[1].set_ylabel('Cumulative proportion')
    ax[1].legend(frameon=False, loc='lower right'); ax[1].set_title('(b) Cumulative distributions of distance', loc='left')
    fig.tight_layout(w_pad=1.5); save(fig, 'fig1_distances')


def fig_hazard():
    h = pd.read_csv(rpath('hazard_bins.csv')); sm = pd.read_csv(rpath('survival_summary.csv')).set_index('dataset')
    fig, ax = plt.subplots(figsize=(W, 2.5))
    for key, col, lab, dx in [('D84', 'k', 'D84 (published set)', -0.07), ('D62', BLUE, 'D62 (road class of the database)', 0.07)]:
        s = h[(h.dataset == key) & (h.b <= 12)]; edges = np.r_[s.a.values, s.b.values[-1]]; mid = (s.a.values + s.b.values) / 2 + dx
        ax.stairs(100 * s.hazard.values, edges, color=col, lw=1.2, label=lab)
        ax.errorbar(mid, 100 * s.hazard.values, yerr=[100 * (s.hazard - s.lo).values, 100 * (s.hi - s.hazard).values],
                    fmt='none', ecolor=col, elinewidth=0.7, capsize=1.5, alpha=0.8)
        ax.plot([0, 2], [100 * sm.loc[key, 'h_near']] * 2, color=col, ls='--', lw=0.7)
        ax.plot([2, 12], [100 * sm.loc[key, 'h_far']] * 2, color=col, ls='--', lw=0.7)
    ax.set_xlim(0, 12); ax.set_ylim(0, 13); ax.set_xlabel('Straight-line distance from quarry centroid (km)')
    ax.set_ylabel('Failure rate (% per km travelled)'); ax.legend(frameon=False, loc='upper right')
    ax.text(6.0, 7.2, 'bars: 95% bootstrap intervals\ndashed: average rates for 0-2 km and 2-12 km', fontsize=6.8, color=GREY)
    fig.tight_layout(); save(fig, 'fig2_hazard')


def fig_copula():
    f = pd.read_csv(dpath('figure13_annotated.csv')); n = len(f)
    u = stats.rankdata(f.total_length_cm) / (n + 1); v = stats.rankdata(f.distance_from_quarry_km) / (n + 1)
    road = f['class'].values == 'ROAD'; pw = pd.read_csv(rpath('power_n46.csv')); rk = pd.read_csv(rpath('rank_tests.csv')).set_index('sample')
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.95), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax[0].add_patch(plt.Rectangle((0.75, 0), 0.25, 0.25, color=RED, alpha=0.13, lw=0))
    for q in (0.25, 0.5, 0.75):
        ax[0].axhline(q, color=GREY, lw=0.4, ls=':'); ax[0].axvline(q, color=GREY, lw=0.4, ls=':')
    ax[0].scatter(u[road], v[road], s=13, color=BLUE, label=f'road moai (n = {road.sum()})', zorder=3)
    ax[0].scatter(u[~road], v[~road], s=17, marker='^', facecolor='none', edgecolor=ORANGE, linewidth=0.9,
                  label=f'quarry statues not on bedrock (n = {(~road).sum()})', zorder=3)
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1); ax[0].set_aspect('equal')
    ax[0].set_xlabel('Length rank (0 = shortest, 1 = longest)'); ax[0].set_ylabel('Distance rank (0 = nearest, 1 = farthest)')
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
    ax[0].set_title(r'(a) Ranks of length and distance ($n = 46$)', loc='left')
    t_obs, t_lo = -rk.loc['all46', 'tau'], -rk.loc['all46', 'tau_lo']
    ax[1].axvspan(0, t_lo, color=GREY, alpha=0.10, lw=0)
    ax[1].plot(-pw.tau, pw.gauss, color='k', marker='o', ms=3, lw=1.0, label='association spread evenly (Gaussian copula)')
    ax[1].plot(-pw.tau, pw.clayton90, color=RED, marker='s', ms=3, lw=1.0, ls='--', label='association among long, near statues (rotated Clayton copula)')
    ax[1].axhline(0.8, color=GREY, ls=':', lw=0.8); ax[1].axvline(t_obs, color=BLUE, lw=0.9)
    ax[1].text(t_obs + 0.01, 0.42, r'observed $|\tau| = %.2f$' % t_obs, color=BLUE, fontsize=6.8, rotation=90, va='bottom')
    ax[1].text(0.012, 0.99, 'shaded: strengths of association\ncompatible with the data (95%)', fontsize=6.3, color=GREY, va='top')
    ax[1].set_xlim(0, 0.5); ax[1].set_ylim(0, 1.02); ax[1].set_xlabel(r"Strength of association (Kendall's $|\tau|$)")
    ax[1].set_ylabel('Probability of detection (5% level)'); ax[1].legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1)
    ax[1].set_title('(b) Chance of detecting an association with 46 statues', loc='left')
    fig.tight_layout(w_pad=1.2); save(fig, 'fig3_copula')


def fig_housner():
    hc = pd.read_csv(rpath('housner_curve.csv')); mj = json.load(open(rpath('mechanics.json')))
    fig, ax = plt.subplots(figsize=(W * 0.66, 2.45))
    ax.plot(hc.theta0_deg, hc.T_proto, color='k', lw=1.1, label='statue 12-220-01 (rocking block)')
    ax.plot(hc.theta0_deg, hc.T_replica, color=BLUE, lw=1.1, label=f"replica at scale {mj['replica_scale']:.2f} (rocking block)")
    ax.axhline(mj['T_simple_R'], color='k', ls='--', lw=0.8, label='simple-pendulum formula, statue')
    ax.axhline(mj['T_simple_R_replica'], color=BLUE, ls='--', lw=0.8, label='simple-pendulum formula, replica')
    ax.set_xlim(0, mj['housner_alpha_deg']); ax.set_ylim(0, 6)
    ax.set_xlabel('Rocking amplitude (degrees)'); ax.set_ylabel('Free-rocking period (s)')
    ax.legend(frameon=False, loc='upper left'); fig.tight_layout(); save(fig, 'fig4_housner')


def run():
    fig_distances(); fig_hazard(); fig_copula(); fig_housner(); print('figures written to', FIG)
