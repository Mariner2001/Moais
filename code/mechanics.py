"""Mechanical consistency checks: mass and centre of mass from the authors' mesh, rocking-block period, Froude scaling."""
import json
from common import *

G = 9.81; RHO = 1.86; SCALE = 4.894   # mesh scale used by the authors (7.35 m / 1.502 mesh units)


def load_obj(path):
    V, F = [], []
    for line in open(path):
        if line.startswith('v '):
            V.append([float(t) for t in line.split()[1:4]])
        elif line.startswith('f '):
            F.append([int(t.split('/')[0]) - 1 for t in line.split()[1:4]])
    return np.array(V), np.array(F)


def eq1(h, W):   # Lipo & Hunt Eq. (1): V = h * (0.7 W) * (0.5 * 0.7 W) * 0.65
    wb = 0.7 * W; V = h * wb * 0.5 * wb * 0.65; return V, RHO * V


def housner_T(theta0, R, alpha):
    p = np.sqrt(3 * G / (4 * R)); return 4 / p * np.arccosh(1 / (1 - theta0 / alpha))


def run():
    V, F = load_obj(dpath('SimplifiedMoai.obj'))
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = np.einsum('ij,ij->i', v0, np.cross(v1, v2)) / 6.0; Vt = vol.sum()
    com = ((v0 + v1 + v2) * vol[:, None]).sum(0) / 4.0 / Vt
    E = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1); _, cnt = np.unique(E, axis=0, return_counts=True)
    ymin, ymax = V[:, 1].min(), V[:, 1].max(); H = (ymax - ymin) * SCALE; hc = (com[1] - ymin) * SCALE
    base = V[V[:, 1] < ymin + 0.05]
    bx, bz = (base[:, 0].min(), base[:, 0].max()), (base[:, 2].min(), base[:, 2].max())
    cx, cz = (bx[0] + bx[1]) / 2, (bz[0] + bz[1]) / 2
    off_x, off_z = (com[0] - cx) * SCALE, (com[2] - cz) * SCALE
    e_zmax, e_zmin = (bz[1] - com[2]) * SCALE, (com[2] - bz[0]) * SCALE
    e_x1, e_x2 = (bx[1] - com[0]) * SCALE, (com[0] - bx[0]) * SCALE
    vol_m3 = Vt * SCALE ** 3; mass = RHO * vol_m3
    V283, M283 = eq1(7.35, 2.83); V289, M289 = eq1(7.35, 2.89)
    # second reading of Eq. 1: depth = 50% of the base width rather than of the average width
    V283b = 7.35 * (0.7 * 2.83) * (0.5 * 2.83) * 0.65; M283b = RHO * V283b
    s = (4.35 / mass) ** (1 / 3); m3 = mass * (3.0 / 7.35) ** 3
    b = 2.89 / 2; R = np.hypot(b, hc); alpha = np.arctan(b / hc)
    th = np.radians(np.array([2, 5, 10])); Tp = housner_T(th, R, alpha); Tr = Tp * np.sqrt(s)
    T_simple_R = 2 * np.pi * np.sqrt(R / G); T_simple_h = 2 * np.pi * np.sqrt(hc / G)
    grid = np.radians(np.linspace(0.25, 0.97 * np.degrees(alpha), 300))
    pd.DataFrame(dict(theta0_deg=np.degrees(grid), T_proto=housner_T(grid, R, alpha), T_replica=housner_T(grid, R, alpha) * np.sqrt(s))).to_csv(rpath('housner_curve.csv'), index=False)
    v_lo, v_hi = 0.89 / 3.8 * 3.6, 0.89 / 2.2 * 3.6
    P = 0.4 * 20000 * G * (0.31 / 3.6)
    out = dict(mesh_watertight=bool((cnt == 2).all()), height_m=H, volume_m3=vol_m3, mass_t=mass,
               com_height_m=hc, com_frac=hc / H, base_width_m=(bx[1] - bx[0]) * SCALE, base_depth_m=(bz[1] - bz[0]) * SCALE,
               off_lateral_cm=off_x * 100, off_foreaft_cm=off_z * 100, edge_zmax_m=e_zmax, edge_zmin_m=e_zmin, edge_x1_m=e_x1, edge_x2_m=e_x2,
               tip_foreaft_deg=[float(np.degrees(np.arctan(e_zmin / hc))), float(np.degrees(np.arctan(e_zmax / hc)))],
               tip_lateral_deg=[float(np.degrees(np.arctan(min(e_x1, e_x2) / hc))), float(np.degrees(np.arctan(max(e_x1, e_x2) / hc)))],
               authors_lean_deg=float(np.degrees(np.arctan(abs(off_z) / hc))),
               eq1_vol_W283=V283, eq1_mass_W283=M283, eq1_vol_W289=V289, eq1_mass_W289=M289, mass_ratio_mesh_eq1=mass / M283,
               eq1b_vol_W283=V283b, eq1b_mass_W283=M283b, mass_ratio_mesh_eq1b=mass / M283b,
               replica_scale=s, replica_height_m=7.35 * s, mass_3m_replica_t=m3, force_ratio=1 / s ** 3, time_ratio=s ** -0.5,
               power_ratio=s ** -3.5, crew_froude=18 / s ** 3, housner_R=R, housner_alpha_deg=float(np.degrees(alpha)),
               T_proto_2_5_10=Tp.tolist(), T_replica_2_5_10=Tr.tolist(), T_simple_R=T_simple_R, T_simple_h=T_simple_h,
               T_simple_R_replica=T_simple_R * np.sqrt(s), v_range_kmh=[v_lo, v_hi], hours_11000_cycles=[11000 * 2.2 / 3600, 11000 * 3.8 / 3600],
               hours_10km_at_031=10 / 0.31, power_20t_kW=P / 1000, power_per_person_W=[P / 20, P / 15], crew_at_75_100W=[P / 100, P / 75])
    out = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}
    json.dump(out, open(rpath('mechanics.json'), 'w'), indent=1)
    for k, v in out.items():
        print(f'{k:24s} {v}')
