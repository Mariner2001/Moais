"""Shared paths, constants and dataset definitions for the road-moai reappraisal.
Data: Lipo & Hunt public repository (https://github.com/clipo/moai_walking, MIT licence)."""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
TAB = os.path.join(ROOT, 'tables')
for _d in (DATA, RES, FIG, TAB):
    os.makedirs(_d, exist_ok=True)

QLAT, QLON = -27.125175, -109.288170   # quarry centroid used by Lipo & Hunt (2025)
# HUNT/LIPO-only record, SHEPARDSON record, separation (m). 63/154 is flagged as a probable match in the database.
DUPLICATE_PAIRS = [(695, 334, 0.6), (694, 463, 2.8), (63, 154, 4.9), (696, 459, 10.2), (693, 461, 11.1)]
PROBABLE_DUPLICATES = [a for a, b, d in DUPLICATE_PAIRS]
DS_KEYS = ['D84', 'D62', 'D61', 'D56']
DS_LABEL = {'D84': r'D84: published set (Figs.~12 and S1)', 'D62': r'D62: road class of the source database',
            'D61': r'D61: D62 with one record per location', 'D56': r'D56: D61 without probable duplicates'}


def dpath(*a):
    return os.path.join(DATA, *a)


def rpath(*a):
    return os.path.join(RES, *a)


def load_db():
    return pd.read_csv(dpath('schumacher2013_extract.csv'))


def datasets():
    m = pd.read_csv(dpath('road84_annotated.csv'))
    db = load_db()
    road_db = db[(db.LOCATION_TYPE == 'ROAD') & db.lat.notna()]
    d61 = m[m.LOCATION_TYPE == 'ROAD']
    out = {'D84': m.distance_from_quarry_km.values,
           'D62': road_db.dist_km.values,
           'D61': d61.distance_from_quarry_km.values,
           'D56': d61[~d61.OBJECTID.isin(PROBABLE_DUPLICATES)].distance_from_quarry_km.values}
    return out, m, db
