"""Build the data extracts used in this note from the authors' public repository
(https://github.com/clipo/moai_walking, MIT licence; Zenodo doi:10.5281/zenodo.17329904).
Usage:  python code/prepare_data.py /path/to/moai_walking"""
import os
import shutil
import sys
import numpy as np
import pandas as pd
from common import DATA, QLAT, QLON


def main(repo):
    j = lambda *a: os.path.join(repo, *a)
    db = pd.read_excel(j('data', 'MOAI_DATABASE_PUBLIC.xlsx'))
    db['lat'] = pd.to_numeric(db.latitude, errors='coerce')
    db['lon'] = pd.to_numeric(db.longitude, errors='coerce')
    # same planar approximation as create_distance_dataset.R
    db['dist_km'] = np.sqrt(((db.lat - QLAT) * 111000) ** 2 + ((db.lon - QLON) * 99000) ** 2) / 1000
    cols = ['OBJECTID', 'SOURCE', 'ORIGINAL_SHEPARDSON_ID', 'EI_ATLAS_ID', 'ALTERNATIVE_GIS_ID', 'LOCATION_TYPE',
            'LOCATION_NAME', 'LOCATION_NOTES', 'STATE_FOUND', 'NUMBER_OF_FRAGMENTS', 'GENERAL_CONDITION',
            'TOTAL_LENGTH_cm', 'BASE_WIDTHcm', 'MATERIAL_TYPE', 'lat', 'lon', 'dist_km']
    ext = db[cols].copy()
    ext.to_csv(os.path.join(DATA, 'schumacher2013_extract.csv'), index=False)
    r = pd.read_csv(j('data', 'road_moai_distances.csv'))
    r.to_csv(os.path.join(DATA, 'road_moai_distances_published84.csv'), index=False)
    m = r.merge(ext, left_on=['latitude', 'longitude'], right_on=['lat', 'lon'], how='left')
    m['pref'] = (m.LOCATION_TYPE == 'ROAD').astype(int)
    m = m.sort_values(['latitude', 'longitude', 'pref'], ascending=[True, True, False]).drop_duplicates(['latitude', 'longitude'])
    keep = ['latitude', 'longitude', 'distance_from_quarry_km', 'location_type', 'OBJECTID', 'SOURCE', 'LOCATION_TYPE',
            'LOCATION_NAME', 'LOCATION_NOTES', 'STATE_FOUND', 'NUMBER_OF_FRAGMENTS', 'TOTAL_LENGTH_cm']
    m[keep].sort_values('distance_from_quarry_km').to_csv(os.path.join(DATA, 'road84_annotated.csv'), index=False)
    f13 = pd.read_csv(j('figures', 'Figure_13_analysis_data.csv'))
    f13['transport_phase'] = f13['transport_phase'].str.replace('\n', ' ')
    f13.to_csv(os.path.join(DATA, 'figure13_size_distance_published46.csv'), index=False)
    t = pd.read_excel(j('data', 'Table_1_road_moai_orientation.xlsx'), header=None)
    rows = []
    for lab, key in [('Face Down', 'face_down'), ('Face Up', 'face_up'), ('Lateral', 'lateral')]:
        rr = t[t.iloc[:, 1].astype(str).str.strip() == lab].iloc[0]
        rows.append(dict(position=key, downhill=int(rr.iloc[2]), flat=int(rr.iloc[3]), uphill=int(rr.iloc[4])))
    pd.DataFrame(rows).to_csv(os.path.join(DATA, 'table1_orientation_counts.csv'), index=False)
    prof = []
    for fn, name in [('moai_road_slope_from_raraku.xlsx', 'raraku_southcoast'), ('southcoast_road_only_slope.xlsx', 'southcoast')]:
        x = pd.read_excel(j('data', fn))
        x = x[['Distance', 'Elevation', 'SLength']].dropna(subset=['Distance', 'Elevation']).copy()
        x['road'] = name
        prof.append(x)
    pd.concat(prof).to_csv(os.path.join(DATA, 'road_slope_profiles.csv'), index=False)
    v = pd.read_excel(j('data', 'VanTilburgData.xlsx'))
    vt = pd.DataFrame(dict(statue_id=v.iloc[:, 0], location=v['Location'],
                           length=pd.to_numeric(v['Length:Statue'], errors='coerce'),
                           base_width=pd.to_numeric(v['Width:Base'], errors='coerce'),
                           shoulder_width=pd.to_numeric(v['Width:Shoulders'], errors='coerce')))
    vt.to_csv(os.path.join(DATA, 'vantilburg1986_extract.csv'), index=False)
    shutil.copy(j('data', 'SimplifiedMoai.obj'), os.path.join(DATA, 'SimplifiedMoai.obj'))
    print('extracts written to', DATA)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/home/claude/moai_walking')
