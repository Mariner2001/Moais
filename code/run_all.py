"""Run every analysis, then write figures and LaTeX tables.  Usage: python code/run_all.py"""
import time
import s1_reproduction, distance_models, gof, survival, breakage, copula_size_distance, terrain_orientation, mechanics, morphometrics

if __name__ == '__main__':
    for mod in [s1_reproduction, distance_models, gof, survival, breakage, copula_size_distance, terrain_orientation, mechanics, morphometrics]:
        t0 = time.time(); print(f'\n################ {mod.__name__}'); mod.run(); print(f'[{time.time() - t0:.1f} s]')
    import make_figures, make_tables
    make_figures.run(); make_tables.run()
