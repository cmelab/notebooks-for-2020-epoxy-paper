import signac
import numpy as np
import pandas as pd

import sys

from common import (
    getDiffusivities,
    fit_Tg_to_DiBenedetto,
    DiBenedetto,
    Fit_Diffusivity1,
)




def get_custom_ranges(cooling_method):
    if cooling_method == "quench":
        custom_ranges_l1 = {
            00.0: [0.1, 0.8],
            30.0: [0.1, 0.8],
            50.0: [0.1, 0.8],
            70.0: [0.1, 0.8],
        }
        custom_ranges_l2 = {
            00.0: [0.7, 1.2],
            30.0: [0.85, 1.4],
            50.0: [1.0, 1.8],
            70.0: [1.15, 2.5],
        }
    elif cooling_method == "anneal":
        custom_ranges_l1 = {
            00.0: [0.1, 0.8],
            30.0: [0.1, 0.8],
            50.0: [0.1, 0.8],
            70.0: [0.1, 0.8],
        }
        custom_ranges_l2 = {
            00.0: [0.7, 1.2],
            30.0: [0.85, 1.4],
            50.0: [1.0, 1.8],
            70.0: [1.15, 2.5],
        }
    else:
        raise ValueError(cooling_method + "is unknown")
    return custom_ranges_l1, custom_ranges_l2


def get_tg_data(data_path, df):
    project = signac.get_project(data_path)

    PROP_NAME = "bparticles"
    filter_saps = [0.0, 30.0, 50.0, 70.0]
    Tgs = []
    Tgs_tangent = []
    cure_percents = []
    Cure_Ts = []
    cooling_method = "quench"

    df_filtered = df[
        (df.quench_T <= 3.0)
        & (df.quench_T >= 0.1)
        & (df.CC_bond_angle != 109.5)
        & (df.cooling_method == cooling_method)
    ]
    for i, sap in enumerate(filter_saps):
        for j, (cooling_method, df_grp) in enumerate(df_filtered.groupby("cooling_method")):
            df_curing = df_grp[
                (df_grp.bond == False)
                & (df_grp.calibrationT == 305)
                & (df_grp.cooling_method == cooling_method)
                & (df_grp.stop_after_percent == sap)
            ]
            cure_percent = df_curing.cure_percent.mean()
            cure_percents.append(cure_percent)
            Ts, Ds = getDiffusivities(project, df_curing, name=PROP_NAME)
            Cure_Ts.append(Ts)
            # Pretty sure this helps with the fits
            mul_fact = 1000000
            Ds_scaled = Ds * mul_fact
            custom_ranges_l1, custom_ranges_l2 = get_custom_ranges(cooling_method)
            Tg, Tg_prop, line_vals = Fit_Diffusivity1(
                Ts,
                Ds_scaled,
                method="use_viscous_region",
                min_D=0,
                ver=4,
                viscous_line_index=0,
                l1_T_bounds=custom_ranges_l1[sap],
                l2_T_bounds=custom_ranges_l2[sap],
            )
            Tgs.append(Tg)
    Tgs = np.asarray(Tgs)
    cure_percents = np.asarray(cure_percents)

    cure_percents = np.asarray(cure_percents)
    Tgs = np.asarray(Tgs)
    Tgs_tangent = np.asarray(Tgs_tangent)
    cure_percents_ss = cure_percents
    Tgs_ss = Tgs
    R2, fit_Tgs, T1, inter_parm, T0 = fit_Tg_to_DiBenedetto(
        cure_percents_ss / 100.0, Tgs_ss, T1=None, T0=None
    )
    alphas = np.linspace(0, 1)
    fit_ydata = DiBenedetto(alphas, T1, T0=T0, inter_param=inter_parm)
    cure_percents = np.asarray(cure_percents)
    Tgs = np.asarray(Tgs)
    Tgs_tangent = np.asarray(Tgs_tangent)
    cure_percents_ss = cure_percents
    Tgs_ss = Tgs
    R2, fit_Tgs, T1, inter_parm, T0 = fit_Tg_to_DiBenedetto(
        cure_percents_ss / 100.0, Tgs_ss, T1=None, T0=None
    )
    alphas = np.linspace(0, 1)
    fit_ydata = DiBenedetto(alphas, T1, T0=T0, inter_param=inter_parm)

    return alphas, fit_ydata, R2, cure_percents, Tgs