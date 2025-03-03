import pandas as pd
import numpy as np

class Rate_Interpolator():

    def __init__(self):
        self.max_projection = 360

    def interpolate_curves(self, tx_params):
        dic_tx_swap = {}
        ND_DAYS_COL = "NB_DAYS"
        curves_df = tx_params["curves_df"]["data"].copy()
        CURVE_CODE = tx_params["curves_df"]["curve_code"]
        TENOR = tx_params["curves_df"]["tenor"]
        curves_df["NB_DAYS"] = curves_df[TENOR].map(tx_params["curves_df"]["maturity_to_days"])
        filter_nb_mat = (
                curves_df[[CURVE_CODE, TENOR]].groupby([CURVE_CODE]).transform(lambda x: x.count()) >= 1).values
        curves_to_interpolate = curves_df.loc[filter_nb_mat, CURVE_CODE].unique().tolist()
        curves_df = curves_df[curves_df[CURVE_CODE].isin(curves_to_interpolate)].copy()

        tenors = [x for x in ["1D"] + [str(i) + "M" for i in range(1, 12)] + [str(i) + "Y" for i in range(1, 31)]]
        curves_df = curves_df[curves_df[TENOR].isin(tenors)].copy()

        necessary_days = pd.DataFrame([1] + [30 * i for i in range(1, 361)], columns=[ND_DAYS_COL])
        cols_cms = tx_params["curves_df"]["cols"]

        max_cms = tx_params["curves_df"]["max_proj"]
        cols_sup = ["M" + str(i) for i in
                    range(max_cms + 1, self.max_projection + 1)]
        if len(cols_sup) > 0:
            curves_df_num_cols = pd.concat(
                [curves_df[cols_cms], pd.DataFrame(np.full((curves_df.shape[0], len(cols_sup)), np.nan), index=curves_df.index,
                                                   columns=cols_sup)], axis=1).ffill(axis=1)

            curves_df = pd.concat([curves_df[[x for x in curves_df.columns if x not in cols_cms]],
                                   curves_df_num_cols], axis=1)

        for name_curve in curves_df[CURVE_CODE].unique().tolist():
            rate_curve = curves_df[curves_df[CURVE_CODE] == name_curve].copy()
            merged_curve = necessary_days.merge(rate_curve, how='outer', on=ND_DAYS_COL)
            merged_curve = merged_curve.set_index(ND_DAYS_COL)
            merged_curve = merged_curve.sort_index()
            merged_curve = merged_curve[cols_cms + cols_sup].interpolate(method='index', limit_direction="forward")
            merged_curve = merged_curve[merged_curve.index.isin(necessary_days[ND_DAYS_COL].unique().tolist())].copy()
            merged_curve = merged_curve[cols_cms + cols_sup].bfill(axis=0)
            dic_tx_swap[name_curve]  = merged_curve.values

        return dic_tx_swap

