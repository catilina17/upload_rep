import re
import numpy as np
import logging
import pandas as pd
from . import tx_referential as gp
from . import maturities_referential as mref

logger = logging.getLogger(__name__)

not_interpolated_all = []
def refactor_curves(referential, curves_df, curves_to_interpolate):
    curves_df = curves_df.ffill(axis='columns')
    num_cols = ['M{}'.format(i) for i in range(gp.NB_PROJ_TAUX + 1)]
    curves_df.columns = num_cols
    completed_curves = join_curves_to_referential(referential, curves_df)
    completed_curves = add_missing_maturities(completed_curves, curves_to_interpolate)
    completed_curves[gp.CN_TYPE2_COURBE] = completed_curves[gp.CN_TYPE2_COURBE].ffill()
    completed_curves[num_cols] = completed_curves[num_cols].fillna(0)
    return completed_curves


def join_curves_to_referential(referential_df, curves_df):
    ref_pass_alm_df = referential_df.set_index(gp.CN_CODE_PASSALM)
    completed_rate_curve = ref_pass_alm_df.merge(curves_df, left_index=True, right_index=True, how='right')
    return completed_rate_curve


def add_missing_maturities(curves_df, curves_to_interpolate):
    global not_interpolated_all
    curves_df = curves_df.reset_index().copy()
    curves_df[gp.CN_CODE] = curves_df[gp.CN_CODE_PASSALM].apply(lambda x: get_pass_alm_curve_code(x))
    order = curves_df[gp.CN_CODE_COURBE].unique()
    list_curves = curves_to_interpolate.iloc[:, 0].str.strip().values.tolist()
    curves_df[gp.CN_CODE_COURBE] = curves_df[gp.CN_CODE_COURBE].str.strip()
    result = interpolate_missing_maturities(mref.ALL_MATURITIES_DF, curves_df, list_curves)
    interpolated = curves_df[curves_df[gp.CN_CODE_COURBE].isin(list_curves)][gp.CN_CODE_COURBE].unique()
    not_interpolated = [x for x in list_curves if x not in interpolated]
    if len([x for x in not_interpolated if x not in not_interpolated_all]) > 0:
        logger.warning("        Certains courbes n'ont pas pu être interpolées car absentes : %s" % not_interpolated)
        not_interpolated_all = not_interpolated_all + not_interpolated
    result = result[~result[gp.CN_CODE_PASSALM].isna()]
    result.reset_index(inplace=True, drop=True)
    final_result = pd.DataFrame(order, columns=[gp.CN_CODE_COURBE]).merge(result)
    return final_result.drop([gp.CN_CODE], axis=1)


def interpolate_missing_maturities(complete_maturities, curves_df, curves_to_interpolate):
    list_curves =[x for x in curves_df[gp.CN_CODE_COURBE].unique().tolist() if x not in ["nan", np.nan]]
    for name_curve in list_curves:
        rate_curve = curves_df[curves_df[gp.CN_CODE_COURBE] == name_curve].copy()
        if rate_curve[gp.CN_CODE_COURBE].iloc[0] in curves_to_interpolate:
            list_exiting_codes = rate_curve[gp.CN_CODE_PASSALM].unique().tolist()
            merged_curve = complete_maturities.merge(rate_curve, how='outer')
            merged_curve["INDEX"] = merged_curve[gp.CN_MATURITY].map(mref.maturity_to_days2).astype(float)
            merged_curve = merged_curve.set_index("INDEX")
            merged_curve = merged_curve.sort_index()
            merged_curve.loc[:, [gp.CN_CODE, gp.CN_CODE_DEVISE, gp.CN_TYPE, gp.CN_CODE_COURBE]] = merged_curve.loc[:,
                                                                                      [gp.CN_CODE, gp.CN_CODE_DEVISE, gp.CN_TYPE,
                                                                                       gp.CN_CODE_COURBE]].ffill(axis=0).bfill(axis=0)
            new_code = merged_curve[gp.CN_CODE] + merged_curve[gp.CN_MATURITY]
            new_code = pd.Series(np.where(new_code.isin(list_exiting_codes), new_code + "_INT", new_code))
            merged_curve[gp.CN_CODE_PASSALM] = np.where(merged_curve[gp.CN_CODE_PASSALM].isnull(), new_code,
                                                        merged_curve[gp.CN_CODE_PASSALM].values)
            merged_curve.loc[:, "M0":"M" + str(gp.NB_PROJ_TAUX)]\
                = merged_curve.loc[:, "M0":"M" + str(gp.NB_PROJ_TAUX)].interpolate(method='index', limit_direction="both")

            curves_df = curves_df[curves_df[gp.CN_CODE_COURBE] != name_curve].copy()
            curves_df = pd.concat([curves_df, merged_curve])

    return curves_df


def get_pass_alm_curve_code(x):
    try:
        if 'UNSECURED E1D' in x:
            return 'UNSECURED E1D '
        if 'UNSECURED E3M' in x:
            return 'UNSECURED E3M '
        if 'SECURED E3M' in x:
            return 'SECURED E3M '
        if 'TCI EUR' in x:
            return 'TCI EUR '
        if 'UNSECURED U3M' in x:
            return 'UNSECURED U3M '
        if 'TCI USD' in x:
            return 'TCI USD '
        if '-F' in x:
            return 'F-' + re.match(r"^[A-Z ]+", x).group(0)
        return re.match(r"^[A-Z ]+", x).group(0)
    except:
        return x

