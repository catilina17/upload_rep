import pandas as pd
import numpy as np
import utils.general_utils as gu
from services.rate_services.params import tx_referential as gp, maturities_referential as mr
import traceback
import logging

logger = logging.getLogger(__name__)

global gb_list_curve_absent_mat
global gb_duplicated_curves
gb_list_curve_absent_mat = []
gb_duplicated_curves = []


def generate_calculated_curves(curves_df, calc_curves_ref, filtering_tenor_curves_ref, mapping_rate_code_curve):
    logger.info('      Génération des courbes de taux calculées')
    cols_num = ['M%s' % i for i in range(0, gp.NB_PROJ_TAUX + 1)]
    all_curves = curves_df.copy()
    all_curves[gp.CN_MATURITY + "_MONTHS"] = all_curves[gp.CN_MATURITY].replace(mr.maturity_to_months)
    n = all_curves.shape[0]
    all_curves[cols_num] = np.where((all_curves[gp.CN_TYPE] == "LIQ").values.reshape(n, 1),
                                    all_curves[cols_num].values / 100, all_curves[cols_num].values)
    all_curves = construct_filtering_tenor_curves(all_curves, filtering_tenor_curves_ref)
    all_curves = construct_calc_curves(all_curves, calc_curves_ref, mapping_rate_code_curve)
    courbes_excl = calc_curves_ref[calc_curves_ref["OUT"] == "NON"][gp.CN_CODE_COURBE].values.tolist()
    all_curves = all_curves[~all_curves[gp.CN_CODE_COURBE].isin(courbes_excl)].copy()
    all_curves = all_curves.drop([gp.CN_MATURITY + "_MONTHS"], axis=1)
    check_for_duplicates(all_curves)
    return all_curves


def generate_calculated_curves2(curves_df):
    return curves_df


def check_for_duplicates(all_curves):
    global gb_duplicated_curves
    duplicated = all_curves[[gp.CN_CODE_COURBE, gp.CN_MATURITY]].duplicated([gp.CN_CODE_COURBE, gp.CN_MATURITY])
    if duplicated.any():
        list_curves = all_curves.loc[duplicated, [gp.CN_CODE_COURBE, gp.CN_MATURITY]].drop_duplicates().values.tolist()
        if len([x for x in list_curves if x not in gb_duplicated_curves]) > 0:
            logger.warning("         Il y a des doublons dans vos courbes : %s " % list_curves)
            gb_duplicated_curves = gb_duplicated_curves + list_curves


def construct_filtering_tenor_curves(all_curves, filtering_tenor_curves_ref):
    cols_num = ['M%s' % i for i in range(0, gp.NB_PROJ_TAUX + 1)]
    for index, curve in filtering_tenor_curves_ref.iterrows():
        curve_df = mr.ALL_MATURITIES_DF.copy()
        curve_df = pd.concat(
            [curve_df, pd.DataFrame(np.full((curve_df.shape[0], len(cols_num)), np.nan), columns=cols_num)], axis=1)
        curve_df[gp.CN_MATURITY + "_MONTHS"] = curve_df[gp.CN_MATURITY].astype(str).map(mr.maturity_to_months)
        curve_intervals = generate_intervals(curve_df, curve["INTERVALLES TENORS"])
        curve_vals = eval(curve["VALEURS"])
        curve_df["M0"] = np.select(curve_intervals, curve_vals, default=0)
        curve_df[cols_num] = curve_df[cols_num].ffill(axis=1)
        curve_df = gu.add_constant_new_col_pd(curve_df, [gp.CN_CODE_COURBE], [curve[gp.CN_CODE_COURBE]])
        code_pass_alm = pd.DataFrame(curve[gp.CN_CODE_COURBE] + "_" + curve_df[gp.CN_MATURITY].astype(str).values,
                                     columns=[gp.CN_CODE_PASSALM])
        curve_df = pd.concat([curve_df.reset_index(drop=True), code_pass_alm.reset_index(drop=True)], axis=1)
        all_curves = pd.concat([all_curves, curve_df])
    return all_curves


def generate_intervals(curve_df, intervals):
    intervals = intervals.replace("*", str(30 * 12))
    intervals = eval(intervals)
    intervals_buckets = [(curve_df[gp.CN_MATURITY + "_MONTHS"] >= int(intv.split("-")[0]))
                         & (curve_df[gp.CN_MATURITY + "_MONTHS"] <= int(intv.split("-")[1])) for intv in intervals]
    return intervals_buckets


def complete_map_rate_code(mapping_rate_code_curve, list_maturities):
    list_global_mat = list_maturities + mapping_rate_code_curve["TENOR"].unique().tolist()
    list_global_mat = list(set([x for x in list_global_mat if x != "NA"]))
    filter_na = mapping_rate_code_curve["TENOR"] == "NA"
    mapping_na = mapping_rate_code_curve[filter_na]
    mapping_rest = mapping_rate_code_curve[~filter_na]
    s = len(list_global_mat)
    mapping_na = mapping_na.loc[np.repeat(mapping_na.index, s)]
    mapping_na["TENOR"] = np.tile(np.array(list_global_mat), len(mapping_na) // s)
    mapping_na["RATE_CODE"] = mapping_na["RATE_CODE"] + mapping_na["TENOR"]
    final_map = pd.concat([mapping_rest, mapping_na])
    final_map = final_map.drop_duplicates(["CURVE_NAME", "TENOR"])
    final_map = final_map.set_index(["CURVE_NAME", "TENOR"])
    final_map = final_map.rename(columns={"RATE_CODE": gp.CN_CODE_PASSALM, "CCY_CODE": gp.CN_CODE_DEVISE})
    return final_map


def construct_calc_curves(all_curves, pn_pricing_curves_ref, mapping_rate_code_curve):
    global gb_list_curve_absent_mat
    pn_pricing_curves_ref = pn_pricing_curves_ref.sort_values(["PRIORITE"])
    cols = ['M%s' % i for i in range(0, gp.NB_PROJ_TAUX + 1)]
    cols2 = ['M%s' % i for i in range(1, gp.NB_PROJ_TAUX + 1)]
    list_maturities = list(mr.maturity_to_months.keys())
    mapping_rate_code_curve = complete_map_rate_code(mapping_rate_code_curve, list_maturities)
    list_curve_absent_mat = []
    for index, curve in pn_pricing_curves_ref.iterrows():
        try:
            curves_l = eval(curve["COURBES SOUS-JACENTES"])
            if True:  # curve["CODE COURBE"] in ['LGBPASK']:
                mat_l = eval(curve["MATURITY"])
                underlying_curves = [all_curves[(all_curves[gp.CN_CODE_COURBE] == curve_name) & (
                    all_curves[gp.CN_MATURITY].isin(list_maturities))] if mat == "ALL"
                                     else all_curves[
                    (all_curves[gp.CN_CODE_COURBE] == curve_name) & (all_curves[gp.CN_MATURITY].isin([mat]))]
                                     for curve_name, mat in zip(curves_l, mat_l)]

                existing_maturities = [list_maturities if mat == "ALL" else [mat] for x, mat in
                                       zip(underlying_curves, mat_l)]

                for und_curve, name_curve, mat, i in zip(underlying_curves, curves_l, mat_l,
                                                         range(0, len(existing_maturities))):
                    if (len(und_curve) == 0 and mat != "ALL") | (
                            len(und_curve) != len(list_maturities) and mat == "ALL"):
                        existing_maturities[i] = [x for x in existing_maturities[i] if
                                                  x in und_curve[gp.CN_MATURITY].unique().tolist()]

                all_sizes_deb = [len(x) for x in underlying_curves]

                existing_maturities_diff_1 = [x for x, s in zip(existing_maturities, all_sizes_deb) if s != 1]
                if len(existing_maturities_diff_1) > 0:
                    existing_maturities = list(
                        set(existing_maturities_diff_1[0]).intersection(*existing_maturities_diff_1))

                    underlying_curves_final = [under_curve[under_curve[gp.CN_MATURITY].isin(existing_maturities)].copy()
                                               for under_curve, s in zip(underlying_curves, all_sizes_deb) if s != 1]
                else:
                    existing_maturities = [
                        sorted([(mat[0], mr.maturity_to_months[mat[0]]) for mat in existing_maturities],
                               key=lambda x: x[1])[0][0]]
                    if (str(curve["RESULT TENOR"]) != "" and curve["RESULT TENOR"] is not None) and set(mat_l) != {
                        "ALL"}:
                        existing_maturities = [curve["RESULT TENOR"]]

                    underlying_curves_final = underlying_curves

                all_sizes = [len(x) for x in underlying_curves_final]
                min_size = min(all_sizes)
                not_compatible = (len(set(all_sizes)) > 2) or (len(set(all_sizes)) == 2 and 1 not in all_sizes)

                if not_compatible:
                    logger.warning("      Les courbes sous-jacentes %s ne sont pas compatible" % curves_l)

                if min_size == 0:
                    list_curve_absent_mat = list_curve_absent_mat + [curve["CODE COURBE"]]

                if min_size > 0 and not not_compatible:
                    underlying_curves_sorted = [x.sort_values(gp.CN_MATURITY + "_MONTHS")[cols] for x in
                                                underlying_curves_final]
                    fill_value_lag = [0 if lag_fill == 0 else np.nan for lag_fill in eval(curve["LAG FILL"])]
                    curves_lagged = [curve[cols].shift(lag, fill_value=fill_value, axis=1)
                                     for curve, lag, fill_value in zip(underlying_curves_sorted, eval(curve["LAG"]),
                                                                       fill_value_lag)]
                    curves_lagged_filled = [curve.bfill(axis=1) if lag_val == 'LKV' else curve
                                            for curve, lag_val in zip(curves_lagged, eval(curve["LAG FILL"]))]
                    if sum(eval(curve["MOYENNE MOBILE"])) != 0:
                        curves_wa = [calculate_weighted_average(curve[cols2], w) if m != 0 else curve[cols2]
                                     for curve, m, w in zip(curves_lagged_filled, eval(curve["MOYENNE MOBILE"]),
                                                            eval(curve["POIDS MOYENNE MOBILE"]))]
                    else:
                        curves_wa = curves_lagged_filled.copy()
                    real_operation = curve["OPERATION"]
                    for i in range(1, len(curves_l) + 1):
                        real_operation = real_operation.replace("C%s" % i, "np.array(curves_wa[%s])" % (i - 1))
                    real_operation = real_operation.replace("round", "np.round")
                    real_operation = real_operation.replace("maximum", "np.maximum")
                    real_operation = real_operation.replace("minimum", "np.minimum")
                    curve_df = eval(real_operation)

                    if sum(eval(curve["MOYENNE MOBILE"])) != 0:
                        curve_df = pd.DataFrame(curve_df, columns=cols2)
                        curve_df["M0"] = curve_df["M1"].values
                        curve_df = curve_df[cols].copy()
                    else:
                        curve_df = pd.DataFrame(curve_df, columns=cols)

                    curve_df = gu.add_constant_new_col_pd(curve_df, \
                                                          [gp.CN_CODE_COURBE, gp.CN_TYPE_COURBE,
                                                           gp.CN_TYPE2_COURBE], \
                                                          [curve[gp.CN_CODE_COURBE],
                                                           curve[gp.CN_TYPE_COURBE],
                                                           curve[gp.CN_TYPE2_COURBE]])

                    curve_df[gp.CN_MATURITY] = sorted(existing_maturities, key=lambda x: mr.maturity_to_months[x])

                    final_curve_df = curve_df.join(mapping_rate_code_curve, on=[gp.CN_CODE_COURBE, gp.CN_MATURITY])
                    if len(final_curve_df) > len(curve_df):
                        msg = " Il y a un problème de doublons dans le mapping RATE_CODE-DEVISE_CURVE_NAME-TENOR"
                        logger.error(msg)
                        raise ValueError(msg)

                    final_curve_df[gp.CN_CODE_PASSALM] = np.where(final_curve_df[gp.CN_CODE_PASSALM].isnull(),
                                                                  final_curve_df[gp.CN_CODE_COURBE]
                                                                  + final_curve_df[gp.CN_MATURITY],
                                                                  final_curve_df[gp.CN_CODE_PASSALM])

                    final_curve_df[gp.CN_CODE_DEVISE] = np.where(final_curve_df[gp.CN_CODE_DEVISE].isnull(),
                                                                 curve["DEVISE"],
                                                                 final_curve_df[gp.CN_CODE_DEVISE])

                    final_curve_df = final_curve_df.drop_duplicates([gp.CN_CODE_PASSALM, gp.CN_CODE_DEVISE])

                    all_curves = pd.concat([all_curves, final_curve_df])

        except Exception:
            logger.warning("          Il y a eu un problème avec le calcul de la courbe %s" % curve[gp.CN_CODE_COURBE])
            logger.warning(traceback.format_exc())

    if len([x for x in list(set(list_curve_absent_mat)) if x not in gb_list_curve_absent_mat]) > 0:
        logger.warning(
            "          Impossible de calculer les courbes suivantes car certaines de leurs courbes sous-jacentes sont absentes:")
        logger.warning("          %s" % list(set(list_curve_absent_mat)))
        gb_list_curve_absent_mat = gb_list_curve_absent_mat + list_curve_absent_mat

    return all_curves


def calculate_weighted_average(x, w):
    list_mv = []
    mv_x = x.T.rolling(len(w)).apply(lambda x: (x * w[::-1]).sum() / (np.array(w[::-1]).sum())).T
    for j in range(0, len(w) - 1):
        mv_x.iloc[:, j] = (x.iloc[:, :j + 1] * w[:j + 1][::-1]).sum(axis=1) / (np.array(w[:j + 1][::-1]).sum())
    list_mv.append(mv_x)
    return np.vstack(list_mv)
