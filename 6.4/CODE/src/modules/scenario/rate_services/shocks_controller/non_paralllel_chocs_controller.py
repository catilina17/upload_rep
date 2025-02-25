import numpy as np
from ..tx_referential import *
from ..maturities_referential import *

def generate_twist_rate_curve_shocks(referential, currency_code, curve_code, start_pivot_threshold, end_pivot_threshold, start_pivot_shock, end_pivot_shock, debut, fin):
    start_pivot_threshold = maturity_to_days[start_pivot_threshold]
    end_pivot_threshold = maturity_to_days[end_pivot_threshold]

    referential = referential.loc[(referential[CN_CODE_COURBE] == curve_code) & (referential[CN_CODE_DEVISE] == currency_code) & \
        (referential[CN_TYPE2_COURBE] == "SWAP")]

    curve_maturities = list(referential[CN_MATURITY])
    curve_maturities = [maturity_to_days.get(item, item) for item in curve_maturities]

    interpolated_shocks = get_interpolated_shocks(end_pivot_shock, end_pivot_threshold, curve_maturities, start_pivot_shock, start_pivot_threshold)

    shocks = referential[[CN_CODE_PASSALM, CN_MATURITY, CN_CODE_DEVISE]].merge(interpolated_shocks)[[CN_CODE_PASSALM, CN_SHOCK, CN_CODE_DEVISE]]

    shocks = format_as_shocks_list(shocks, debut, fin)
    return shocks


def format_as_shocks_list(shocks, debut, fin):
    if debut == 0 and fin == 0:
        debut = 1
        fin = 240
    shocks.insert(0, CN_TYPE, CONSTANT_SHOCK_TYPE)
    shocks.insert(3, CN_STEP, 0)
    shocks.insert(4, CN_START, debut)
    shocks.insert(5, CN_END, fin)
    return shocks


def get_interpolated_shocks(end_pivot_shock, end_pivot_threshold, curve_maturities, start_pivot_shock, start_pivot_threshold):

    shocks_by_maturity = pd.DataFrame(columns=[CN_MATURITY, CN_SHOCK])
    shocks_by_maturity[CN_MATURITY] = curve_maturities
    shocks_by_maturity.loc[shocks_by_maturity[CN_MATURITY] < start_pivot_threshold, CN_SHOCK] = start_pivot_shock
    shocks_by_maturity.loc[shocks_by_maturity[CN_MATURITY] > end_pivot_threshold, CN_SHOCK] = end_pivot_shock

    x = [item for item in curve_maturities if start_pivot_threshold <= item <= end_pivot_threshold]
    interpolated_points = np.interp(x, [start_pivot_threshold, end_pivot_threshold], [start_pivot_shock, end_pivot_shock])

    interpolated_shocks_1 = pd.DataFrame({CN_MATURITY: x, CN_SHOCK: interpolated_points}).set_index(CN_MATURITY)
    shocks_by_maturity.set_index(CN_MATURITY, inplace=True)
    shocks_by_maturity.update(interpolated_shocks_1)
    shocks_by_maturity.reset_index(inplace=True)
    shocks_by_maturity.replace({CN_MATURITY: days_to_maturity}, inplace=True)
    return shocks_by_maturity