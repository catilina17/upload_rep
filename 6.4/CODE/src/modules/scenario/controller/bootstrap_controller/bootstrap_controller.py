import numpy as np
import modules.scenario.rate_services.maturities_referential as tm
from utils.excel_utils import get_dataframe_from_range
from dateutil.relativedelta import *
from modules.scenario.referentials.general_parameters import *
import pandas as pd
from modules.scenario.rate_services import tx_referential as tx_ref
from modules.scenario.parameters import user_parameters as up

YEAR_NUMBER_OF_DAY_365 = 365
YEAR_NBR_OF_DAYS_360 = 360

import calendar
import datetime as dt


def get_holidays_dates(wb):
    holidays_df = get_dataframe_from_range(wb, '_euro_holidays', False)
    holidays_df.drop(0, inplace=True)
    holidays_df[0] = holidays_df[0].astype(str)
    holidays_df[0] = holidays_df[0].str.split(n=1, expand=True)[0]
    my_holidays = np.array([dt.datetime.strptime(x, '%Y-%m-%d') for x in holidays_df[0]], dtype='datetime64[D]')
    return my_holidays


def days360(start_date, end_date, method_eu=False):
    start_day = start_date.day
    start_month = start_date.month
    start_year = start_date.year

    end_day = end_date.day
    end_month = end_date.month
    end_year = end_date.year

    if (
            start_day == 31 or
            (
                    method_eu is False and
                    start_month == 2 and (
                            start_day == 29 or (
                            start_day == 28 and
                            calendar.isleap(start_year) is False
                    )
                    )
            )
    ):
        start_day = 30

    if end_day == 31:
        if method_eu is False and start_day != 30:
            end_day = 1

            if end_month == 12:
                end_year += 1
                end_month = 1
            else:
                end_month += 1
        else:
            end_day = 30

    return (
            end_day + end_month * 30 + end_year * 360 -
            start_day - start_month * 30 - start_year * 360
    )


def find_next_busday(x, bus_day_calendar):
    while ~np.all(np.is_busday(x, busdaycal=bus_day_calendar)):
        x = np.array(x, dtype='datetime64[D]')
        result = np.where(np.is_busday(x, busdaycal=bus_day_calendar), np.timedelta64(0, 'D'), np.timedelta64(1, 'D'))
        x = x + result
    return x


def convert_to_pd_datetime(x):
    return x.astype(dt.datetime)


def convert_to_numpy_datetime(x):
    return np.array(x, dtype='datetime64[D]')


def get_days_forward(dar, maturities_relative_day, busdaycalendar):
    boot_days = np.array(
        [dar + dt.timedelta(days=1) + relativedelta(months=0)] + [dar + dt.timedelta(days=1) + relativedelta(months=i)
                                                                  for i in range(0, NB_PROJ_ZC)])
    boot_days = np.array(boot_days, dtype='datetime64[D]')

    jj_days = boot_days + [np.timedelta64(1, 'D') for x in range(0, NB_PROJ_ZC + 1)]
    jj_days = find_next_busday(jj_days, busdaycalendar)

    tn_days = jj_days + [np.timedelta64(1, 'D') for x in range(0, NB_PROJ_ZC + 1)]
    tn_days = find_next_busday(tn_days, busdaycalendar)

    tn_days = convert_to_pd_datetime(tn_days)

    swap_dates = np.array([tn_days, ] * (len(tm.ZC_MATURITIES_NAMES) - 2), ) + np.array(maturities_relative_day).reshape((len(tm.ZC_MATURITIES_NAMES) - 2), 1)

    swap_dates = np.vstack((jj_days, tn_days, swap_dates))

    swap_dates = convert_to_numpy_datetime(swap_dates)
    swap_dates = find_next_busday(swap_dates, busdaycalendar)

    swap_day = swap_dates - np.array(boot_days, dtype='datetime64[D]')

    nbr_swap_days = swap_day / np.timedelta64(1, 'D')

    nbr_swap_days = np.vstack((np.zeros(NB_PROJ_ZC + 1), nbr_swap_days[1:]))
    return nbr_swap_days, pd.DataFrame(swap_dates)


def interpolate_2d_array_columns_by_column(x, y, z):
    s = []
    for j in range(y.shape[1]):
        xp = list(map(float, y[:, j]))
        yp = list(map(float, z[:, j]))
        result = np.interp(x[j], xp, yp)
        s = np.append(s, result)
    return np.array(s, dtype=np.float64)


def get_maturities_relative_day():
    return [relativedelta(days=7)] + [relativedelta(months=x) for x in range(1, 12)] + [relativedelta(years=x) for x in
                                                                                        range(1, 31)]


def bootstrap(dev, curve, days_fwd, forward_dates, scenario_tx_df):
    #Step 1: Retrieve Rate Data'
    rate_df = get_currency_rate_curves(dev, curve, scenario_tx_df)[:, :NB_PROJ_ZC + 1]

    #Step 2: Calculate the Overnight (TN) Discount Factor
    # DF(TN) = 1 / (1 + days_fwd * Rate[0] /360
    tn_discount_factor = np.ones(NB_PROJ_ZC + 1) / (1 + days_fwd[1] * rate_df[0] / YEAR_NBR_OF_DAYS_360)

    #Step 2: Calculate Discount Factors for Short-Term Maturities
    # DF = DF(TN) / (1 + (days_fwd - days_TN) /360
    discount_factor = tn_discount_factor / (1 + (days_fwd[2:len(tm.ZC_MATURITIES_NAMES_INF_1Y)] - days_fwd[1])
                                            * rate_df[2:len(tm.ZC_MATURITIES_NAMES_INF_1Y)] / YEAR_NBR_OF_DAYS_360)
    discount_factor = np.vstack((np.ones(NB_PROJ_ZC + 1), tn_discount_factor, discount_factor))

    #Step 5: Calculate Zero-Coupon Rates for short-term maturities
    # N * (1 + ZC)**(nb_days/360) = N * DF
    zc = np.power(discount_factor[1:len(tm.ZC_MATURITIES_NAMES_INF_1Y)],
                  (-YEAR_NUMBER_OF_DAY_365 / (days_fwd[1:len(tm.ZC_MATURITIES_NAMES_INF_1Y)]))) - 1
    zc = np.vstack((rate_df[1], zc))

    #Step 6: Handle Long-Term Maturities
    forward_dates = forward_dates.values
    swap_days = ((forward_dates - forward_dates[1, :]) / np.timedelta64(1, 'D'))
    swap_accrual = get_swap_accruals_2(forward_dates)
    swap_accrual = pd.DataFrame(swap_accrual)
    someprod = np.zeros(NB_PROJ_ZC + 1)
    for i in range(len(tm.ZC_MATURITIES_NAMES_INF_1Y) - 1, len(tm.ZC_MATURITIES_NAMES)):
        if i > len(tm.ZC_MATURITIES_NAMES_INF_1Y) - 1:
            discount_factor = np.vstack((discount_factor,
                                         (discount_factor[1] * (1 - someprod * rate_df[i] / YEAR_NBR_OF_DAYS_360)) / (
                                                     1 + rate_df[i] * swap_accrual.loc[i, :] / YEAR_NBR_OF_DAYS_360),))
            zc = np.vstack((zc, np.power(discount_factor[i], (-YEAR_NUMBER_OF_DAY_365 / days_fwd[i])) - 1))

        temp = (1 + interpolate_2d_array_columns_by_column(swap_days[i], days_fwd[:i + 1, :], zc)) ** (
                    -swap_days[i] / YEAR_NUMBER_OF_DAY_365)
        someprod = someprod + temp * swap_accrual.loc[i, :]
    return days_fwd, zc


def get_forward_days_and_dates():
    busdaycalendar = np.busdaycalendar(holidays=up.holidays_list)
    maturities_relative_day = get_maturities_relative_day()
    days_fwd, forward_dates = get_days_forward(up.dar, maturities_relative_day, busdaycalendar)
    return days_fwd, forward_dates


def get_currency_rate_curves(dev, curve, scenario_tx_df):
    filter_curve = (scenario_tx_df[tx_ref.CN_DEVISE] == dev) & (scenario_tx_df[tx_ref.CN_CODE_COURBE] == curve)
    filter_curve_mat = filter_curve & (scenario_tx_df[tx_ref.CN_MATURITY].isin(list(tm.curves_maturities_needed_for_bootstrap)))
    if not filter_curve_mat.any():
        raise ValueError("There no is curve in the referential with %s='%s' and %s='%s'" % (
        tx_ref.CN_DEVISE, dev, tx_ref.CN_CODE_COURBE, curve))
    rate_df = (scenario_tx_df[filter_curve].set_index(tx_ref.CN_MATURITY).loc[np.array(tm.curves_maturities_needed_for_bootstrap), 'M0':] / 100).values
    return rate_df


def get_swap_accruals_2(forward_dates):
    forward_shift = np.copy(forward_dates)

    forward_shift[1:, :] = forward_dates[0:-1, :]
    forward_shift[len(tm.ZC_MATURITIES_NAMES_INF_1Y) - 1, :] = forward_dates[1, :]

    end_years = forward_dates.astype(str).astype('<U4').astype(int)
    end_month = (forward_dates.astype('datetime64[M]') - forward_dates.astype('datetime64[Y]') + 1).astype(int)
    end_day = (forward_dates.astype('datetime64[D]') - forward_dates.astype('datetime64[M]') + 1).astype(int)

    start_years = forward_shift.astype(str).astype('<U4').astype(int)
    start_month = (forward_shift.astype('datetime64[M]') - forward_shift.astype('datetime64[Y]') + 1).astype(int)
    start_day = (forward_shift.astype('datetime64[D]') - forward_shift.astype('datetime64[M]') + 1).astype(int)

    start_day = np.where((start_day == 31) | (start_month == 2) & (start_day == 29), 30, start_day)
    start_day = np.where(
        ((start_day == 28) & ((start_years % 4 == 0) & ((start_years % 100) != 0 | (start_years % 400 == 0)))), 30,
        start_day)
    end_day = np.where(((end_day == 31) & (start_day == 30)), 30, end_day)
    end_day = np.where(((end_day == 31) & (start_day != 30)), 1, end_day)
    end_month = np.where(((end_day == 31) & (start_day != 30) & (end_years != 12)), end_month + 1, end_month)
    end_years = np.where(((end_day == 31) & (start_day != 30) & (end_years == 12)), end_years + 1, end_years)
    end_month = np.where(((end_day == 31) & (start_day != 30) & (end_years == 12)), 1, end_month)
    result = end_day + end_month * 30 + end_years * 360 - start_day - start_month * 30 - start_years * 360
    return result
