from datetime import datetime
import numpy as np
import numexpr as ne
import pandas as pd
from skimage.util.shape import view_as_windows as viewW
import itertools
import logging
from numba import njit

logger = logging.getLogger(__name__)

def dt2cal(dt):
    """
    Convert array of datetime64 to a calendar array of year, month, day, hour,
    minute, seconds, microsecond with these quantites indexed on the last axis.

    Parameters
    ----------
    dt : datetime64 array (...)
        numpy.ndarray of datetimes of arbitrary shape

    Returns
    -------
    cal : uint32 array (..., 7)
        calendar array with last axis representing year, month, day, hour,
        minute, second, microsecond
    """

    # allocate output
    out = np.empty(dt.shape + (7,), dtype="u4")
    # decompose calendar floors
    Y, M, D, h, m, s = [dt.astype(f"M8[{x}]") for x in "YMDhms"]
    out[..., 0] = Y + 1970 # Gregorian Year
    out[..., 1] = (M - Y) + 1 # month
    out[..., 2] = (D - M) + 1 # dat
    out[..., 3] = (dt - D).astype("m8[h]") # hour
    out[..., 4] = (dt - h).astype("m8[m]") # minute
    out[..., 5] = (dt - m).astype("m8[s]") # second
    out[..., 6] = (dt - s).astype("m8[us]") # microsecond
    return out

def dt2cal(dt):
    """
    Convert array of datetime64 to a calendar array of year, month, day, hour,
    minute, seconds, microsecond with these quantites indexed on the last axis.

    Parameters
    ----------
    dt : datetime64 array (...)
        numpy.ndarray of datetimes of arbitrary shape

    Returns
    -------
    cal : uint32 array (..., 7)
        calendar array with last axis representing year, month, day, hour,
        minute, second, microsecond
    """

    # allocate output
    out = np.empty(dt.shape + (7,), dtype="u4")
    # decompose calendar floors
    Y, M, D, h, m, s = [dt.astype(f"M8[{x}]") for x in "YMDhms"]
    out[..., 0] = Y + 1970 # Gregorian Year
    out[..., 1] = (M - Y) + 1 # month
    out[..., 2] = (D - M) + 1 # dat
    #out[..., 3] = (dt - D).astype("m8[h]") # hour
    #out[..., 4] = (dt - h).astype("m8[m]") # minute
    #out[..., 5] = (dt - m).astype("m8[s]") # second
    #out[..., 6] = (dt - s).astype("m8[us]") # microsecond
    return out

def upper_columns_names(data):
    data.columns = [str(x).upper().strip() for x in data.columns.tolist()]
    return data


def first_sup_strict_zero(arr, axis, val, invalid_val=-1):
    mask = arr > val
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def first_sup_val(arr, axis, val=0, invalid_val=-1):
    mask = arr >= val
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def np_ffill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return arr[np.arange(idx.shape[0])[:, None], idx]

def np_bfill(arr):
    return np_ffill(arr[:, ::-1])[:, ::-1]

def roll_and_null(data_in, shift=1, val=0):
    data = np.roll(data_in, shift, axis=1)
    if shift >= 0:
        data[:, :shift] = val
    else:
        data[:, shift:] = val
    return data


def roll_and_null_axis0(data_in, shift=1, val=0):
    data = np.roll(data_in, shift, axis=0)
    if shift >= 0:
        data[:shift] = val
    else:
        data[shift:] = val
    return data


def roll_and_null_axis2(data_in, shift=1, val=0):
    data = np.roll(data_in, shift, axis=2)
    if shift >= 0:
        data[:, :, :shift] = val
    else:
        data[:, :, shift:] = val
    return data


def roll_and_null3D(data_in, shift=1, val=0):
    data = np.roll(data_in, shift, axis=2)
    if shift >= 0:
        data[:, :, :shift] = val
    else:
        data[:, :, shift:] = val
    return data


def first_nonzero(arr, axis, val, invalid_val=-1):
    mask = arr != val
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def first_nonzero_inf(arr, axis, val, invalid_val=-1):
    mask = arr <= val
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def first_nonzero_str_inf(arr, axis, val, invalid_val=-1):
    mask = arr < val
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def clean_df(df):
    df = 0
    del df


def clean_ldf(ldf):
    for df in ldf:
        df = 0
        del df


def strided_indexing_roll(a, r, rep_val=0, rep_nan=True, val_nan=np.nan):
    # Concatenate with sliced to cover all rolls
    p = np.full((a.shape[0], a.shape[1] - 1), val_nan)
    a_ext = np.concatenate((p, a, p), axis=1)

    # Get sliding windows; use advanced-indexing to select appropriate ones
    n = a.shape[1]
    outo = viewW(a_ext, (1, n))[np.arange(len(r)), -r + (n - 1), 0]
    if rep_nan:
        outo = np.nan_to_num(outo, nan=rep_val)

    return outo


def remove_leading_zeros(a):
    shift_vect = (a != 0).argmax(axis=1)
    a = strided_indexing_roll(a, -shift_vect)
    return a


def transform_date_in_num(data, col_date):
    data[col_date] = data[col_date].fillna(".").replace(".", "01/01/1900")
    def_value = nb_months_in_date(datetime(1900, 1, 1).date())
    data[col_date] = nb_months_in_date_pd(data[col_date], def_value=def_value, format="%d/%m/%Y")
    return data


def get_day_from_col_date(data, col_date, format="%d/%m/%Y"):
    return pd.to_datetime(data[col_date], format=format).dt.day


def nb_months_in_date_pd(col_date, def_value=22801, format="%d/%m/%Y"):
    col_date = pd.to_datetime(col_date, format=format).dt
    return np.array([x if x != def_value else 0 for x in 12 * col_date.year + col_date.month])


def nb_months_in_date(datum):
    return 12 * datum.year + datum.month


def PMT(tx, duree, value):
    filtero = (tx == 0)
    filtero = filtero.reshape(tx.shape)
    return np.where(filtero, (-1 * (value) / duree),
                    (tx * (value * (1 + tx) ** duree)) / ((1) * (1 - (1 + tx) ** duree)))


def IPMT(tx, mois, duree, value):
    ipmt = -(((1 + tx) ** (mois - 1)) * (value * tx + PMT(tx, duree, value)) - PMT(tx, duree, value, ))
    return (ipmt)


def PPMT(tx, mois, duree, value):
    ppmt = PMT(tx, duree, value) - IPMT(tx, mois, duree, value)
    return (ppmt)


def pmt_unit(tx, duree):
    filtero = ne.evaluate('tx == 0').reshape(tx.shape)
    return ne.evaluate('where(filtero, -  1/ duree, (tx * (1 + tx) ** duree) / (1 - (1 + tx) ** duree))')


def ipmt_unit(tx, mois, pmt_unit):
    ipmt = ne.evaluate('-((1 + tx) ** (mois - 1) * (tx + pmt_unit) - pmt_unit)')
    return ipmt


def ppmt(tx, mois, pmt_unit):
    ipmt = ipmt_unit(tx, mois, pmt_unit)
    ppmt = ne.evaluate('pmt_unit - ipmt')
    return ppmt


def sum_over_period(mtx, per, deb_sum):
    t = mtx.shape[1]
    n = mtx.shape[0]
    mtx = strided_indexing_roll(mtx, -deb_sum.reshape(n) + 1)
    if t % per != 0:
        mtx = np.concatenate([mtx, np.zeros((n, per - (t % per)))], axis=1)
    mtx_r = mtx.reshape(n, mtx.shape[1] // per, per).copy()
    mtx = mtx * 0
    mtx[:, ::per] = mtx_r.sum(axis=2)
    mtx = strided_indexing_roll(mtx, deb_sum.reshape(n) - 1)
    if t % per != 0:
        mtx = mtx[:, :-(per - (t % per))]
    return mtx


def calculate_ind_moyen(ind2, ind1, tombee, delta_days):
    ind_new = ind1.copy()
    ind1 = ind1[:, 1:].copy()
    ind2 = ind2[:, 1:].copy()
    delta_d = delta_days
    tombee = np.minimum(delta_days, tombee)
    ind_new[:, 1:] = ne.evaluate('(ind1 * tombee + ind2 * (delta_d - tombee)) / (delta_d)')
    return ind_new


def calculate_ind_moyen3D(ind2, ind1, tombee, delta_days):
    tombee = np.minimum(delta_days, tombee)
    ind_new = ne.evaluate('(ind1 * tombee + ind2 * (delta_days - tombee)) / (delta_days)')
    return ind_new

def build_current_month(n, t):
    current_month = np.arange(1, t + 1).reshape(1, t)
    return np.vstack([current_month] * n)


def add_months_date(date_arr, months_add, weeks=None, hours=None, minutes=None,
              seconds=None, milliseconds=None, microseconds=None, nanoseconds=None):
    years = year(date_arr)
    months = month(date_arr) + months_add,
    days = day(date_arr)
    years = np.asarray(years) - 1970
    months = np.asarray(months) - 1
    days = np.asarray(days) - 1
    types = ('<M8[Y]', '<m8[M]', '<m8[D]', '<m8[W]', '<m8[h]',
             '<m8[m]', '<m8[s]', '<m8[ms]', '<m8[us]', '<m8[ns]')
    vals = (years, months, days, weeks, hours, minutes, seconds,
            milliseconds, microseconds, nanoseconds)
    new_dates = sum(np.asarray(v, dtype=t) for t, v in zip(types, vals)
               if v is not None)
    new_dates = new_dates.reshape(new_dates.shape[1])
    return new_dates

def add_months_date2(date_arr, months_add, weeks=None, hours=None, minutes=None,
              seconds=None, milliseconds=None, microseconds=None, nanoseconds=None, is_array=False):
    years = year(date_arr)
    months = month(date_arr) + months_add,
    days = day(date_arr)
    years = np.asarray(years) - 1970
    if not is_array:
        months = np.asarray(months) - 1
    else:
        months = np.array(months)
    days = np.asarray(days) - 1
    types = ('<M8[Y]', '<m8[M]', '<m8[D]', '<m8[W]', '<m8[h]',
             '<m8[m]', '<m8[s]', '<m8[ms]', '<m8[us]', '<m8[ns]')
    vals = (years, months, days, weeks, hours, minutes, seconds,
            milliseconds, microseconds, nanoseconds)
    new_dates = sum(np.asarray(v, dtype=t) for t, v in zip(types, vals)
               if v is not None)
    new_dates = new_dates.reshape(date_arr.shape)
    return new_dates

def add_years_date(date_arr, years_add, weeks=None, hours=None, minutes=None,
              seconds=None, milliseconds=None, microseconds=None, nanoseconds=None):
    years = year(date_arr) + years_add
    months = month(date_arr)
    days = day(date_arr)
    years = np.asarray(years) - 1970
    months = np.asarray(months) - 1
    days = np.asarray(days) - 1
    types = ('<M8[Y]', '<m8[M]', '<m8[D]', '<m8[W]', '<m8[h]',
             '<m8[m]', '<m8[s]', '<m8[ms]', '<m8[us]', '<m8[ns]')
    vals = (years, months, days, weeks, hours, minutes, seconds,
            milliseconds, microseconds, nanoseconds)
    new_dates = sum(np.asarray(v, dtype=t) for t, v in zip(types, vals)
               if v is not None)
    return new_dates

def end_of_month_numpy(date_arr, weeks=None, hours=None, minutes=None,
              seconds=None, milliseconds=None, microseconds=None, nanoseconds=None):
    years = year(date_arr)
    months = month(date_arr)
    days = -7
    years = np.asarray(years) - 1970
    months = np.asarray(months) - 1
    days = np.asarray(days) - 1
    types = ('<M8[Y]', '<m8[M]', '<m8[D]', '<m8[W]', '<m8[h]',
             '<m8[m]', '<m8[s]', '<m8[ms]', '<m8[us]', '<m8[ns]')
    vals = (years, months, days, weeks, hours, minutes, seconds,
            milliseconds, microseconds, nanoseconds)
    new_dates = sum(np.asarray(v, dtype=t) for t, v in zip(types, vals)
               if v is not None)
    return new_dates

def add_days_date(date_arr, days_add, weeks=None, hours=None, minutes=None,
              seconds=None, milliseconds=None, microseconds=None, nanoseconds=None):
    years = year(date_arr)
    months = month(date_arr)
    days = day(date_arr) + days_add
    years = np.asarray(years) - 1970
    months = np.asarray(months) - 1
    days = np.asarray(days) - 1
    types = ('<M8[Y]', '<m8[M]', '<m8[D]', '<m8[W]', '<m8[h]',
             '<m8[m]', '<m8[s]', '<m8[ms]', '<m8[us]', '<m8[ns]')
    vals = (years, months, days, weeks, hours, minutes, seconds,
            milliseconds, microseconds, nanoseconds)
    new_dates = sum(np.asarray(v, dtype=t) for t, v in zip(types, vals)
               if v is not None)
    return new_dates

def year(dates):
    "Return an array of the years given an array of datetime64s"
    return dates.astype('M8[Y]').astype('i8') + 1970

def month(dates):
    "Return an array of the months given an array of datetime64s"
    return dates.astype('M8[M]').astype('i8') % 12 + 1

def day(dates):
    "Return an array of the days of the month given an array of datetime64s"
    return (dates - dates.astype('M8[M]')) / np.timedelta64(1, 'D') + 1


def explode_dataframe(data, columns):
    for col in columns:
        data[col] = data[col].str.rstrip(',').str.split(',')
    for col in columns:
        data = data.explode(column=[col])
    for col in columns:
        data[col] = data[col].str.strip()

    return data

@njit
def parse_date(date):
    for i, s in enumerate(date):
        if str(date[i] ).strip() != "." and int(str(date[i] ).split("/")[-1]) >= 2200:
            date[i] = "01/01/2200"
    return date




