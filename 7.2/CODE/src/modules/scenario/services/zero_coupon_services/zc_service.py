import os.path
import pandas as pd
import numpy as np
import logging
from modules.scenario.utils import paths_resolver as pr
from modules.scenario.services.rates_services.rate_temp_files_saver import get_temp_tx_files_path
from mappings import general_mappings as mp
from services.rate_services.params import maturities_referential as tm
from dateutil.relativedelta import *
from modules.scenario.parameters.general_parameters import *
from services.rate_services.params import tx_referential as tx_ref

YEAR_NUMBER_OF_DAY_365 = 365
YEAR_NBR_OF_DAYS_360 = 360

import calendar
import datetime as dt

logger = logging.getLogger(__name__)

non_dispo_curves_all = []


class ZCGenerator():

    def __init__(self, cls_usr):
        self.up = cls_usr
        self.non_dispo_curves_all = []

    def process_zero_coupons(self, scenario_tx_df, scenario_name, scenario_baseline_name, scen_orig_name):
        logger.info('      Calcul des Zéros Coupons')
        result = pd.DataFrame()
        courves_a_zc = mp.mapping_taux["COURBES A ZC"]
        days_fwd, forward_dates = self.get_forward_days_and_dates()

        dispo_curves = self.get_available_curves(scenario_tx_df, courves_a_zc)

        for dev, curve in zip(courves_a_zc[tx_ref.CN_DEVISE], courves_a_zc[tx_ref.CN_CODE_COURBE]):
            if curve in dispo_curves:
                df = self.create_curve_zc_df(curve, days_fwd, forward_dates, scenario_tx_df)
                df.insert(0, 'Courbe', curve)
                result = pd.concat([result, df])

        file_path = get_temp_tx_files_path('{}_{}'.format(scenario_baseline_name, scenario_name), TEMP_DIR_BOOTSRAP)
        result = self.replace_zc_for_eve_simul(result, scen_orig_name, courves_a_zc)
        result.to_csv(file_path, index=False)

    def create_curve_zc_df(self, curve, days_fwd, forward_dates, scenario_tx_df):
        days_fwd, zc = self.bootstrap_curve(curve, days_fwd, forward_dates, scenario_tx_df)
        row_a, col_a = np.shape(days_fwd)
        row_b, col_b = np.shape(zc)
        result = np.ravel([days_fwd.T, zc.T], order="F").reshape(row_a, col_a + col_b)
        currency_df = pd.DataFrame(result, index=tm.ZC_MATURITIES_NAMES)
        currency_df.reset_index(inplace=True)
        return currency_df

    def get_available_curves(self, scenario_tx_df, courves_a_zc):
        dispo_curves = scenario_tx_df[scenario_tx_df[tx_ref.CN_CODE_COURBE].str.strip().isin(
            courves_a_zc[tx_ref.CN_CODE_COURBE].values.tolist())][tx_ref.CN_CODE_COURBE].unique().tolist()
        non_dispo_curves = [x for x in courves_a_zc[tx_ref.CN_CODE_COURBE].values.tolist() if x not in dispo_curves]
        if len([x for x in non_dispo_curves if x not in self.non_dispo_curves_all]) > 0:
            logger.warning("         Les zéros-coupons ne pourront pas être calculés car"
                           " les courbes suivantes sont absentes: %s" % non_dispo_curves)
            self.non_dispo_curves_all = self.non_dispo_curves_all + non_dispo_curves

        return dispo_curves

    def replace_zc_for_eve_simul(self, boostrap_data, scenario_name, CURVES_TO_BOOTSTRAP):
        filter_curves = boostrap_data["Courbe"].isin(
            CURVES_TO_BOOTSTRAP[CURVES_TO_BOOTSTRAP["ZC FICHIER"] == "OUI"][tx_ref.CN_CODE_COURBE].values.tolist())
        bootstrap_replace = boostrap_data[filter_curves].copy()
        if self.up.zc_file_path == "" or not os.path.exists(self.up.zc_file_path):
            logger.warning("         Le fichier ZC n'est pas disponible")
            logger.warning("         Le processus se poursuit avec les ZC générés par PASS-ALM")
            return boostrap_data
        zc_data = pr.get_zc_data(self.up.zc_file_path)
        cols_sc = [x for x in zc_data.columns if scenario_name == "_".join(x.split("_")[:-2])]
        if cols_sc == []:
            msg_err = "         Le scénario %s n'est pas disponible dans le fichier ZC" % scenario_name
            logger.warning(msg_err)
            logger.warning("         Le processus se poursuit avec les Zéro Coupons générés par PASS-ALM")

        zc_data = zc_data[cols_sc].copy()
        all_months = list(set([int(x.split("_")[-2][1:]) for x in zc_data.columns]))
        for m in all_months:
            mois = "M" + str(m)
            zc_data_m = zc_data[[x for x in zc_data.columns if mois == x.split("_")[-2]]].copy()
            if zc_data_m.shape[1] > 2:
                logger.warning(
                    "         Le fichier ZC contient plusieurs colonnes pour le mois %s et le scénario %s" % (
                        mois, scenario_name))
                logger.warning("         Les ZC pour le mois % ne seront pas remplacés" % mois)
            else:
                bootstrap_replace.iloc[:, 2 * m + 2] = zc_data_m.iloc[:, 0].values
                bootstrap_replace.iloc[:, 2 * m + 3] = zc_data_m.iloc[:, 1].values

        boostrap_data[filter_curves] = bootstrap_replace
        return boostrap_data

    def days360(self, start_date, end_date, method_eu=False):
        start_day, start_month, start_year = start_date.day, start_date.month, start_date.year
        end_day, end_month, end_year = end_date.day, end_date.month, end_date.year

        if start_day == 31 or (not method_eu and start_month == 2 and (
                start_day == 29 or (start_day == 28 and not calendar.isleap(start_year)))):
            start_day = 30

        if end_day == 31:
            if not method_eu and start_day != 30:
                end_day, end_month = 1, (end_month + 1 if end_month < 12 else 1)
                if end_month == 1: end_year += 1
            else:
                end_day = 30

        return (end_day + end_month * 30 + end_year * 360 - start_day - start_month * 30 - start_year * 360)

    def find_next_busday(self, x, bus_day_calendar):
        while ~np.all(np.is_busday(x, busdaycal=bus_day_calendar)):
            x = np.array(x, dtype='datetime64[D]')
            result = np.where(np.is_busday(x, busdaycal=bus_day_calendar), np.timedelta64(0, 'D'),
                              np.timedelta64(1, 'D'))
            x = x + result
        return x

    def convert_to_pd_datetime(self, x):
        return x.astype(dt.datetime)

    def convert_to_numpy_datetime(self, x):
        return np.array(x, dtype='datetime64[D]')

    def get_days_forward(self, dar, maturities_relative_day, busdaycalendar):
        boot_days = np.array(
            [dar + dt.timedelta(days=1) + relativedelta(months=0)] + [
                dar + dt.timedelta(days=1) + relativedelta(months=i)
                for i in range(0, NB_PROJ_ZC)])
        boot_days = np.array(boot_days, dtype='datetime64[D]')

        jj_days = boot_days + [np.timedelta64(1, 'D') for x in range(0, NB_PROJ_ZC + 1)]
        jj_days = self.find_next_busday(jj_days, busdaycalendar)

        tn_days = jj_days + [np.timedelta64(1, 'D') for x in range(0, NB_PROJ_ZC + 1)]
        tn_days = self.find_next_busday(tn_days, busdaycalendar)

        tn_days = self.convert_to_pd_datetime(tn_days)

        swap_dates = np.array([tn_days, ] * (len(tm.ZC_MATURITIES_NAMES) - 2), ) + np.array(
            maturities_relative_day).reshape((len(tm.ZC_MATURITIES_NAMES) - 2), 1)

        swap_dates = np.vstack((jj_days, tn_days, swap_dates))

        swap_dates = self.convert_to_numpy_datetime(swap_dates)
        swap_dates = self.find_next_busday(swap_dates, busdaycalendar)

        swap_day = swap_dates - np.array(boot_days, dtype='datetime64[D]')

        nbr_swap_days = swap_day / np.timedelta64(1, 'D')

        nbr_swap_days = np.vstack((np.zeros(NB_PROJ_ZC + 1), nbr_swap_days[1:]))
        return nbr_swap_days, pd.DataFrame(swap_dates)

    def interpolate_2d_array_columns_by_column(self, x, y, z):
        s = []
        for j in range(y.shape[1]):
            xp = list(map(float, y[:, j]))
            yp = list(map(float, z[:, j]))
            result = np.interp(x[j], xp, yp)
            s = np.append(s, result)
        return np.array(s, dtype=np.float64)

    def get_maturities_relative_day(self, ):
        return ([relativedelta(days=7)] + [relativedelta(months=x) for x in range(1, 12)]
                + [relativedelta(years=x) for x in range(1, 31)])

    def bootstrap_curve(self, curve, days_fwd, forward_dates, scenario_tx_df):
        # Step 1: Retrieve Rate Data'
        rate_df = self.get_currency_rate_curves(curve, scenario_tx_df)[:, :NB_PROJ_ZC + 1]

        # Step 2: Calculate the Overnight (TN) Discount Factor
        # DF(TN) = 1 / (1 + days_fwd * Rate[0] /360
        tn_discount_factor = np.ones(NB_PROJ_ZC + 1) / (1 + days_fwd[1] * rate_df[0] / YEAR_NBR_OF_DAYS_360)

        # Step 2: Calculate Discount Factors for Short-Term Maturities
        # DF = DF(TN) / (1 + (days_fwd - days_TN) /360
        discount_factor = tn_discount_factor / (1 + (days_fwd[2:len(tm.ZC_MATURITIES_NAMES_INF_1Y)] - days_fwd[1])
                                                * rate_df[2:len(tm.ZC_MATURITIES_NAMES_INF_1Y)] / YEAR_NBR_OF_DAYS_360)
        discount_factor = np.vstack((np.ones(NB_PROJ_ZC + 1), tn_discount_factor, discount_factor))

        # Step 5: Calculate Zero-Coupon Rates for short-term maturities
        # N * (1 + ZC)**(nb_days/360) = N * DF
        zc = np.power(discount_factor[1:len(tm.ZC_MATURITIES_NAMES_INF_1Y)],
                      (-YEAR_NUMBER_OF_DAY_365 / (days_fwd[1:len(tm.ZC_MATURITIES_NAMES_INF_1Y)]))) - 1
        zc = np.vstack((rate_df[1], zc))

        # Step 6: Handle Long-Term Maturities
        forward_dates = forward_dates.values
        swap_days = ((forward_dates - forward_dates[1, :]) / np.timedelta64(1, 'D'))
        swap_accrual = self.get_swap_accruals_2(forward_dates)
        swap_accrual = pd.DataFrame(swap_accrual)
        someprod = np.zeros(NB_PROJ_ZC + 1)
        for i in range(len(tm.ZC_MATURITIES_NAMES_INF_1Y) - 1, len(tm.ZC_MATURITIES_NAMES)):
            if i > len(tm.ZC_MATURITIES_NAMES_INF_1Y) - 1:
                discount_factor = np.vstack((discount_factor,
                                             (discount_factor[1] * (
                                                     1 - someprod * rate_df[i] / YEAR_NBR_OF_DAYS_360)) / (
                                                     1 + rate_df[i] * swap_accrual.loc[i, :] / YEAR_NBR_OF_DAYS_360),))
                zc = np.vstack((zc, np.power(discount_factor[i], (-YEAR_NUMBER_OF_DAY_365 / days_fwd[i])) - 1))

            temp = (1 + self.interpolate_2d_array_columns_by_column(swap_days[i], days_fwd[:i + 1, :], zc)) ** (
                    -swap_days[i] / YEAR_NUMBER_OF_DAY_365)
            someprod = someprod + temp * swap_accrual.loc[i, :]
        return days_fwd, zc

    def get_forward_days_and_dates(self, ):
        busdaycalendar = np.busdaycalendar(holidays=self.up.holidays_list)
        maturities_relative_day = self.get_maturities_relative_day()
        days_fwd, forward_dates = self.get_days_forward(self.up.dar, maturities_relative_day, busdaycalendar)
        return days_fwd, forward_dates

    def get_currency_rate_curves(self, curve, scenario_tx_df):
        filter_curve = (scenario_tx_df[tx_ref.CN_CODE_COURBE] == curve)
        filter_curve_mat = filter_curve & (
            scenario_tx_df[tx_ref.CN_MATURITY].isin(list(tm.curves_maturities_needed_for_bootstrap)))
        if not filter_curve_mat.any():
            raise ValueError("There no is curve in the referential with %s='%s'" % (tx_ref.CN_CODE_COURBE, curve))
        rate_df = (scenario_tx_df[filter_curve].set_index(tx_ref.CN_MATURITY).loc[
                   np.array(tm.curves_maturities_needed_for_bootstrap), 'M0':] / 100).values
        return rate_df

    def get_swap_accruals_2(self, forward_dates):
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
