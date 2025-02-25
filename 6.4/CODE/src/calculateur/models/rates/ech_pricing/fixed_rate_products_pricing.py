import logging
import copy
import numpy as np
import numexpr as ne
from calculateur.models.run_offs.ech_amortization.amortissement_statique.class_init_ecoulement import Init_Ecoulement
from calculateur.models.run_offs.ech_amortization.amortissement_statique.class_static_amortization import Static_Amortization
from calculateur.models.run_offs.ech_amortization.amortissement_statique.amortization_commons import AmorCommons as ac
from calculateur.models.utils import utils as ut
from calculateur.models.rarn.class_rarn_manager import RARN_Manager

logger = logging.getLogger(__name__)


class FIXED_RATE_PRODUCT_PRICING():
    """
    Formate les données
    """

    def __init__(self, cls_proj, cls_zc_curves):
        self.cls_proj = cls_proj
        self.cls_fields = cls_proj.cls_fields
        self.cls_cal = self.cls_proj.cls_cal
        self.cls_zc_curves = cls_zc_curves
        self.contracts_with_floor = ['A-CRE-AV', 'A-EFT-CRECOM', 'A-CR-TRESO', 'A-LIGNE-TRES', 'A-CR-HAB-STD',
                                     'A-CR-HAB-AJU', 'A-CR-HAB-LIS', 'A-PR-STARDEN', 'A-CR-REL-HAB', 'A-CR-HAB-BON',
                                     'A-PR-PEL', 'A-PR-CEL', 'A-PTZ', 'A-PTZ+', 'A-PR-PERSO', 'A-CR-EQ-STD',
                                     'A-CR-EQ-CPLX',
                                     'A-CR-EQ-MUL', 'A-CR-EQ-STR', 'A-CR-LBO', 'A-CR-EQ-AIDE', 'A-PR-LDD', 'A-CR-BAIL',
                                     'A-PR-PATRI', 'P-PLAN-PREF', 'P-BON', 'P-CAT-STD', 'P-CAT-ER', 'P-CAT-VIE',
                                     'P-CAT-PROG',
                                     'P-CAT-PELP', 'P-CAT-OPTION', 'P-PEP-STD', 'P-PEP-IND', 'P-PEP-RENTE',
                                     'P-PEP-PROG',
                                     'A-CR-EQ-OC', 'A-CR-PRJ']

    ############@profile
    def generate_linear_amortization(self, cls_format, cls_cash_flow, cls_model_params, tx_params):
        cls_proj_linear = copy.deepcopy(self.cls_proj)
        dar_mois = cls_proj_linear.cls_hz_params.dar_mois
        cls_proj_linear.t = np.max(cls_proj_linear.data_ldp[self.cls_fields.NC_LDP_MATUR_DATE].values - dar_mois + 2)
        cls_proj_linear.t_max = max(cls_proj_linear.t, cls_proj_linear.t_max)
        cls_proj_linear.current_month, cls_proj_linear.current_month_max = cls_proj_linear.build_current_month()
        cls_proj_linear.cls_rate.sc_rates = np.zeros((cls_proj_linear.n, cls_proj_linear.t_max))
        cls_proj_linear.cls_rate.sc_rates_lag = np.zeros((cls_proj_linear.n, cls_proj_linear.t_max))
        cls_proj_linear.cls_rate.sc_rates_ftp = np.zeros((cls_proj_linear.n, cls_proj_linear.t_max))
        cls_proj_linear.cls_rate.sc_rates_ftp_lag = np.zeros((cls_proj_linear.n, cls_proj_linear.t_max))
        cls_proj_linear.cls_rate.tombee_fixing = np.zeros((cls_proj_linear.n, cls_proj_linear.t_max))
        cls_proj_linear.cls_rate.period_fixing = np.zeros((cls_proj_linear.n, cls_proj_linear.t_max))
        profil_amor = cls_proj_linear.data_ldp[self.cls_fields.NC_PROFIL_AMOR].values
        cls_proj_linear.data_ldp[self.cls_fields.NC_PROFIL_AMOR] = np.where(profil_amor != "INFINE", "LINEAIRE",
                                                                            profil_amor)
        cls_proj_linear.data_ldp[self.cls_fields.NC_LDP_NOMINAL] = 1
        cls_proj_linear.data_ldp[self.cls_fields.NC_NOM_MULTIPLIER] = 1
        cls_proj_linear.data_ldp[self.cls_fields.NC_LDP_OUTSTANDING] = 0

        cls_proj_linear.data_ldp[self.cls_fields.NC_CAPITALIZE] = False
        cls_proj_linear.data_ldp[self.cls_fields.NC_SUSPEND_AMOR_OR_CAPITALIZE_INTERESTS] = False

        cls_proj_linear.cls_cal.prepare_calendar_parameters(cls_proj_linear)
        cls_proj_linear.cls_cal.get_calendar_periods(cls_proj_linear)
        cls_proj_linear.cls_cal.load_rarn_calendar_parameters(cls_proj_linear)
        cls_proj_linear.cls_fixing_cal.get_fixing_parameters(cls_proj_linear)

        cls_init_ec = Init_Ecoulement(cls_proj_linear, cls_model_params)
        cls_init_ec.get_ec_before_amortization()

        """ CALCUL AMORTISSEMENT """
        cls_stat_amor = Static_Amortization(cls_format, cls_init_ec, cls_cash_flow)
        cls_stat_amor.generate_static_amortization(tx_params)

        self.drac_amor = cls_stat_amor.cls_cal.drac_amor
        self.generate_rates_chronicles(cls_proj_linear, cls_model_params, tx_params)
        cls_rarn_liq = self.get_ra_rn_params(cls_proj_linear, cls_model_params)
        with_ra = (cls_proj_linear.data_ldp[self.cls_fields.NC_LDP_FREQ_INT] != "N").values
        #self.remaining_capital = cls_stat_amor.capital_ec.copy()
        self.remaining_capital = self.get_ra_effect(cls_rarn_liq, cls_stat_amor.capital_ec.copy(), with_ra)
        self.max_duree = cls_proj_linear.t
        self.interests_calc_periods = cls_proj_linear.cls_cal.interests_calc_periods
        self.current_month = cls_proj_linear.current_month
        cls_proj_linear = None

    def generate_real_amortization(self, cls_format, cls_cash_flow, cls_model_params, tx_params):
        cls_proj_linear = copy.deepcopy(self.cls_proj)
        dar_mois = cls_proj_linear.cls_hz_params.dar_mois
        cls_proj_linear.t = np.max(cls_proj_linear.data_ldp[self.cls_fields.NC_LDP_MATUR_DATE].values - dar_mois + 2)
        cls_proj_linear.t_max = max(cls_proj_linear.t, cls_proj_linear.t_max)
        cls_proj_linear.current_month, cls_proj_linear.current_month_max = cls_proj_linear.build_current_month()
        cls_proj_linear.data_ldp[self.cls_fields.NC_LDP_NOMINAL] = 1
        cls_proj_linear.data_ldp[self.cls_fields.NC_LDP_OUTSTANDING] = 0
        cls_proj_linear.data_ldp[self.cls_fields.NC_NOM_MULTIPLIER] = 1

        cls_proj_linear.data_ldp[self.cls_fields.NC_CAPITALIZE] = False
        cls_proj_linear.data_ldp[self.cls_fields.NC_SUSPEND_AMOR_OR_CAPITALIZE_INTERESTS] = False

        cls_proj_linear.cls_cal.prepare_calendar_parameters(cls_proj_linear)
        cls_proj_linear.cls_cal.get_calendar_periods(cls_proj_linear)
        cls_proj_linear.cls_cal.load_rarn_calendar_parameters(cls_proj_linear)
        cls_proj_linear.cls_fixing_cal.get_fixing_parameters(cls_proj_linear)
        self.generate_rates_chronicles(cls_proj_linear, cls_model_params, tx_params)

        cls_init_ec = Init_Ecoulement(cls_proj_linear, cls_model_params)
        cls_init_ec.get_ec_before_amortization()

        """ CALCUL AMORTISSEMENT """
        cls_stat_amor = Static_Amortization(cls_format, cls_init_ec, cls_cash_flow)
        cls_stat_amor.generate_static_amortization(tx_params)

        self.drac_amor = cls_stat_amor.cls_cal.drac_amor

        cls_rarn_liq = self.get_ra_rn_params(cls_proj_linear, cls_model_params, is_ftp=True)
        with_ra = (cls_proj_linear.data_ldp[self.cls_fields.NC_LDP_FREQ_INT] != "N").values
        #self.remaining_capital = cls_stat_amor.capital_ec.copy()
        self.remaining_capital = self.get_ra_effect(cls_rarn_liq, cls_stat_amor.capital_ec.copy(), with_ra)
        self.max_duree = cls_proj_linear.t
        self.interests_calc_periods = cls_proj_linear.cls_cal.interests_calc_periods
        self.current_month = cls_proj_linear.current_month
        cls_proj_linear = None

    def get_ra_rn_params(self, cls_proj, cls_model_params, is_ftp=False):
        data_ldp = cls_proj.data_ldp
        mois_depart_rarn = cls_proj.cls_cal.mois_depart
        rarn_periods = cls_proj.cls_cal.rarn_periods
        drac_rarn = cls_proj.cls_cal.drac_rarn
        drac_init = cls_proj.cls_cal.drac_init
        current_month = cls_proj.cls_cal.current_month
        n = cls_proj.n
        t = cls_proj.t
        if not is_ftp:
            sc_rates = cls_proj.cls_rate.sc_rates.copy()
            sc_rates_lag = cls_proj.cls_rate.sc_rates_lag.copy()
        else:
            sc_rates = cls_proj.cls_rate.sc_rates_ftp.copy()
            sc_rates_lag = cls_proj.cls_rate.sc_rates_ftp_lag.copy()
        cls_rarn_liq = RARN_Manager(cls_proj, cls_model_params)
        cls_rarn_liq.get_rarn_values(sc_rates, sc_rates_lag, data_ldp, mois_depart_rarn,
                                     current_month, rarn_periods, drac_rarn, drac_init, n, t)
        return cls_rarn_liq

    def generate_rates_chronicles(self, cls_proj, cls_model_params, tx_params):
        cls_proj.cls_data_rate.prepare_curve_rates(cls_proj, cls_model_params, tx_params)
        cls_proj.cls_rate.get_rates(cls_proj, cls_model_params)

    def get_ra_effect(self, cls_rarn, remaining_capital, no_ra):
        tx_ra = cls_rarn.tx_ra
        rarn_na = np.isnan(tx_ra)
        rarn_effect = ne.evaluate('where(rarn_na, 1, 1 - tx_ra)')
        rarn_effect_cum = rarn_effect.cumprod(axis=1)

        remaining_capital_proj = remaining_capital[:, 1:]
        remaining_capital[no_ra, 1:] = remaining_capital_proj[no_ra] * rarn_effect_cum[no_ra]
        return remaining_capital

    def check_pricing_curves(self):
        self.pricing_curve = self.cls_proj.data_ldp[self.cls_fields.NC_PRICING_CURVE]
        absent_zc_curve = self.pricing_curve[
            ~self.pricing_curve.isin(list(self.cls_zc_curves.zc_all_curves.keys()))].copy().unique().tolist()
        if len(absent_zc_curve) > 0:
            msg_erreur = "Certaines courbes ZC sont absentes : %s" % absent_zc_curve
            logger.error(msg_erreur)
            raise ValueError(msg_erreur)
        self.pricing_curve = self.pricing_curve.values

    #############@profile
    def get_discount_factor(self):
        n = self.cls_proj.n
        discount_factor = np.zeros((n, self.max_duree))
        dar_mois = self.cls_proj.cls_hz_params.dar_mois
        for curve_name in np.unique(self.pricing_curve):
            filter_curve = (self.pricing_curve == curve_name).reshape(n)
            zc_curve = self.cls_zc_curves.zc_all_curves[curve_name]
            index_mois = self.cls_proj.data_ldp.loc[
                             filter_curve, self.cls_fields.NC_LDP_TRADE_DATE].values - dar_mois - 1
            zc_curve_data = zc_curve[index_mois][:, :self.max_duree]
            delta_days = self.cls_zc_curves.delta_days_from_fwd_month[index_mois][:, :self.max_duree]
            discount_factor[filter_curve] = 1 / ((1 + (zc_curve_data)) ** (delta_days / 365))

        self.discount_factor = discount_factor

    def get_paid_interests_deblo(self, cls_proj, paid_interests):
        num_rule = (np.nan_to_num(np.array(cls_proj.data_ldp[self.cls_fields.NC_LDP_RELEASING_RULE]), nan=-1).astype(int))
        is_release_rule = (num_rule != -1)
        _n = is_release_rule[is_release_rule].shape[0]
        if _n > 0:
            val_date_month = (cls_proj.data_ldp.loc[is_release_rule,
            self.cls_fields.NC_LDP_VALUE_DATE].values - self.cls_proj.cls_hz_params.dar_mois).reshape(_n, 1)
            paid_interests_sum_before_deblo = (paid_interests[is_release_rule]
                                               * np.where(self.current_month[is_release_rule]
                                                          < val_date_month, 1, 0)).sum(axis=1).reshape(_n, 1)
            paid_interests[is_release_rule] = np.where(self.current_month[is_release_rule] < val_date_month,
                                                       0, paid_interests[is_release_rule])
            paid_interests[is_release_rule] = np.where(self.current_month[is_release_rule] == val_date_month,
                                                       paid_interests_sum_before_deblo, paid_interests[is_release_rule])
        return paid_interests

    def get_paid_interests(self, cls_proj):
        n = self.cls_proj.n
        begin_month = self.cls_proj.cls_cal.mois_depart
        remaining_cap_shifted = ut.roll_and_null(self.remaining_capital[:, 1:], shift=1)
        nb_days_an = np.array(self.cls_proj.data_ldp[self.cls_fields.NB_DAYS_AN]).reshape(n, 1)
        interests_schedule = self.current_month
        alpha_factor = self.interests_calc_periods / nb_days_an * np.where(interests_schedule > 0, 1, 0)
        paid_interests = alpha_factor * remaining_cap_shifted

        #paid_interests = self.get_paid_interests_deblo(cls_proj, paid_interests)

        periodicity_interests = self.cls_proj.data_ldp[self.cls_fields.NC_FREQ_INT + "_REAL"].values.reshape(n, 1)
        periodicity_interests = np.where(self.cls_proj.data_ldp[self.cls_fields.NC_CAPITALIZE].values.reshape(n, 1),
                                         self.cls_proj.data_ldp[self.cls_fields.NC_FREQ_CAP].values.reshape(n, 1),
                                         periodicity_interests)

        cond_existence = ((periodicity_interests > 1)).reshape(n)
        interest_shift = self.cls_proj.data_ldp[self.cls_fields.NC_DECALAGE_VERS_INT].values.reshape(n, 1)
        interest_shift = np.where(self.cls_proj.data_ldp[self.cls_fields.NC_CAPITALIZE].values.reshape(n, 1),
                                  self.cls_proj.data_ldp[self.cls_fields.NC_DECALAGE_INT_CAP].values.reshape(n, 1),
                                  interest_shift)

        # Pour les contrats à fréquence non mensuelle, on accumule les intérêts à partir de la value date de manière périodique
        if np.any(cond_existence, axis=0):
            for int_freq in np.unique(periodicity_interests):
                if int_freq > 1:
                    filter_freq = (periodicity_interests == int_freq).reshape(n)
                    interest_tombee = (self.current_month[filter_freq] - begin_month[filter_freq]
                                       - interest_shift[filter_freq]) % int_freq
                    paid_interests_freq = np.where(self.current_month[filter_freq] >= begin_month[filter_freq],
                                                 paid_interests[filter_freq], 0)
                    paid_interests_freq = ac.cum_reset_2d(paid_interests_freq, interest_tombee)
                    paid_interests[filter_freq] = ne.evaluate('where(interest_tombee == 0, paid_interests_freq, 0)')

        return paid_interests

    #############@profile
    def assign_fixed_rate(self, cls_proj):
        capital_inflow = np.maximum(0, ut.roll_and_null(self.remaining_capital[:, 1:], shift=1)
                                    - self.remaining_capital[:,1:])

        paid_interests = self.get_paid_interests(cls_proj)

        fixed_rate = np.nan_to_num(
            np.divide((1 - np.minimum(np.sum((self.discount_factor * capital_inflow), axis=1) , 1)).astype(float), (
                np.sum((self.discount_factor * paid_interests), axis=1)).astype(
                float)), posinf=0, neginf=0)

        """freq_cap = self.cls_proj.data_ldp[self.cls_fields.NC_FREQ_CAP].values
        duree = self.drac_amor.reshape(self.cls_proj.n)
        fixed_rate_cap = ((1/(np.sum((self.discount_factor * capital_inflow), axis=1)))**(1/((duree//freq_cap))) - 1)*12 / freq_cap
        fixed_rate = np.where(self.cls_proj.data_ldp[self.cls_fields.NC_CAPITALIZE].values,
                                         fixed_rate_cap, fixed_rate)"""

        is_fixed_rate = (self.cls_proj.data_ldp[self.cls_fields.NC_LDP_RATE_TYPE] == "FIXED").values
        marge_fix = self.cls_proj.data_ldp.loc[is_fixed_rate, self.cls_fields.NC_LDP_MKT_SPREAD].values
        self.cls_proj.data_ldp.loc[is_fixed_rate, self.cls_fields.NC_LDP_FTP_RATE] = fixed_rate[is_fixed_rate] / 12
        self.cls_proj.data_ldp.loc[is_fixed_rate, self.cls_fields.NC_LDP_RATE] = (fixed_rate[
                                                                                      is_fixed_rate] + marge_fix) / 12

        self.liq_rate_tv = fixed_rate[~is_fixed_rate].copy()
