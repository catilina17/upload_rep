import numpy as np
import numexpr as ne
from ...utils import utils as ut
from . import nmd_runoffs_commons as nrc


class NMD_RUNOFF_CALCULATION():
    def __init__(self, class_init_strat, cls_cash_flow, cls_vrsments):
        self.cls_init_ec = class_init_strat
        self.cls_proj = class_init_strat.cls_proj
        self.cls_hz_params = class_init_strat.cls_hz_params
        self.cls_rate = class_init_strat.cls_rate
        self.cls_data_rate = class_init_strat.cls_data_rate
        self.cls_cal = class_init_strat.cls_proj.cls_cal
        self.cls_fields = class_init_strat.cls_fields
        self.cls_palier = class_init_strat.cls_proj.cls_palier
        self.cls_versements = cls_vrsments
        self.cls_cash_flow = cls_cash_flow
        self.nb_rm_groups = class_init_strat.nb_rm_groups
        self.has_volatile_part = class_init_strat.cls_proj.has_volatile_part


    def compute_nmd_runoffs(self):
        data_ldp = self.cls_proj.data_ldp
        init_strat_ec = self.cls_init_ec.ec_depart
        coeffs_strates = self.cls_init_ec.coeffs_strates
        coeffs_strates_gptx = self.cls_init_ec.coeffs_gptx
        coeffs_strates_all = self.cls_init_ec.coeffs_strates_all
        coeffs_strates_gptx_all = self.cls_init_ec.coeffs_gptx_all
        n = self.cls_proj.n
        t = self.cls_proj.t
        dar_mois = self.cls_hz_params.dar_mois

        sc_rates = self.cls_rate.sc_rates[:, :t].copy()
        current_month = self.cls_cal.current_month

        interests_periods = self.cls_cal.interests_calc_periods
        year_nb_days = np.array(data_ldp[self.cls_fields.NB_DAYS_AN]).reshape(n, 1)
        capi_freq = np.array(data_ldp[self.cls_fields.NC_FREQ_CAP]).reshape(n, 1)
        #interest_cap_shift = np.array(data_ldp[self.cls_fields.NC_DECALAGE_INT_CAP]).reshape(n, 1)
        first_coupon_date = np.array(data_ldp[self.cls_fields.NC_LDP_FIRST_COUPON_DATE]).reshape(n, 1)
        capitalize = np.array(data_ldp[self.cls_fields.NC_CAPITALIZE]).astype(bool)
        is_gptx_model = (data_ldp[self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX].fillna("") != "").values
        capitalize_gptx = np.array(data_ldp[self.cls_fields.NC_CAPITALIZE]).astype(bool) & is_gptx_model
        accrued_interests = np.array(data_ldp[self.cls_fields.NC_LDP_INTERESTS_ACCRUALS])
        nb_days_m0 = np.array(data_ldp[self.cls_fields.NB_DAYS_M0])
        mat_date = data_ldp[self.cls_fields.NC_LDP_MATUR_DATE + "_REAL"]
        mat_date_day = mat_date.dt.day
        breakdown = np.array(data_ldp[self.cls_fields.NC_LDP_RM_GROUP_PRCT])
        begin_month = self.cls_cal.mois_depart
        begin_capital = init_strat_ec[np.arange(0, n), np.minimum(begin_month.reshape(n) - 1, t)].reshape(n, 1)
        capi_rate = np.array(data_ldp[self.cls_fields.NC_LDP_CAPITALIZATION_RATE])
        versements = self.cls_versements.ec_versements_all

        sc_rates_lag = ut.roll_and_null(sc_rates, 1)
        sc_rates_lag[:, 0] = np.array(data_ldp[self.cls_fields.NC_LDP_RATE]) * 12
        sc_rates_lag = nrc.adjust_interest_rate_to_accruals(sc_rates_lag, interests_periods, accrued_interests,
                                                            begin_capital, year_nb_days, nb_days_m0, n)

        self.get_stratified_capital_default(init_strat_ec[~capitalize], coeffs_strates[~capitalize],
                                            coeffs_strates_gptx[~capitalize], begin_month[~capitalize],
                                            current_month[~capitalize], t)

        self.get_stratified_capital_with_capi(init_strat_ec[capitalize], coeffs_strates[capitalize],
                                              coeffs_strates_all[capitalize], sc_rates_lag[capitalize],
                                              interests_periods[capitalize],
                                              year_nb_days[capitalize], capi_freq[capitalize],
                                              begin_month[capitalize], first_coupon_date[capitalize],
                                              current_month[capitalize], mat_date_day[capitalize], breakdown[capitalize],
                                              capi_rate[capitalize], versements[capitalize], dar_mois, t)

        self.get_stratified_capital_with_capi(init_strat_ec[capitalize_gptx], coeffs_strates_gptx[capitalize_gptx],
                                              coeffs_strates_gptx_all[capitalize_gptx],
                                              sc_rates_lag[capitalize_gptx], interests_periods[capitalize_gptx],
                                              year_nb_days[capitalize_gptx], capi_freq[capitalize_gptx],
                                              begin_month[capitalize_gptx], first_coupon_date[capitalize_gptx],
                                              current_month[capitalize_gptx], mat_date_day[capitalize_gptx],
                                              breakdown[capitalize_gptx], capi_rate[capitalize_gptx],
                                              versements[capitalize_gptx], dar_mois,
                                              t, is_gptx=True)

        capital_ec = np.zeros((n, t + 1))
        capital_ec_stable = np.zeros((n, t + 1))
        capital_ec_mni_volatile = np.zeros((n, t + 1))
        capital_ec_gptx = np.zeros((n, t + 1))
        capital_ec_gptx_stable = np.zeros((n, t + 1))
        capital_ec_gptx_mni_volatile = np.zeros((n, t + 1))

        capital_ec[capitalize] = self.capital_ec_with_capi
        capital_ec[~capitalize] = self.capital_ec_default

        capital_ec_mni_volatile[capitalize] = self.capital_ec_mni_volatile_with_capi

        capital_ec_stable[capitalize] = self.capital_ec_stable_with_capi
        capital_ec_stable[~capitalize] = self.capital_ec_stable_default

        capital_ec_gptx[~capitalize] = self.capital_ec_gptx_default
        capital_ec_gptx[capitalize_gptx] = self.capital_ec_with_capi_gptx

        capital_ec_gptx_stable[~capitalize] = self.capital_ec_stable_gptx_default
        capital_ec_gptx_stable[capitalize_gptx] = self.capital_ec_stable_with_capi_gptx
        capital_ec_gptx_mni_volatile[capitalize_gptx] = self.capital_ec_mni_volatile_with_capi_gptx

        multiplier = (data_ldp[self.cls_fields.NC_NOM_MULTIPLIER].values
                      * data_ldp[self.cls_fields.NC_LDP_NB_CONTRACTS].values.astype(np.float32))

        self.capital_ec_gptx = capital_ec_gptx  * multiplier.reshape(n, 1)
        self.capital_ec_stable = capital_ec_stable * multiplier.reshape(n, 1)
        self.capital_ec_mni_volatile = capital_ec_mni_volatile * multiplier.reshape(n, 1)
        self.capital_ec = capital_ec * multiplier.reshape(n, 1)
        self.capital_ec_gptx_stable = capital_ec_gptx_stable * multiplier.reshape(n, 1)
        self.capital_ec_gptx_mni_volatile = capital_ec_gptx_mni_volatile * multiplier.reshape(n, 1)
        self.year_nb_days = year_nb_days * np.ones(current_month.shape)

    def fill_capital_before_begin_month(self, remaining_capital, ec_avt_amor, begin_month, current_month):
        remaining_cap_proj = remaining_capital[:, 1:]
        ec_avt_amor_proj = ec_avt_amor[:, 1:]
        interests_schedule = ne.evaluate("current_month - begin_month + 1")
        remaining_capital[:, 1:] = ne.evaluate("where(interests_schedule <= 0, ec_avt_amor_proj, remaining_cap_proj)")
        remaining_capital[:, 1:] = ne.evaluate("where(remaining_cap_proj < 0, 0, remaining_cap_proj)")
        remaining_capital[:, 0] = ec_avt_amor[:, 0]
        return remaining_capital
    ######@profile
    def get_stratified_capital_default(self, init_strat_ec, stratif_coeffs, coeffs_strates_gptx,
                                       begin_month, current_month, t):
        n = init_strat_ec.shape[0]
        if n > 0:
            begin_capital = init_strat_ec[np.arange(0, n), np.minimum(begin_month.reshape(n) - 1, t)].reshape(n, 1)
            capital = (1 - np.cumsum(stratif_coeffs.reshape(n, t), axis=1)) * begin_capital.reshape(n, 1)
            #capital =  np.where(current_month < begin_month.reshape(n, 1) - 1, 0, capital)
            capital = np.concatenate([np.zeros((n, 1)), capital], axis=1)
            capital = np.round(capital, 10)

            capital_gptx = (1 - np.cumsum(coeffs_strates_gptx.reshape(n, t), axis=1)) * begin_capital.reshape(n, 1)
            capital_gptx = np.concatenate([np.zeros((n, 1)), capital_gptx], axis=1)
            capital_gptx = np.round(capital_gptx, 10)

            self.capital_ec_default = self.fill_capital_before_begin_month(capital, init_strat_ec, begin_month, current_month)
            self.capital_ec_stable_default = self.capital_ec_default.copy()
            self.capital_ec_mni_volatile_default = np.zeros((n, t + 1))

            self.capital_ec_gptx_default = self.fill_capital_before_begin_month(capital_gptx, init_strat_ec, begin_month, current_month)
            self.capital_ec_stable_gptx_default = self.capital_ec_gptx_default.copy()

        else:
            self.capital_ec_default = np.zeros((n, t + 1))
            self.capital_ec_stable_default = np.zeros((n, t + 1))
            self.capital_ec_mni_volatile_default = np.zeros((n, t + 1))

            self.capital_ec_gptx_default = np.zeros((n, t + 1))
            self.capital_ec_stable_gptx_default = np.zeros((n, t + 1))

    ###@profile
    def get_stratified_capital_with_capi(self, init_strat_ec, coeffs_strates, coeffs_strates_all, sc_rates, interests_periods,
                                         year_nb_days, capi_freq, begin_month, first_coupon_date,
                                         current_month, mat_date_day, breakdown, capi_rate, versements, dar_mois, t, is_gptx=False):
        n = init_strat_ec.shape[0]
        if n > 0:
            capi_freq2 = ne.evaluate("where(capi_freq ==0, t, capi_freq)")
            #maturity = t - ut.first_nonzero(coeffs_strates[:,::-1], axis=1, val = 0, invalid_val=-1).reshape(n, 1)
            capi_schedule = ne.evaluate("((current_month == first_coupon_date - dar_mois)  & (capi_freq == 0))"
                                        " | ((( ( (current_month - first_coupon_date + dar_mois) % capi_freq) == 0) & (capi_freq != 0)))  ")
            nb_rm_groups = self.nb_rm_groups
            nb_vol_parts = 1 if self.has_volatile_part else 0
            monthly_interest = interests_periods / year_nb_days
            capital = np.zeros((n, t + 1))
            mni = np.zeros((n, t + 1))
            mni_vol = np.zeros((n, t + 1))
            capital_mni = np.zeros((n, t + 1))
            capital_mni_volatile = np.zeros((n, t + 1))
            index_stable = np.mod(np.arange(0,len(capital)), nb_rm_groups) != (nb_rm_groups - nb_vol_parts)
            breakdown_stable = np.zeros((n))
            breakdown_stable[index_stable] = (breakdown[index_stable]
                                / breakdown[index_stable].reshape(n//nb_rm_groups, nb_rm_groups - nb_vol_parts).sum(axis=1)
                                              .repeat(nb_rm_groups - nb_vol_parts, axis=0))
            breakdown_stable = breakdown_stable.reshape(n, 1)
            for p in np.unique(capi_freq2):
                f_capi = capi_freq2.reshape(n) == p
                capital_f = capital[f_capi].copy()
                mni_f = mni[f_capi]
                mni_vol_f = mni_vol[f_capi]
                capital_mni_f = capital_mni[f_capi]
                capital_mni_volatile_f = capital_mni_volatile[f_capi]
                f_capi_rate = capi_rate[f_capi].copy()
                capi_schedule_f = capi_schedule[f_capi]
                monthly_interest_f = monthly_interest[f_capi]
                sc_rates_f = sc_rates[f_capi]
                coeffs_strates_f = coeffs_strates[f_capi]
                coeffs_strates_all_f = coeffs_strates_all[f_capi]
                current_month_f = current_month[f_capi]
                _n = f_capi[f_capi].shape[0]
                index_stable_f = index_stable[f_capi]
                init_strat_ec_f = init_strat_ec[f_capi]
                versements_f = versements[f_capi]
                for h in range(1, t):
                    sum_coeff_h = 1 - np.sum(coeffs_strates_f[:, :h - 1], axis=1)
                    sum_coeff_h = np.where(sum_coeff_h == 0, 1, sum_coeff_h)
                    coeff_ec = (1 - coeffs_strates_f[:, h - 1] / (sum_coeff_h))
                    capital_f[:, h - 1] = capital_f[:, h - 1] + init_strat_ec_f[:, h - 1]
                    capital_f[:, h] = capital_f[:, h - 1] * coeff_ec
                    is_capital_zero = np.where(coeffs_strates_all_f[:, h:].sum(axis=1) == 0, True, False)
                    capital_mni_f[:, h] = np.where(is_capital_zero, 0, capital_mni_f[:, h - 1] * coeff_ec)
                    capital_mni_volatile_f[:, h] = np.where(is_capital_zero, 0, capital_mni_volatile_f[:, h - 1] * coeff_ec)
                    mni_f[:, h] = (capital_f[:, h - 1] + capital_mni_f[:, h - 1]) * sc_rates_f[:, h - 1] * monthly_interest_f[:, h - 1] * f_capi_rate
                    mni_vol_f[:, h] = capital_mni_volatile_f[:, h - 1] * sc_rates_f[:, h - 1] * monthly_interest_f[:, h - 1] *  f_capi_rate

                    capi_schedule_f_h = capi_schedule_f[:, h - 1]
                    capital_mni_add = (mni_f[:, 1:][:,max(0,h - p) :h]).sum(axis=1)
                    capital_mni_add = ne.evaluate("capital_mni_add * capi_schedule_f_h")
                    capital_mni_f[index_stable_f, h] = (capital_mni_f[index_stable_f, h]
                                                      + np.where(is_capital_zero[index_stable_f], 0, capital_mni_add[index_stable_f]))
                    if nb_vol_parts >=1 :
                        capital_mni_vol_add = (mni_vol_f[:, 1:][:,max(0,h - p) :h]).sum(axis=1)
                        capital_mni_vol_add = ne.evaluate("capital_mni_vol_add * capi_schedule_f_h")
                        is_loss_volatile = capital_f[~index_stable_f, h] == 0
                        is_loss_volatile_st = is_loss_volatile.repeat(nb_rm_groups - nb_vol_parts, axis=0)
                        capital_add_volatile = capital_mni_add[~index_stable_f].repeat(nb_rm_groups - nb_vol_parts, axis=0)
                        part_volatile = np.where(is_loss_volatile_st & ~is_capital_zero[index_stable_f], capital_add_volatile, 0)
                        capital_mni_volatile_f[index_stable_f, h]\
                            = (capital_mni_volatile_f[index_stable_f, h] + part_volatile
                               + np.where(is_capital_zero[index_stable_f], 0, capital_mni_vol_add[index_stable_f]))
                        capital_mni_f[~index_stable_f, h] = (capital_mni_f[~index_stable_f, h]
                                                             + np.where(is_loss_volatile, 0, capital_mni_add[~index_stable_f]))
                    else:
                        capital_mni_volatile_f[index_stable_f, h] = 0


                    capital_f[:, h] = capital_f[:, h] + versements_f[:, h]

                capital[f_capi] = np.round(capital_f, 10)
                capital_mni[f_capi] = np.round(capital_mni_f, 10)
                capital_mni_volatile[f_capi] = np.round(capital_mni_volatile_f, 10)

            capital_with_capi = self.fill_capital_before_begin_month(capital + capital_mni, init_strat_ec, begin_month, current_month)
            if not is_gptx:
                self.capital_ec_stable_with_capi = capital_with_capi
                self.capital_ec_mni_volatile_with_capi = capital_mni_volatile
                self.capital_ec_with_capi = capital_with_capi + capital_mni_volatile * breakdown_stable
            else:
                self.capital_ec_stable_with_capi_gptx = capital_with_capi
                self.capital_ec_mni_volatile_with_capi_gptx  = capital_mni_volatile
                self.capital_ec_with_capi_gptx  = capital_with_capi + capital_mni_volatile * breakdown_stable

        else:
            if not is_gptx:
                self.capital_ec_stable_with_capi = np.zeros((n, t + 1))
                self.capital_ec_mni_volatile_with_capi = np.zeros((n, t + 1))
                self.capital_ec_with_capi = np.zeros((n, t + 1))
            else:
                self.capital_ec_stable_with_capi_gptx = np.zeros((n, t + 1))
                self.capital_ec_mni_volatile_with_capi_gptx = np.zeros((n, t + 1))
                self.capital_ec_with_capi_gptx = np.zeros((n, t + 1))
