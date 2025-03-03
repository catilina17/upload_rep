import numpy as np
from .default_amor.class_default_amortization import Default_Amortization
from .palier_amor.class_palier_amortization import Palier_Amortization
from .other_amor.class_douteux_amortization import Douteux_Amortization
from .other_amor.class_cash_flow_amortization import Cash_Flow_Amortization


class Static_Amortization():
    """
    Formate les données
    """
    def __init__(self, cls_format, class_init_ecoulement, cls_cash_flow):
        self.cls_init_ec = class_init_ecoulement
        self.cls_proj = class_init_ecoulement.cls_proj
        self.cls_rate = class_init_ecoulement.cls_rate
        self.cls_cal = class_init_ecoulement.cls_cal
        self.cls_hz_params = class_init_ecoulement.cls_proj.cls_hz_params
        self.cls_cash_flow = cls_cash_flow
        self.cls_fields = class_init_ecoulement.cls_fields
        self.cls_palier = class_init_ecoulement.cls_proj.cls_palier
        self.cls_data_rate = class_init_ecoulement.cls_data_rate
        self.cls_format = cls_format
    ###########@profile
    def generate_static_amortization(self, tx_params):
        data_ldp = self.cls_proj.data_ldp
        cap_before_amor = self.cls_init_ec.ec_depart
        n = self.cls_proj.n
        t = self.cls_proj.t
        t_max = self.cls_proj.t_max

        dic_palier = self.cls_palier.dic_palier
        is_palier = np.array(dic_palier["is_palier"]).reshape(n)
        not_palier = ~np.array(dic_palier["is_palier"]).reshape(n)

        current_month = self.cls_cal.current_month
        current_month_max = self.cls_proj.current_month_max
        begin_month = self.cls_cal.mois_depart
        amor_begin_month = self.cls_cal.mois_depart_amor
        amor_end_month = self.cls_cal.mois_fin_amor
        drac_amor = self.cls_cal.drac_amor

        sc_rates = self.cls_rate.sc_rates[:, :t]
        sc_rates_ftp = self.cls_rate.sc_rates_ftp[:, :t]
        sc_rates_lag = self.cls_rate.sc_rates_lag[:, :t]
        sc_rates_ftp_lag = self.cls_rate.sc_rates_ftp_lag[:, :t]
        tombee_fixing = self.cls_rate.tombee_fixing[:, :t]
        period_fixing = self.cls_rate.period_fixing[:, :t]

        dar_mois = self.cls_hz_params.dar_mois
        interests_periods = self.cls_cal.interests_calc_periods
        data_cf = self.cls_cash_flow.data_cash_flows

        prof_ech_const = np.array(data_ldp[self.cls_fields.NC_PROFIL_AMOR] == "ECHCONST").reshape(n, 1)
        prof_linear = np.array(data_ldp[self.cls_fields.NC_PROFIL_AMOR].isin(["LINEAIRE", "LINEAIRE_ECH"])).reshape(n, 1)
        prof_linear_ech = np.array(data_ldp[self.cls_fields.NC_PROFIL_AMOR] == "LINEAIRE_ECH").reshape(n, 1)
        amor_freq = np.array(data_ldp[self.cls_fields.NC_FREQ_AMOR]).reshape(n, 1)
        suspend_or_capitalize = np.array(data_ldp[self.cls_fields.NC_SUSPEND_AMOR_OR_CAPITALIZE_INTERESTS]).astype(bool).reshape(n, 1)
        capitalize = np.array(data_ldp[self.cls_fields.NC_CAPITALIZE]).astype(bool).reshape(n, 1)
        ech_value = np.array(data_ldp[self.cls_fields.NC_LDP_ECHEANCE_VAL]).reshape(n, 1)
        capi_freq = np.array(data_ldp[self.cls_fields.NC_FREQ_CAP]).reshape(n, 1)
        capi_rate = np.array(data_ldp[self.cls_fields.NC_LDP_CAPITALIZATION_RATE]).reshape(n, 1)
        year_nb_days = np.array(data_ldp[self.cls_fields.NB_DAYS_AN]).reshape(n, 1) * np.ones(current_month.shape)
        capital_amor_shift = np.array(data_ldp[self.cls_fields.NC_DECALAGE_AMOR_CAP]).reshape(n, 1)
        interest_cap_shift = np.array(data_ldp[self.cls_fields.NC_DECALAGE_INT_CAP]).reshape(n, 1)
        accrued_interests = np.array(data_ldp[self.cls_fields.NC_LDP_INTERESTS_ACCRUALS]).reshape(n)
        is_fixed = data_ldp[self.cls_fields.NC_LDP_RATE_TYPE].values.reshape(n, 1) == "FIXED"
        fixed_rate = data_ldp[self.cls_fields.NC_LDP_RATE].values

        nb_day_m0 = np.array(data_ldp[self.cls_fields.NB_DAYS_M0]).reshape(n)
        ech_rate = self.cls_rate.mean_sc_rate(self.cls_rate.sc_rates, n, drac_amor, amor_end_month, amor_begin_month,
                                                        current_month_max, t_max, ~is_fixed.reshape(n), fixed_rate)
        orig_ech_value = ech_value.copy()


        ech_value = np.where(prof_ech_const & ~is_fixed, np.nan, ech_value)

        mat_date = np.array(data_ldp[self.cls_fields.NC_LDP_MATUR_DATE]).reshape(n, 1)

        cap_init = self.cls_init_ec.ec_depart
        begin_capital = cap_init[np.arange(0, n), np.minimum(amor_begin_month.reshape(n), t)].reshape(n, 1)
        cap_init_df = cap_init[not_palier]
        cap_init_palier = cap_init[is_palier]

        cls_def_amor = Default_Amortization(self.cls_init_ec)

        cls_def_amor.calculate_default_amortization(data_ldp[not_palier], cap_init_df,
                                                          ech_value[not_palier], sc_rates[not_palier],
                                                          ech_rate[not_palier], begin_capital[not_palier],
                                                          current_month[not_palier], amor_freq[not_palier],
                                                          capital_amor_shift[not_palier], begin_month[not_palier],
                                                          amor_begin_month[not_palier], amor_end_month[not_palier],
                                                          mat_date[not_palier], interests_periods[not_palier],
                                                          prof_ech_const[not_palier], prof_linear[not_palier],
                                                          prof_linear_ech[not_palier],
                                                          drac_amor[not_palier],
                                                          year_nb_days[not_palier], capi_freq[not_palier],
                                                          suspend_or_capitalize[not_palier], capitalize[not_palier],
                                                          capi_rate[not_palier], accrued_interests[not_palier],
                                                          nb_day_m0[not_palier], orig_ech_value[not_palier],
                                                          interest_cap_shift[not_palier], is_fixed[not_palier], t)

        cls_pal_amor = Palier_Amortization(self.cls_format, self.cls_init_ec, self.cls_rate.tombee_fixing[is_palier],
                                           self.cls_rate.period_fixing[is_palier],
                                           self.cls_rate.sc_rates[is_palier], self.cls_rate.sc_rates_ftp[is_palier],
                                           self.cls_rate.sc_rates_lag[is_palier], self.cls_rate.sc_rates_ftp_lag[is_palier],
                                           current_month[is_palier], current_month_max[is_palier],
                                           interests_periods[is_palier], year_nb_days[is_palier],  t, t_max)

        cls_pal_amor.calculate_palier_amortization(data_ldp[is_palier], dic_palier, cap_init_palier,
                                                   begin_capital[is_palier],
                                                   capi_freq[is_palier], begin_month[is_palier],
                                                   amor_begin_month[is_palier], amor_end_month[is_palier],
                                                   mat_date[is_palier], dar_mois,
                                                   data_ldp[is_palier][self.cls_fields.NC_LDP_CLE].values,
                                                   drac_amor[is_palier],
                                                   accrued_interests[is_palier], nb_day_m0[is_palier],
                                                   capi_rate[is_palier], tx_params)

        cls_cash_flow = Cash_Flow_Amortization(self.cls_init_ec)
        capital_ec_cf = cls_cash_flow.get_amortized_capital_with_cash_flows(data_ldp, cap_before_amor, data_cf,
                                                           current_month, dar_mois, t)

        capital_ec = np.zeros((n, t + 1))

        capital_ec[not_palier] = cls_def_amor.remaining_capital
        capital_ec[is_palier] = cls_pal_amor.remaining_capital
        sc_rates[is_palier] = cls_pal_amor.sc_rates[:, :t]
        sc_rates_ftp[is_palier] = cls_pal_amor.sc_rates_ftp[:, :t]
        sc_rates_lag[is_palier] = cls_pal_amor.sc_rates_lag[:, :t]
        sc_rates_ftp_lag[is_palier] = cls_pal_amor.sc_rates_ftp_lag[:, :t]
        tombee_fixing[is_palier] = cls_pal_amor.tombee_fixing[:, :t]
        period_fixing[is_palier] =  cls_pal_amor.period_fixing[:, :t]
        year_nb_days[is_palier] =  cls_pal_amor.year_nb_days

        if len(data_cf) > 0:
            capital_ec[data_cf.index] = capital_ec_cf

        douteux = np.array(data_ldp[self.cls_fields.NC_LDP_PERFORMING] == "T").reshape(n)
        cls_douteux = Douteux_Amortization(self.cls_init_ec)
        capital_non_performing\
            = cls_douteux.calculate_non_performing_amortization(data_ldp[douteux], capital_ec[douteux],
                                                                current_month[douteux], t)
        capital_ec[douteux] = capital_non_performing

        multiplier =  (data_ldp[self.cls_fields.NC_NOM_MULTIPLIER].values
                       * data_ldp[self.cls_fields.NC_LDP_NB_CONTRACTS].values.astype(np.float32))
        capital_ec = capital_ec * multiplier.reshape(n, 1)

        self.cls_init_ec.ec_depart = self.cls_init_ec.ec_depart * multiplier.reshape(n, 1)

        self.cls_rate.sc_rates[:, :t] = sc_rates
        self.cls_rate.sc_rates_ftp[:, :t] = sc_rates_ftp
        self.cls_rate.sc_rates_lag[:, :t] = sc_rates_lag
        self.cls_rate.sc_rates_ftp_lag[:, :t] = sc_rates_ftp_lag
        self.cls_rate.tombee_fixing[:, :t] = tombee_fixing
        self.cls_rate.period_fixing[:, :t] = period_fixing
        self.year_nb_days = year_nb_days
        self.capital_ec = capital_ec
        self.capital_ec_stable = 0
        self.capital_ec_mni_volatile = 0
        self.capital_ec_gptx = 0
        self.capital_ec_gptx_stable = 0
        self.capital_ec_gptx_mni_volatile = 0
