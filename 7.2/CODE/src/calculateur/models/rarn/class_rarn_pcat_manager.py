import numpy as np
import numexpr as ne
from calculateur.models.utils import utils as ut
from utils import general_utils as gu

nan = np.nan


class RARN_PCAT_Manager():
    """
    Formate les données
    """

    def __init__(self, cls_proj, cls_model_params):
        self.cls_proj = cls_proj
        self.cls_rate = cls_proj.cls_rate
        self.cls_data_rate = cls_proj.cls_data_rate
        self.cls_cal = cls_proj.cls_cal
        self.cls_ra_rn_params = cls_model_params.cls_ra_rn_params
        self.cls_hz_params = cls_proj.cls_hz_params
        self.cls_fields = cls_proj.cls_fields
        self.dic_tx_swap  = cls_proj.cls_data_rate.dic_tx_swap
        self.dic_devises_index = cls_proj.cls_data_rate.dic_devises_index

    def get_ra_rn_model_cat(self, sc_rates_lag, data_ldp, current_month, mois_depart_rarn, ra_model, rn_model, drac_rarn,
                            delta_days, n, t):

        is_ra = ((ra_model == self.cls_ra_rn_params.MODEL_NAME_RA_CAT_LIQ) | (
                    ra_model == self.cls_ra_rn_params.MODEL_NAME_RA_CAT_TX)).values
        is_rn = ((rn_model == self.cls_ra_rn_params.MODEL_NAME_RN_CAT_LIQ) | (
                    rn_model == self.cls_ra_rn_params.MODEL_NAME_RN_CAT_TX)).values
        is_ra_rn = is_ra | is_rn
        is_ra_c = is_ra[is_ra_rn]
        is_rn_c = is_rn[is_ra_rn]

        sc_rates, sc_rates_tb_amor, day_tombee, day_fixing\
            = self.apply_ra_rn_cat_params_to_data(sc_rates_lag, drac_rarn, is_ra_rn, t)

        tx_ra_pcat_arbi, tx_ra_pcat_ss_arbi, tx_rn_pcat_arbi, tx_rn_pcat_ss_arbi, ra_pcat_calc_type, rn_pcat_calc_type \
            = self.get_arbitrage_and_non_arbitrage_ra_rn(data_ldp[is_ra_rn], ra_model[is_ra_rn], rn_model[is_ra_rn])

        tx_cms_tb_amor, tx_cms_tb_fix \
            = self.get_rate_arbitrage_from_cms(data_ldp[is_ra_rn], drac_rarn[is_ra_rn],
                                               current_month[is_ra_rn], mois_depart_rarn[is_ra_rn],
                                               day_fixing, day_tombee, t)

        is_arbitrage =  self.get_arbitrage_cases(sc_rates, sc_rates_tb_amor, tx_cms_tb_fix, tx_cms_tb_amor)

        tx_ra_cat = self.apply_RA_RN_CAT_rate_model(tx_ra_pcat_arbi[is_ra_c], tx_ra_pcat_ss_arbi[is_ra_c],
                                                    is_arbitrage[is_ra_c], mois_depart_rarn[is_ra], current_month[is_ra],
                                                    ra_pcat_calc_type[is_ra_c], delta_days[is_ra], t)

        tx_rn_cat = self.apply_RA_RN_CAT_rate_model(tx_rn_pcat_arbi[is_rn_c], tx_rn_pcat_ss_arbi[is_rn_c],
                                                    is_arbitrage[is_rn_c], mois_depart_rarn[is_rn],
                                                    current_month[is_rn], rn_pcat_calc_type[is_rn_c], delta_days[is_rn]
                                                    , t)

        return tx_ra_cat, tx_rn_cat, is_ra, is_rn

    def get_arbitrage_and_non_arbitrage_ra_rn(self, data_ldp, ra_model, rn_model):

        n = data_ldp.shape[0]

        ra_pcat_arbi = np.full((n), np.nan)
        ra_pcat_ss_arbi = np.full((n), np.nan)
        rn_pcat_arbi = np.full((n), np.nan)
        rn_pcat_ss_arbi = np.full((n), np.nan)
        ra_pcat_calc_type = np.full((n), "MENSUEL")
        rn_pcat_calc_type = np.full((n), "MENSUEL")

        is_ra_liq = (ra_model == self.cls_ra_rn_params.MODEL_NAME_RA_CAT_LIQ).values
        is_ra_tx =  (ra_model == self.cls_ra_rn_params.MODEL_NAME_RA_CAT_TX).values
        is_rn_liq = (rn_model == self.cls_ra_rn_params.MODEL_NAME_RN_CAT_LIQ).values
        is_rn_tx =  (rn_model == self.cls_ra_rn_params.MODEL_NAME_RN_CAT_TX).values

        cles_a_combiner = [self.cls_fields.NC_LDP_CONTRACT_TYPE, self.cls_fields.NC_LDP_ETAB,
                           self.cls_fields.NC_LDP_RATE_TYPE,
                           self.cls_fields.NC_LDP_MATUR,
                           self.cls_fields.NC_LDP_BASSIN,
                           self.cls_fields.NC_LDP_MARCHE]

        data_spec = data_ldp[cles_a_combiner].copy()
        data_spec[self.cls_fields.NC_LDP_MARCHE] = data_spec[self.cls_fields.NC_LDP_MARCHE].fillna("")

        data_spec_liq = data_spec[is_ra_liq | is_rn_liq].copy()

        if len(data_spec_liq) > 0:
            params = self.cls_ra_rn_params.ra_rn_pcat_liq_model.copy()
            for col, i in zip(cles_a_combiner, range(len(cles_a_combiner))):
                list_vals = data_ldp[col].unique().tolist() + ["*"]
                params = params[params.index.str.split("$").str[i].isin(list_vals)].copy()

            ra_or_rn_params_liq = gu.map_with_combined_key(data_spec_liq, params, cles_a_combiner,
                                                       symbol_any="*", no_map_value=np.nan, filter_comb=True,
                                                       necessary_cols=1, error=False, sep="$", upper_strip=False)

            ra_pcat_arbi[is_ra_liq | is_rn_liq] = ra_or_rn_params_liq[self.cls_ra_rn_params.NC_RA_ARBI].values
            ra_pcat_ss_arbi[is_ra_liq | is_rn_liq]  = ra_or_rn_params_liq[self.cls_ra_rn_params.NC_RA_WITHOUT_ARBI].values
            rn_pcat_arbi[is_ra_liq | is_rn_liq] = ra_or_rn_params_liq[self.cls_ra_rn_params.NC_RN_ARBI].values
            rn_pcat_ss_arbi[is_ra_liq | is_rn_liq]  = ra_or_rn_params_liq[self.cls_ra_rn_params.NC_RN_WITHOUT_ARBI].values

            ra_pcat_calc_type[is_ra_liq] = self.cls_ra_rn_params.type_ra_pcat_liq
            rn_pcat_calc_type[is_rn_liq] = self.cls_ra_rn_params.type_rn_pcat_liq

        data_spec_tx = data_spec[is_ra_tx | is_rn_tx].copy()

        if len(data_spec_tx) > 0:
            params = self.cls_ra_rn_params.ra_rn_pcat_tx_model.copy()
            for col, i in zip(cles_a_combiner, range(len(cles_a_combiner))):
                list_vals = data_ldp[col].unique().tolist() + ["*"]
                params = params[params.index.str.split("$").str[i].isin(list_vals)].copy()

            ra_or_rn_params_tx = gu.map_with_combined_key(data_spec_tx, params, cles_a_combiner,
                                                           symbol_any="*", no_map_value=np.nan, filter_comb=True,
                                                           necessary_cols=1, error=False, sep="$", upper_strip=False)

            ra_pcat_arbi[is_ra_tx | is_rn_tx] = ra_or_rn_params_tx[self.cls_ra_rn_params.NC_RA_ARBI].values
            ra_pcat_ss_arbi[is_ra_tx | is_rn_tx] = ra_or_rn_params_tx[self.cls_ra_rn_params.NC_RA_WITHOUT_ARBI].values
            rn_pcat_arbi[is_ra_tx | is_rn_tx] = ra_or_rn_params_tx[self.cls_ra_rn_params.NC_RN_ARBI].values
            rn_pcat_ss_arbi[is_ra_tx | is_rn_tx] = ra_or_rn_params_tx[self.cls_ra_rn_params.NC_RN_WITHOUT_ARBI].values

            ra_pcat_calc_type[is_ra_tx] = self.cls_ra_rn_params.type_ra_pcat_tx
            rn_pcat_calc_type[is_rn_tx] = self.cls_ra_rn_params.type_rn_pcat_tx

        return (ra_pcat_arbi.reshape(n, 1), ra_pcat_ss_arbi.reshape(n, 1), rn_pcat_arbi.reshape(n, 1),
                rn_pcat_ss_arbi.reshape(n, 1), ra_pcat_calc_type, rn_pcat_calc_type)

    def apply_RA_RN_CAT_rate_model(self, ra_or_rn_arbi, ra_or_rn_ss_arbi, is_arbitrage, mois_dep_rarn,
                                   current_month, type_ra_rn, delta_days, t):
        tx_ra_or_rn = ne.evaluate("where(is_arbitrage, ra_or_rn_arbi, ra_or_rn_ss_arbi)")
        tx_ra_or_rn = ne.evaluate("where(current_month < mois_dep_rarn, 0, tx_ra_or_rn)")
        tx_ra_or_rn = np.nan_to_num(tx_ra_or_rn)

        is_rar_or_rn_annual = (type_ra_rn == "ANNUEL")
        tx_ra_or_rn[is_rar_or_rn_annual] = self.render_monthly_ra_rn(tx_ra_or_rn[is_rar_or_rn_annual],
                                                                     delta_days[is_rar_or_rn_annual], t)
        return tx_ra_or_rn

    def render_monthly_ra_rn(self, tx_ra_rn_annual, delta_days, t):
        tx_ra_rn = ne.evaluate("1 - (1 - tx_ra_rn_annual) ** (30 / 360)")
        tx_ra_rn_2 = ne.evaluate("1 - (1 - tx_ra_rn_annual) ** (delta_days / 360)")
        tx_ra_rn = np.where(delta_days != self.cls_cal.delta_days[:, :t], tx_ra_rn_2, tx_ra_rn)
        return tx_ra_rn

    def get_arbitrage_cases(self, sc_rates, sc_rates_tb_amor, tx_cms_tb_fix, tx_cms_tb_amor):
        #Il y a deux taux dans le mois, celui de la tombée d'amortissement et celui de la tombée de fixing
        #Si l'un des taux franchit le seuil, alors RCO fait l'arbitrage
        delta_tx1 = ne.evaluate('sc_rates - tx_cms_tb_fix')
        delta_tx2 = ne.evaluate('sc_rates_tb_amor - tx_cms_tb_amor')
        is_arbitrage = (delta_tx1 < 0) | (delta_tx2 < 0)
        return is_arbitrage

    def get_rate_arbitrage_from_cms(self, data, drac_rarn, current_month, mois_depart_rarn,
                                    day_fixing, day_tombee, t):
        _n = data.shape[0]
        rate_arb_tombee_amor = np.zeros((_n, t))
        rate_arb_tombee_fix = np.zeros((_n, t))
        # On pourrait ajouter l'interpolation sur la tenor en particulier pour la fixing date
        # ou elle peut osciller entre deux valeurs
        indice_tenor = np.maximum(0, np.nan_to_num(drac_rarn) - 1)
        indice_tenor = np.minimum(self.cls_hz_params.max_projection - 1, indice_tenor).astype(int)
        indice_mois_fwd = np.minimum(current_month, self.cls_hz_params.max_projection - 1)
        nb_days = self.cls_cal.delta_days[:, 1:t + 1]
        for devise in data[self.cls_fields.NC_LDP_CURRENCY].unique():
            is_devise = np.array(data[self.cls_fields.NC_LDP_CURRENCY] == devise)
            if devise in self.dic_tx_swap:
                tx_cms1 = self.dic_tx_swap[self.dic_devises_index[devise]][1:, 1:][indice_tenor[is_devise], np.maximum(0, indice_mois_fwd[is_devise] - 2)]
                tx_cms2 = self.dic_tx_swap[self.dic_devises_index[devise]][1:, 1:][indice_tenor[is_devise], indice_mois_fwd[is_devise] - 1]
            else:
                tx_cms1 = self.dic_tx_swap[self.dic_devises_index["EUR"]][1:, 1:][indice_tenor[is_devise], np.maximum(0,indice_mois_fwd[is_devise] - 2)]
                tx_cms2 = self.dic_tx_swap[self.dic_devises_index["EUR"]][1:, 1:][indice_tenor[is_devise], indice_mois_fwd[is_devise] - 1]

            #interpolation linéaire sur la position dans le mois
            rate_arb_tombee_amor[is_devise] = (tx_cms1 * (nb_days - day_tombee[is_devise]) +
                                               tx_cms2 * (day_tombee[is_devise])) / nb_days
            rate_arb_tombee_fix[is_devise] = (tx_cms1 * (nb_days - day_fixing[is_devise]) +
                                              tx_cms2 * (day_fixing[is_devise])) / nb_days

        drac_nan = np.isnan(drac_rarn)
        rate_arb_tombee_amor = ne.evaluate('where(current_month < mois_depart_rarn, nan, rate_arb_tombee_amor)')
        rate_arb_tombee_amor = ne.evaluate('where(drac_nan, nan, rate_arb_tombee_amor)')

        rate_arb_tombee_fix = ne.evaluate('where(current_month < mois_depart_rarn, nan, rate_arb_tombee_fix)')
        rate_arb_tombee_fix = ne.evaluate('where(drac_nan, nan, rate_arb_tombee_fix)')
        rate_arb_tombee_fix = np.nan_to_num(rate_arb_tombee_fix)

        return rate_arb_tombee_amor, rate_arb_tombee_fix


    def get_rate_renego(self, sc_rates_lag, data, drac_rarn, current_month, mois_depart_rarn, renego_model, t):

        is_renego = (renego_model == self.cls_ra_rn_params.MODEL_NAME_RENEGO_CAT).values
        n = is_renego[is_renego].shape[0]

        if n > 0:
            sc_rates, sc_rates_tb_amor, day_tombee, day_fixing\
                = self.apply_ra_rn_cat_params_to_data(sc_rates_lag, drac_rarn, is_renego, t)

            rate_arb_tombee_amor, rate_arb_tombee_fix\
                = self.get_rate_arbitrage_from_cms(data[is_renego], drac_rarn[is_renego], current_month[is_renego],
                                                   mois_depart_rarn[is_renego], day_fixing, day_tombee, t)

            rate_renego = np.maximum(rate_arb_tombee_amor, rate_arb_tombee_fix)

        else:
            rate_renego = np.full((n, t), np.nan)

        return rate_renego, is_renego


    def apply_ra_rn_cat_params_to_data(self, sc_rates_lag, drac_rarn, is_r, t):
        drac_rarn_p_cat = ut.roll_and_null(drac_rarn[is_r])
        drac_rarn_p_cat[:, 0] = drac_rarn[is_r][:, 0] + 1

        sc_rates = sc_rates_lag[is_r][:, :t]
        sc_rates_lag = ut.roll_and_null(sc_rates)
        sc_rates_lag[:, 0] = sc_rates_lag[:, 1]

        day_tombee = self.cls_cal.period_begin_date[is_r][:, :t] - self.cls_cal.period_begin_date[is_r][:, :t].astype(
            'datetime64[M]') + 1
        day_tombee = np.nan_to_num(day_tombee.astype(int))
        day_fixing = ut.roll_and_null(self.cls_rate.tombee_fixing[is_r][:, :t], 1, val=np.nan)
        sc_rates_tb_amor = np.where(np.nan_to_num(day_fixing) <= day_tombee, sc_rates, sc_rates_lag)

        return sc_rates, sc_rates_tb_amor, day_tombee, day_fixing

    def apply_RA_CAT_rate_model(self, tx_ra_liq, sc_rates, sc_rates_tb_amor, tx_cms_tb_fix, tx_cms_tb_amor,
                                mois_dep_rarn, current_month):
        delta_tx1 = ne.evaluate('sc_rates - tx_cms_tb_fix')
        delta_tx2 = ne.evaluate('sc_rates_tb_amor - tx_cms_tb_amor')
        tx_ra = np.where((delta_tx1 < 0) | (delta_tx2 < 0), 1, tx_ra_liq)
        tx_ra = ne.evaluate("where(current_month < mois_dep_rarn, 0, tx_ra)")
        tx_ra = np.nan_to_num(tx_ra)
        return tx_ra

