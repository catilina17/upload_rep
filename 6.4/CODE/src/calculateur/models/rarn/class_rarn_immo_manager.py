import numpy as np
import numexpr as ne
from calculateur.models.utils import utils as ut
import logging

logger = logging.getLogger(__name__)

nan = np.nan


class RARN_IMMO_Manager():
    def __init__(self, cls_proj, cls_model_params):
        self.cls_proj = cls_proj
        self.cls_rate = cls_proj.cls_rate
        self.cls_data_rate = cls_proj.cls_data_rate
        self.cls_cal = cls_proj.cls_cal
        self.cls_ra_rn_params = cls_model_params.cls_ra_rn_params
        self.cls_hz_params = cls_proj.cls_hz_params
        self.cls_fields = cls_proj.cls_fields
        self.dic_tx_swap = cls_proj.cls_data_rate.dic_tx_swap
        self.dic_devises_index = cls_proj.cls_data_rate.dic_devises_index

    def get_ra_rn_model_immo(self, sc_rates, data_ldp, current_month, mois_depart_rarn, adj_delta_days,
                             ra_model, rn_model, drac_rarn, drac_init, n, t):

        is_ra = (ra_model == self.cls_ra_rn_params.MODEL_NAME_RA_IMMO).values
        is_rn = (rn_model == self.cls_ra_rn_params.MODEL_NAME_RN_IMMO).values
        is_ra_rn = is_ra | is_rn
        is_ra_c = is_ra[is_ra_rn]
        is_rn_c = is_rn[is_ra_rn]
        n_mod = is_ra_rn[is_ra_rn].shape[0]

        sc_rates = sc_rates[:, :t]

        tx_cms_arbi \
            = self.get_rate_arbitrage_from_cms(data_ldp[is_ra_rn],
                                               drac_rarn[is_ra_rn],
                                               current_month[is_ra_rn],
                                               mois_depart_rarn[is_ra_rn], n_mod, t)

        self.apply_ra_rn_immo_params_to_data(sc_rates[is_ra_rn], tx_cms_arbi, drac_rarn[is_ra_rn],
                                             drac_init[is_ra_rn], adj_delta_days[is_ra_rn], n_mod, t)

        tx_ra_model \
            = self.apply_ra_immo_model(sc_rates[is_ra_rn], mois_depart_rarn[is_ra_rn], current_month[is_ra_rn],
                                       is_ra_c, n, t)

        tx_rn_model \
            = self.apply_rn_immo_model(sc_rates[is_ra_rn], mois_depart_rarn[is_ra_rn], current_month[is_ra_rn],
                                       is_rn_c, n, t)

        return tx_ra_model, tx_rn_model, is_ra, is_rn



    def apply_ra_rn_immo_params_to_data(self, rates, tx_cms, drac_rarn, drac_init, delta_days, n, t):
        _rates = rates.reshape(n, t)
        self.delta_tx = ne.evaluate('_rates - tx_cms')
        drac_prct = ne.evaluate('drac_rarn/drac_init')
        drac_prct = ut.roll_and_null(drac_prct, shift=-1)  # burn_rate
        self.cases = [drac_prct <= 0.5, drac_prct <= 0.8, drac_prct <= 1]

        self.tx_ra_struct = np.select(self.cases , [self.cls_ra_rn_params.tx_ra_struct["D50"], \
                                         self.cls_ra_rn_params.tx_ra_struct["D80"],
                                         self.cls_ra_rn_params.tx_ra_struct["D100"]], default=np.nan)

        self.alpha = np.select(self.cases , [self.cls_ra_rn_params.alpha["D50"], self.cls_ra_rn_params.alpha["D80"],
                                  self.cls_ra_rn_params.alpha["D100"]], default=np.nan)
        self.gamma = np.select(self.cases , [self.cls_ra_rn_params.gamma["D50"], self.cls_ra_rn_params.gamma["D80"],
                                  self.cls_ra_rn_params.gamma["D100"]], default=np.nan)
        self.burnout = np.select(self.cases , [self.cls_ra_rn_params.burnout["D50"], self.cls_ra_rn_params.burnout["D80"],
                                    self.cls_ra_rn_params.burnout["D100"]], default=np.nan)
        self.sensi = np.select(self.cases , [self.cls_ra_rn_params.sensi["D50"], self.cls_ra_rn_params.sensi["D80"],
                                  self.cls_ra_rn_params.sensi["D100"]], default=np.nan)
        self.niveau = np.select(self.cases , [self.cls_ra_rn_params.niveau["D50"], self.cls_ra_rn_params.niveau["D80"],
                                   self.cls_ra_rn_params.niveau["D100"]], default=np.nan)
        self.min_ra = np.select(self.cases , [self.cls_ra_rn_params.min_ra["D50"], self.cls_ra_rn_params.min_ra["D80"],
                                   self.cls_ra_rn_params.min_ra["D100"]], default=np.nan)
        self.max_ra = np.select(self.cases , [self.cls_ra_rn_params.max_ra["D50"], self.cls_ra_rn_params.max_ra["D80"],
                                   self.cls_ra_rn_params.max_ra["D100"]], default=np.nan)

        #self.coeff = np.select(self.cases , [0, 1, 1], default=np.nan)

        self.delta_days_n_1 = ut.roll_and_null(delta_days, shift=-1, val=30)
        self.nb_days_an = 365

    def get_ra_immo_model(self, mois_dep_rarn, current_month, ra_b):
        delta_tx = self.delta_tx[ra_b]
        tx_ra_struct = self.tx_ra_struct[ra_b]
        niveau = self.niveau[ra_b]
        burnout = self.burnout[ra_b]
        sensi = self.sensi[ra_b]
        delta_days_n_1 = self.delta_days_n_1[ra_b]
        nb_days_an = self.nb_days_an

        tx_ra = ne.evaluate('tx_ra_struct + niveau * exp(-(100 * (delta_tx + 20 / 10000) - burnout) ** 2 / sensi)')
        tx_ra = np.minimum(np.maximum(self.min_ra[ra_b], tx_ra), self.max_ra[ra_b])

        if self.cls_ra_rn_params.activer_backtest:
            tx_ra = self.apply_ra_back_test(tx_ra, self.cases[ra_b])

        tx_ra = ne.evaluate('1 - (1 - tx_ra) ** (delta_days_n_1 / nb_days_an)')
        tx_ra = ne.evaluate("where(current_month < mois_dep_rarn, 0, tx_ra)")
        tx_ra = np.nan_to_num(tx_ra)

        return tx_ra

    def get_rn_immo_model(self, mois_dep_rarn, current_month, rn_b):
        delta_tx = self.delta_tx[rn_b]
        tx_ra_struct = self.tx_ra_struct[rn_b]
        niveau = self.niveau[rn_b]
        burnout = self.burnout[rn_b]
        sensi = self.sensi[rn_b]
        delta_days_n_1 = self.delta_days_n_1[rn_b]
        nb_days_an = self.nb_days_an
        alpha = self.alpha[rn_b]
        gamma = self.gamma[rn_b]
        #coeff = self.coeff[rn_b]

        tx_ra = ne.evaluate('tx_ra_struct + niveau * exp(-(100 * (delta_tx + 20 / 10000) - burnout) ** 2 / sensi)')
        tx_rn = ne.evaluate('exp(alpha * log(tx_ra) + gamma)')
        tx_rn = np.minimum(1, tx_rn)

        if self.cls_ra_rn_params.activer_backtest:
            tx_rn = self.apply_ra_back_test(tx_rn, self.cases[rn_b])

        tx_rn = ne.evaluate('1 - (1 - tx_rn) ** (delta_days_n_1 / nb_days_an)')
        tx_rn = ne.evaluate("where(current_month < mois_dep_rarn, 0, tx_rn)")
        tx_rn = np.nan_to_num(tx_rn)

        return tx_rn

    def apply_ra_immo_model(self, rates, mois_dep_rarn, current_month, ra_b, n, t):
        if n > 0:
            if not self.cls_ra_rn_params.is_user_mod:
                tx_ra = self.get_ra_immo_model(mois_dep_rarn[ra_b], current_month[ra_b], ra_b)
            else:
                tx_ra = self.cls_ra_rn_params.tx_ra_usr[:, :t] * np.ones((rates[ra_b].shape[0], 1))
                tx_ra = ne.evaluate('1 - (1 - tx_ra) ** (1 / 12)')
        else:
            tx_ra= np.zeros((0, t))

        return tx_ra

    def apply_rn_immo_model(self, rates, mois_dep_rarn, current_month, rn_b, n, t):
        if n > 0:
            n_mod = rn_b[rn_b]
            if not self.cls_ra_rn_params.is_user_mod:
                tx_rn = self.get_rn_immo_model(mois_dep_rarn[rn_b], current_month[rn_b], rn_b)
            else:
                tx_rn = self.cls_ra_rn_params.tx_rn_usr[:, :t] * np.ones((rates[rn_b].shape[0], 1))
                tx_rn = ne.evaluate('1 - (1 - tx_rn) ** (1 / 12)')
        else:
            tx_rn = np.zeros((0, t))

        return tx_rn
    def get_rate_arbitrage_from_cms(self, data_hab, drac_rarn, current_month, mois_depart_rarn, n, t):
        if n > 0:
            is_chf = np.array(data_hab[self.cls_fields.NC_LDP_CURRENCY] == "CHF")
            rate_arb = np.zeros((n, t))
            indice_tenor = np.maximum(0, np.nan_to_num(drac_rarn) - 1)
            indice_tenor = np.minimum(self.cls_hz_params.max_projection - 1, indice_tenor).astype(int)
            indice_mois_fwd = np.minimum(current_month, self.cls_hz_params.max_projection - 1)

            tx_cms_chf = self.dic_tx_swap["CHFLIBOR"][1:, 1:][indice_tenor[is_chf], indice_mois_fwd[is_chf] - 1]
            tx_marge_chf = self.cls_ra_rn_params.marge_md[indice_tenor[is_chf]]
            tx_cms_chf_mg = ne.evaluate('tx_cms_chf + tx_marge_chf / 10000')

            tx_cms_eur = self.dic_tx_swap["EURIBMOY"][1:, 1:][indice_tenor[~is_chf], indice_mois_fwd[~is_chf] - 1]
            tx_marge_eur = self.cls_ra_rn_params.marge_md[indice_tenor[~is_chf]]
            tx_cms_eur_mg = ne.evaluate('tx_cms_eur + tx_marge_eur / 10000')
            tx_cms_eur_mg_f = np.maximum(tx_cms_eur_mg, self.cls_ra_rn_params.tx_floor / 10000)

            rate_arb[is_chf] = tx_cms_chf_mg
            rate_arb[~is_chf] = tx_cms_eur_mg_f

            drac_nan = np.isnan(drac_rarn)
            rate_arb = ne.evaluate('where(current_month < mois_depart_rarn, nan, rate_arb)')
            rate_arb = ne.evaluate('where(drac_nan, nan, rate_arb)')
        else:
            rate_arb = np.zeros((n, t))

        return rate_arb

    def get_rate_renego_from_cms(self, data_hab, drac_rarn, current_month, mois_depart_rarn, renego_model, t):

        is_renego = (renego_model == self.cls_ra_rn_params.MODEL_NAME_RENEGO_IMMO).values
        n = is_renego[is_renego].shape[0]

        if n > 0:

            current_month_r = current_month[is_renego]
            mois_depart_rarn_r = mois_depart_rarn[is_renego]
            drac_rarn_r = drac_rarn[is_renego]

            indice_tenor = np.maximum(0, np.nan_to_num(drac_rarn_r) - 1)
            indice_tenor = np.minimum(self.cls_hz_params.max_projection - 1, indice_tenor).astype(int)
            indice_mois_fwd = np.minimum(current_month_r, self.cls_hz_params.max_projection - 1)

            is_chf = np.array(data_hab.loc[is_renego, self.cls_fields.NC_LDP_CURRENCY] == "CHF")
            rate_renego = np.zeros((n, t))
            spread_renego = self.cls_ra_rn_params.spread_renego / 10000

            tx_cms_chf = self.dic_tx_swap["CHFLIBOR"][1:, 1:][indice_tenor[is_chf], indice_mois_fwd[is_chf] - 1]
            tx_cms_chf_f = np.maximum(tx_cms_chf, 0)
            tx_cms_chf_mg_f = ne.evaluate('tx_cms_chf_f + spread_renego')

            if self.cls_proj.OLD_RENEGO:
                tx_cms_eur = self.dic_tx_swap["EURIBOR"][1:, 1:][indice_tenor[~is_chf], indice_mois_fwd[~is_chf] - 1]
            else:
                tx_cms_eur = self.dic_tx_swap["EURIBMOY"][1:, 1:][indice_tenor[~is_chf], indice_mois_fwd[~is_chf] - 1]
            tx_cms_eur_f = np.maximum(tx_cms_eur, 0)
            tx_cms_eur_mg_f = ne.evaluate('tx_cms_eur_f + spread_renego')

            rate_renego[is_chf] = tx_cms_chf_mg_f
            rate_renego[~is_chf] = tx_cms_eur_mg_f

            rate_renego = ne.evaluate('where(current_month_r < mois_depart_rarn_r, 0, rate_renego)')
            drac_nan = np.isnan(drac_rarn_r)
            rate_renego = ne.evaluate('where(drac_nan, 0, rate_renego)')

        else:
            rate_renego = np.full((n, t), np.nan)

        return rate_renego, is_renego

    def apply_ra_back_test(self, tx_ra, drac_cases):
        tx_ra = np.select(drac_cases,
                          [tx_ra + self.cls_ra_rn_params.drac50_ra, tx_ra + self.cls_ra_rn_params.drac80_ra],
                          default=tx_ra + self.cls_ra_rn_params.drac100_ra)
        return tx_ra

    def apply_rn_back_test(self, tx_rn, drac_cases):
        tx_rn = np.select(drac_cases,
                          [tx_rn + self.cls_ra_rn_params.drac50_rn, tx_rn + self.cls_ra_rn_params.drac80_rn],
                          default=tx_rn + self.cls_ra_rn_params.drac100_rn)
        return tx_rn
