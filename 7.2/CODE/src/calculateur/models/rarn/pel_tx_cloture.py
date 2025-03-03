from calendar import monthrange
import numpy as np
import pandas as pd
import numexpr as ne
from utils import general_utils as gu
import logging

logger = logging.getLogger(__name__)
np.warnings.filterwarnings("ignore", category=RuntimeWarning)

class TAUX_CLOTURE_Manager():
    def __init__(self, cls_proj, cls_format, cls_model_params):
        self.cls_proj = cls_proj
        self.cls_rate = cls_proj.cls_rate
        self.cls_cal = cls_proj.cls_cal
        self.cls_ra_params = cls_model_params.cls_ra_rn_params
        self.cls_hz_params = cls_proj.cls_hz_params
        self.cls_fields = cls_proj.cls_fields
        self.curve_name_arbi = cls_model_params.cls_ra_rn_params.curve_name_arbi
        self.tenor_arbi = cls_model_params.cls_ra_rn_params.tenor_arbi
        self.max_pn = cls_model_params.cls_ra_rn_params.max_pn
        self.cls_format = cls_format

    ###############@profile
    def get_taux_cloture(self, data, tx_params):
        filter_cloture = data[self.cls_fields.NC_LDP_REDEMPTION_MODEL].fillna("").str.contains("RETRAITS_PEL").values
        t = self.cls_proj.t
        c = 1000
        is_fwd = (data[self.cls_fields.NC_LDP_TRADE_DATE] - self.cls_hz_params.dar_mois > 0).values
        data_pel_pn = data[filter_cloture & is_fwd]
        data_pel_st = data[filter_cloture & ~is_fwd]
        self.pel_rates_fwd = self.cls_proj.cls_rate.sc_rates[filter_cloture & is_fwd]
        self.pel_rates_non_fwd = self.cls_proj.cls_rate.sc_rates[filter_cloture & ~is_fwd]

        value_date = (data_pel_st[self.cls_fields.NC_LDP_VALUE_DATE] - self.cls_hz_params.dar_mois).values
        n = value_date.shape[0]
        tombee_mtx = np.where(self.cls_cal.current_month[filter_cloture & ~is_fwd] > value_date.reshape(n, 1), 1, 0)

        tx_rarn_all = np.zeros((self.cls_proj.n, t))
        tx_E5Y = self.load_taux_arbitrage(tx_params)
        if len(data_pel_st) > 0:
            tx_rarn_st = np.zeros((data_pel_st.shape[0] , t))
            for k in range(0, data_pel_st.shape[0] // c + 1):
                data_k = data_pel_st.iloc[c * k:c * (k + 1)].copy()
                n_k = data_k.shape[0]
                if n_k > 0:
                    data_proj_age, age = self.get_age(data_k, False, n_k, t)
                    non_pfu = self.load_pfu_condition(data_k, False, n_k)
                    tx_PEL = self.load_taux_pel(False, c, k, t)
                    mtx_ln_tx_struct = self.get_mtx_tx_struct_by_contract(data_proj_age, self.cls_ra_params.mtx_tx_struct)
                    current_month = self.generate_months_mtx(data_k, n_k, t)
                    tx_rarn_k = self.compute_taux_cloture(age, non_pfu, tx_E5Y, tx_PEL, mtx_ln_tx_struct, n_k, t, current_month)
                    tx_rarn_st[c * k:c * (k + 1)] = tx_rarn_k * tombee_mtx[c * k:c * (k + 1)]

            tx_rarn_all[filter_cloture & ~is_fwd] = tx_rarn_st

        if len(data_pel_pn) > 0:
            tx_rarn_pn = np.zeros((data_pel_pn.shape[0] , t))
            for k in range(0, data_pel_pn.shape[0]  // c + 1):
                data_k = data_pel_pn.iloc[c * k:c * (k + 1)].copy()
                n_k = data_k.shape[0]
                if n_k > 0:
                    data_proj_age, age = self.get_age(data_k, True, n_k, t)
                    non_pfu = self.load_pfu_condition(data_k, True, n_k)
                    tx_PEL = self.load_taux_pel(True,c, k, t)
                    mtx_ln_tx_struct = self.get_mtx_tx_struct_by_contract(data_proj_age, self.cls_ra_params.mtx_tx_struct)
                    current_month = self.generate_months_mtx(data_k, n_k, t)
                    tx_rarn_k = self.compute_taux_cloture(age, non_pfu, tx_E5Y, tx_PEL, mtx_ln_tx_struct, n_k, t, current_month)
                    if self.cls_proj.pel_age_params_begin_at_1:
                        tx_rarn_k =  ne.evaluate("where(age <= 0, 0, tx_rarn_k)")
                    else:
                        tx_rarn_k =  ne.evaluate("where(age < 0, 0, tx_rarn_k)")
                    tx_rarn_pn[c * k:c * (k + 1)]  = tx_rarn_k

            tx_rarn_all[filter_cloture & is_fwd] = tx_rarn_pn

        self.tx_rn, self.tx_rarn, self.tx_ra = np.zeros((self.cls_proj.n, t)), tx_rarn_all, tx_rarn_all
        self.rate_renego = np.zeros((self.cls_proj.n, t))

    ###############@profile
    def compute_taux_cloture(self, age, non_pfu, tx_E5Y, tx_PEL, mtx_ln_tx_struct, nb_c, nb_proj, cur_month):
        tx_E5Y_all = tx_E5Y[:, cur_month].reshape(cur_month.shape[0], nb_proj)

        tx_cotis_soc = self.cls_ra_params.tx_cot_soc
        tx_impots_revenu = self.cls_ra_params.tx_fisc
        tx_survie = self.cls_ra_params.tx_survie

        ln_tx_struct, ln_tx_struct_aps, ln_tx_struct_avs, ln_tx_struct_sm2, ln_tx_struct_sm1, beta, mu, u = \
            self.get_all_tx_struct_params_shaped(mtx_ln_tx_struct, nb_c, nb_proj)

        """ 1. TAUX DE SURVIE SI AGE PEL >= 240 mois """
        ones = np.ones((nb_c, nb_proj))
        tx_cloture = ne.evaluate("ones * log(tx_survie)")

        """ 2. SANS ARBITRAGE (AGE PEL < 48mois et AGE PEL_CAT < 120 mois) """
        no_ln_struct = ~np.isnan(ln_tx_struct)
        tx_cloture = ne.evaluate("where(no_ln_struct, ln_tx_struct, tx_cloture)")

        """ 3. AVEC ARBITRAGE (AGE PEL >= 48 mois et AGE PEL_CAT >= 120 mois) """
        tx_E5Y_api = ne.evaluate("tx_E5Y_all * (1 - tx_cotis_soc) * (1 - tx_impots_revenu)")
        cond_non_pfu = (age <= self.cls_ra_params.limit_age_pfu) & non_pfu
        tx_PEL_api = ne.evaluate("where(cond_non_pfu, tx_PEL * (1 - tx_cotis_soc),"
                                 " tx_PEL * (1 - (tx_cotis_soc + tx_impots_revenu)))")
        diff_taux = ne.evaluate("(tx_E5Y_api - tx_PEL_api) * 100")

        cases = [diff_taux < mu - self.cls_ra_params.strike_2pt,
                 (diff_taux < mu - self.cls_ra_params.strike_1pt) & (diff_taux >= mu - self.cls_ra_params.strike_2pt),
                 (diff_taux < mu) & (diff_taux >= mu - self.cls_ra_params.strike_1pt), diff_taux >= mu]

        values = [ln_tx_struct_avs, ln_tx_struct_sm2, ln_tx_struct_sm1,
                  ne.evaluate("ln_tx_struct_aps + -beta * exp(-u / ((diff_taux - mu)))")]

        no_aps = ~np.isnan(ln_tx_struct_aps)
        aps_and_no_avs = ~np.isnan(ln_tx_struct_avs) & np.isnan(ln_tx_struct_aps)
        tx_cloture = np.where(no_aps, np.select(cases, values), tx_cloture)
        tx_cloture = ne.evaluate("where(aps_and_no_avs, ln_tx_struct_avs, tx_cloture)")

        no_tx_cloture = np.isnan(tx_cloture)
        tx_cloture = ne.evaluate("where(no_tx_cloture, 0, tx_cloture)")

        tx_cloture = ne.evaluate("1 - exp(tx_cloture)")

        if self.cls_proj.make_dar_pel_daily:
            if self.cls_ra_params.type_data == "stock":
                tx_cloture = self.mensualiser_tx_dar(tx_cloture)
            else:
                tx_cloture[:, 0] = 0

        return tx_cloture


    def mensualiser_tx_dar(self, data):
        ratio_1j = 1 / monthrange(self.cls_hz_params.dar_usr.year, self.cls_hz_params.dar_usr.month)[1]
        tx_clot_1st_month = data[:, 0]
        data[:, 0] = ne.evaluate("(1 - (1 - tx_clot_1st_month) ** (ratio_1j))")
        return data


    def get_all_tx_struct_params_shaped(self, data_ln_tx_struct, nb_c, nb_proj):
        ln_tx_struct = np.array(data_ln_tx_struct[self.cls_ra_params.NC_LN_TX_STRUCT_MTS]).reshape(nb_c, nb_proj)
        ln_tx_struct_aps = np.array(data_ln_tx_struct[self.cls_ra_params.NC_LN_TX_STRUCT_APS_MTS]).reshape(nb_c, nb_proj)
        ln_tx_struct_avs = np.array(data_ln_tx_struct[self.cls_ra_params.NC_LN_TX_STRUCT_AVS_MTS]).reshape(nb_c, nb_proj)
        ln_tx_struct_sm2 = np.array(data_ln_tx_struct[self.cls_ra_params.NC_LN_TX_STRUCT_SM2_MTS]).reshape(nb_c, nb_proj)
        ln_tx_struct_sm1 = np.array(data_ln_tx_struct[self.cls_ra_params.NC_LN_TX_STRUCT_SM1_MTS]).reshape(nb_c, nb_proj)

        beta = np.array(data_ln_tx_struct[self.cls_ra_params.NC_BETA]).reshape(nb_c, nb_proj)
        mu = np.array(data_ln_tx_struct[self.cls_ra_params.NC_MU]).reshape(nb_c, nb_proj)
        u = np.array(data_ln_tx_struct[self.cls_ra_params.NC_U]).reshape(nb_c, nb_proj)

        return ln_tx_struct, ln_tx_struct_aps, ln_tx_struct_avs, ln_tx_struct_sm2, ln_tx_struct_sm1, beta, mu, u

    def get_age(self, data, is_fwd, n, t):
        data[self.cls_ra_params.NC_AGE] = (self.cls_format.nb_months_in_date(self.cls_hz_params.dar_usr)
                                           - data[self.cls_fields.NC_LDP_TRADE_DATE])
        if self.cls_proj.pel_age_params_begin_at_1:
            if is_fwd:
                data[self.cls_ra_params.NC_AGE] = data[self.cls_ra_params.NC_AGE] + 1
        data_proj_age = self.create_pel_contract_by_projected_age(data, t, n)
        age = np.array(data_proj_age[self.cls_ra_params.NC_AGE].copy())
        age = age.reshape(n, t)
        return data_proj_age, age

    def create_pel_contract_by_projected_age(self, data_pel, nb_proj, nb_c):
        data_pel["CONTRAT_GEN"] = np.where(data_pel[self.cls_fields.NC_LDP_CONTRACT_TYPE].str.contains("-C"), "P-PEL-C", "P-PEL")
        cols_keep = ["CONTRAT_GEN", self.cls_fields.NC_LDP_CONTRACT_TYPE, self.cls_ra_params.NC_AGE]
        data_pel_proj_age = data_pel[cols_keep].copy()
        data_pel_proj_age = gu.strip_and_upper(data_pel_proj_age, [self.cls_fields.NC_LDP_CONTRACT_TYPE, "CONTRAT_GEN"])
        data_pel_proj_age = pd.DataFrame(np.repeat(data_pel_proj_age.values, nb_proj, axis=0), columns=cols_keep)

        age_vals = data_pel_proj_age[self.cls_ra_params.NC_AGE].values.astype(int)
        dyn_month = np.tile(np.arange(0, nb_proj), nb_c)
        data_pel_proj_age[self.cls_ra_params.NC_AGE] = ne.evaluate("age_vals + dyn_month")
        return data_pel_proj_age

    def get_mtx_tx_struct_by_contract(self, data_pel, mtx_tx_struct):
        data_pel = self.join_data_pel_mapping_contract(data_pel.copy())
        cols_keys = ["CONTRAT_GEN", self.cls_ra_params.NC_AGE,  self.cls_ra_params.NC_CLE_CC]
        orig_cols = [x for x in data_pel.columns]
        mtx_tx_struct_all = gu.map_with_combined_key(data_pel, mtx_tx_struct, cols_keys, symbol_any="*", \
                                                  filter_comb=True, necessary_cols=2, map_null_values=False,
                                                  strip_upper=False)
        cols_out = [x for x in mtx_tx_struct_all.columns if x not in orig_cols]
        mtx_tx_struct_all = mtx_tx_struct_all.reset_index(drop=True)[cols_out].copy()
        return mtx_tx_struct_all

    def join_data_pel_mapping_contract(self, data_pel):
        data_pel[self.cls_ra_params.NC_AGE] = [str(int(x)) for x in data_pel[self.cls_ra_params.NC_AGE]]
        data_pel = data_pel.join(self.cls_ra_params.mapping_contrats_cle, how='left', \
                                 on=["CONTRAT_GEN", self.cls_fields.NC_LDP_CONTRACT_TYPE])
        if data_pel[self.cls_ra_params.NC_CLE_CC].isnull().any():
            contracts_with_missing_keys = data_pel[data_pel[self.cls_ra_params.NC_CLE_CC].isnull()][self.cls_fields.NC_LDP_CONTRACT_TYPE].unique().tolist()
            msg = "Il y a des contrats PEL sans clé de génération : %s " % contracts_with_missing_keys
            raise ValueError(msg)
        return data_pel

    def load_pfu_condition(self, data_pel, is_fwd, n_k):
        if not is_fwd:
            non_pfu = np.array(data_pel[self.cls_fields.NC_LDP_TRADE_DATE]
                               <= self.cls_ra_params.date_pfu).reshape(n_k, 1)
        else:
            non_pfu = np.array([[[False]] * n_k]).reshape(n_k, 1)

        return non_pfu

    def load_taux_pel(self, is_fwd, c, k, t):
        if not is_fwd:
            tx_PEL = self.pel_rates_non_fwd[c * k:c * (k + 1):, :t]
        else:
            tx_PEL = self.pel_rates_fwd[c * k:c * (k + 1):, :t]
        return tx_PEL

    def load_taux_arbitrage(self, tx_params):
        scenario_curves_df = tx_params["curves_df"]["data"]
        cols_num = tx_params["curves_df"]["cols"]
        col_curve = tx_params["curves_df"]["curve_code"]
        col_tenor = tx_params["curves_df"]["tenor"]
        t = self.cls_proj.t
        cols = [x for x in cols_num if int(x[1:]) in range(1, min(t, self.cls_hz_params.max_projection, self.cls_hz_params.max_taux_cms) + 1)]
        filter_tx_pel = (scenario_curves_df[col_curve] == self.curve_name_arbi) & (scenario_curves_df[col_tenor] == self.tenor_arbi)
        tx_E5Y = np.array(scenario_curves_df[filter_tx_pel][cols])
        tx_E5Y = np.resize(tx_E5Y, (1, t))
        if tx_E5Y.shape[1] >= self.cls_hz_params.max_projection:
            tx_E5Y[:, self.cls_hz_params.max_projection:] = tx_E5Y[:, self.cls_hz_params.max_projection - 1]

        if len(tx_E5Y) == 0:
            logger.error("No index name %s is present in the data" % self.index_arbitrage)
            raise ValueError

        return tx_E5Y

    def generate_months_mtx(self, data_pel, nb_c, nb_proj):
        current_month = (np.arange(nb_proj)).reshape(1, nb_proj)
        return current_month

