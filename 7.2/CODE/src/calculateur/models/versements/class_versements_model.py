import pandas as pd
import numpy as np
import numexpr as ne
from utils import general_utils as gu


class Versements_PEL():
    def __init__(self, cls_data_versements):
        self.data_vrst = cls_data_versements
        self.dar_usr = cls_data_versements.cls_hz_params.dar_usr
        self.dar_mois = cls_data_versements.cls_hz_params.dar_mois
        self.cls_fields = cls_data_versements.cls_fields
        self.nb_mois_proj = cls_data_versements.nb_mois_proj
        self.cls_ra_model_params = cls_data_versements.cls_ra_model_params
        self.mapping_contrats_cle = self.cls_ra_model_params.mapping_contrats_cle
        self.max_pn = cls_data_versements.cls_ra_model_params.max_pn

    def get_mtx_versements_by_contract(self, data_pel, mtx_versements, n, t):
        data_pel = self.join_data_pel_mapping_contract(data_pel.copy())
        cols_keys = ["CONTRAT_GEN", self.data_vrst.NC_AGE, self.data_vrst.NC_CLE_CC]
        orig_cols = [x for x in data_pel.columns]
        mtx_versements_all = data_pel.join(mtx_versements, on=cols_keys)
        cols_out = [x for x in mtx_versements_all.columns if x not in orig_cols]
        mtx_versements_all = np.array(mtx_versements_all[cols_out].copy().fillna(0))
        return mtx_versements_all.reshape(n, t)[:, :self.data_vrst.max_age_versement]

    def get_age(self, clas_proj, data, is_fwd, n, t):
        data[self.data_vrst.NC_AGE] = self.dar_mois - data[self.cls_fields.NC_LDP_TRADE_DATE]
        if clas_proj.pel_age_params_begin_at_1 and is_fwd:
            data[self.data_vrst.NC_AGE] = data[self.data_vrst.NC_AGE] + 1
        data_proj_age = self.create_pel_contract_by_projected_age(data.copy(), t, n)
        age = np.array(data_proj_age[self.data_vrst.NC_AGE].copy())
        age = age.reshape(n, t)
        return data_proj_age, age

    def create_pel_contract_by_projected_age(self, data_pel, nb_proj, nb_c):
        data_pel["CONTRAT_GEN"] = np.where(data_pel[self.cls_fields.NC_LDP_CONTRACT_TYPE].str.contains("-C"), "P-PEL-C",
                                           "P-PEL")
        cols_keep = ["CONTRAT_GEN", self.cls_fields.NC_LDP_CONTRACT_TYPE, self.data_vrst.NC_AGE]
        data_pel_proj_age = data_pel[cols_keep].copy()
        data_pel_proj_age = gu.strip_and_upper(data_pel_proj_age, [self.cls_fields.NC_LDP_CONTRACT_TYPE, "CONTRAT_GEN"])
        data_pel_proj_age = pd.DataFrame(np.repeat(data_pel_proj_age.values, nb_proj, axis=0), columns=cols_keep)

        age_vals = data_pel_proj_age[self.data_vrst.NC_AGE].values.astype(int)
        dyn_month = np.tile(np.arange(0, nb_proj), nb_c)
        data_pel_proj_age[self.data_vrst.NC_AGE] = ne.evaluate("age_vals + dyn_month")

        return data_pel_proj_age

    def join_data_pel_mapping_contract(self, data_pel):
        data_pel[self.data_vrst.NC_AGE] = [str(int(x)) for x in data_pel[self.data_vrst.NC_AGE]]
        data_pel = data_pel.join(self.mapping_contrats_cle, how='left', \
                                 on=["CONTRAT_GEN", self.cls_fields.NC_LDP_CONTRACT_TYPE])
        if data_pel[self.data_vrst.NC_CLE_CC].isnull().any():
            contracts_with_missing_keys = data_pel[data_pel[self.data_vrst.NC_CLE_CC].isnull()][self.cls_fields.NC_LDP_CONTRACT_TYPE].unique().tolist()
            msg = "Il y a des contrats PEL sans clé de génération : %s " % contracts_with_missing_keys
            raise ValueError(msg)

        return data_pel

    # ################@profile
    def create_versements_contracts(self, clas_proj):
        filter_pel_vers = (
            clas_proj.data_ldp[self.cls_fields.NC_LDP_SAVINGS_MODEL].fillna("").str.contains("VERSEMENTS_P-PEL")).values
        t = clas_proj.t
        c = min(self.data_vrst.max_age_versement + 1, t + 1)
        nb = 1000
        self.ec_versements_all = np.zeros((clas_proj.n, clas_proj.t + 1))
        is_fwd = (clas_proj.data_ldp[self.cls_fields.NC_LDP_VALUE_DATE] - self.dar_mois > 0).values
        is_fwd_trade_date = (clas_proj.data_ldp[self.cls_fields.NC_LDP_TRADE_DATE] - self.dar_mois > 0).values
        data_pel_vrst_pn = clas_proj.data_ldp[filter_pel_vers & is_fwd & is_fwd_trade_date]
        data_pel_vrst_st = clas_proj.data_ldp[filter_pel_vers & ~is_fwd & ~is_fwd_trade_date]
        data_pel_vrst_col_decol = clas_proj.data_ldp[filter_pel_vers & is_fwd & ~is_fwd_trade_date]

        value_date = (data_pel_vrst_col_decol[self.cls_fields.NC_LDP_VALUE_DATE] - self.dar_mois).values
        n = value_date.shape[0]
        tombee_mtx = np.where(clas_proj.cls_cal.current_month[filter_pel_vers & is_fwd & ~is_fwd_trade_date]
                              > value_date.reshape(n, 1), 1, 0)

        if len(data_pel_vrst_st) > 0:
            self.ec_versements_st = np.zeros((data_pel_vrst_st.shape[0], c - 1))
            for k in range(0, data_pel_vrst_st.shape[0] // nb + 1):
                data_k = data_pel_vrst_st.iloc[nb * k:nb * (k + 1)].copy()
                n_k = data_k.shape[0]
                if n_k > 0:
                    data_proj_age, age = self.get_age(clas_proj, data_k.copy(), False, n_k, t)
                    ec_versements_k = self.get_mtx_versements_by_contract(data_proj_age, self.data_vrst.mtx_versements, n_k, t)
                    if clas_proj.make_dar_pel_daily:
                        ec_vers_m0 = ec_versements_k[:, 0]
                        ec_versements_k[:, 0] = ne.evaluate("1/30 * ec_vers_m0")
                    self.ec_versements_st[nb * k:nb * (k + 1)] = ec_versements_k

            self.ec_versements_all[filter_pel_vers & ~is_fwd & ~is_fwd_trade_date, 1: c] = self.ec_versements_st

        if len(data_pel_vrst_pn) > 0:
            self.ec_versements_pn = np.zeros((data_pel_vrst_pn.shape[0], c - 1))
            for k in range(0, data_pel_vrst_pn.shape[0] // nb + 1):
                data_k = data_pel_vrst_pn.iloc[nb * k:nb * (k + 1)].copy()
                n_k = data_k.shape[0]
                if n_k > 0:
                    data_proj_age, age = self.get_age(clas_proj, data_k.copy(), True, n_k, t)
                    ec_versements_k = self.get_mtx_versements_by_contract(data_proj_age, self.data_vrst.mtx_versements, n_k, t)
                    age_c_1 = age[:, :c - 1]
                    if clas_proj.pel_age_params_begin_at_1:
                        ec_versements_k = ne.evaluate("where(age_c_1 <= 0, 0, ec_versements_k)")
                    else:
                        ec_versements_k = ne.evaluate("where(age_c_1 < 0, 0, ec_versements_k)")
                    self.ec_versements_pn[nb * k:nb * (k + 1)] = ec_versements_k

            self.ec_versements_all[filter_pel_vers & is_fwd & is_fwd_trade_date, 1: c] = self.ec_versements_pn


        if len(data_pel_vrst_col_decol) > 0:
            self.ec_versements_cd = np.zeros((data_pel_vrst_col_decol.shape[0], c - 1))
            for k in range(0, data_pel_vrst_col_decol.shape[0] // nb + 1):
                data_k = data_pel_vrst_col_decol.iloc[nb * k:nb * (k + 1)].copy()
                n_k = data_k.shape[0]
                if n_k > 0:
                    data_proj_age, age = self.get_age(clas_proj, data_k.copy(), False, n_k, t)
                    ec_versements_k = self.get_mtx_versements_by_contract(data_proj_age, self.data_vrst.mtx_versements, n_k, t)
                    self.ec_versements_cd[nb * k:nb * (k + 1)] = ec_versements_k * tombee_mtx[nb * k:nb * (k + 1)][:, :c - 1]

            self.ec_versements_all[filter_pel_vers & is_fwd & ~is_fwd_trade_date, 1: c] = self.ec_versements_cd






