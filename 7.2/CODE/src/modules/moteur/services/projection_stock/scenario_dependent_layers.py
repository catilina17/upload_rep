import modules.moteur.parameters.general_parameters as gp
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import numpy as np
from modules.moteur.services.indicateurs_taux.gap_taux_agreg import Gap_Taux_Calculator as gp_tx_ag
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Stock_Updater():

    def __init__(self, cls_usr, cls_mp):
        self.up = cls_usr
        self.mp = cls_mp
        self.num_cols = ["M" + str(i) for i in range(0, self.up.nb_mois_proj_usr + 1)]

    def update_stock_with_calculator_results(self, data_st, cls_stock_calc):
        calculated_stock = cls_stock_calc.calculated_stock
        if len(calculated_stock) > 0:
            data_st["key_update"] = (data_st[pa.NC_PA_CLE_OUTPUT + [pa.NC_PA_IND03]]
                                     .apply(lambda row: '_'.join(row.values.astype(str)), axis=1))
            data_st["key_update"] = data_st["key_update"].str.upper()
            data_st = data_st.set_index("key_update")
            data_st["UPDATED"] = False
            if self.up.type_simul["EVE"] or self.up.type_simul["EVE_LIQ"]:
                calculated_stock\
                    = self.calculate_lmn_gp_reg(calculated_stock, "LMN", "LMN_GPTX", "LMN_GPRG", self.num_cols,
                                                self.mp.mapping_gp_reg_params, pa.NC_PA_RATE_CODE)
                calculated_stock = self.calculate_lmn_gp_reg(calculated_stock, "LMN_FTP", "LMN_FTP_GPTX",
                                                             "LMN_GPRG" + "_FTP", self.num_cols,
                                                             self.mp.mapping_gp_reg_params, pa.NC_PA_RATE_CODE)
                calculated_stock = self.calculate_lmn_eve(calculated_stock, self.mp.mapping_eve,
                                                          pa.NC_PA_RATE_CODE, pa.NC_PA_ETAB, pa.NC_PA_CONTRACT_TYPE,
                                                          pa.NC_PA_DIM2, "LMN", "LMN_FTP")

            data_st = self.update_recalc_st(data_st, calculated_stock)
        else:
            data_st["UPDATED"] = False
        return data_st

    @staticmethod
    def calculate_lmn_gp_reg(data, ind_mni, ind_mni_gptx, ind_fin, num_cols, mapping_gp_reg, rate_code_col):

        data_on_ind = data[data[pa.NC_PA_IND03] == ind_mni].copy()
        mni = data.loc[data[pa.NC_PA_IND03] == ind_mni, num_cols].values
        mni_gptx = data.loc[data[pa.NC_PA_IND03] == ind_mni_gptx, num_cols].values

        coeff_gp_inf, filtre_inf = gp_tx_ag.get_coeff_gp_inf(mapping_gp_reg, data_on_ind, rate_code_col)
        coeff_index_ntf = gp_tx_ag.get_coeff_gp_tf(mapping_gp_reg, data_on_ind, rate_code_col)

        mni_gptx_adj = mni_gptx + coeff_index_ntf * (mni - mni_gptx)
        mni_inf = np.where(filtre_inf, mni - mni_gptx, 0)
        mni_inf = mni_inf + coeff_gp_inf * (mni - mni_gptx)
        mni_gp_reg = mapping_gp_reg["coeff_reg_tf_usr"] * mni_gptx_adj + mapping_gp_reg["coeff_reg_inf_usr"] * mni_inf

        data_qual = data_on_ind[[x for x in data_on_ind.columns if not x in num_cols]]
        data_mni_gp_reg = pd.concat([data_qual, pd.DataFrame(mni_gp_reg, columns=num_cols, index=data_qual.index)],
                                    axis=1)
        data_mni_gp_reg[pa.NC_PA_IND03] = ind_fin

        return pd.concat([data, data_mni_gp_reg])

    @staticmethod
    def calculate_lmn_eve(data, mapping_eve, rate_code_col, etab_col, contract_type_col, dim2_col,
                          lmn_ind, lmn_ftp_ind):
        ind_mni = pa.NC_PA_LMN_EVE
        filtre_all_lmn = data[pa.NC_PA_IND03].str.contains("LMN").values
        data_eve = data[filtre_all_lmn].copy()

        filtre_crif = ((data_eve[rate_code_col].str.contains("FIXE"))
                       & ((data_eve[contract_type_col].isin(gp.CONTRATS_RA_RN_IMMO))
                          | ((data_eve[contract_type_col].isin(gp.CONTRATS_RA_RN_IMMO_CSDN)) & (
                                data_eve[etab_col] == "CSDN")))).values

        filtre_crif = filtre_crif & np.full(filtre_crif.shape[0], mapping_eve["mode_cal_gap_tx_immo"] == "GAP LIQ")

        filtre_tci = ((data_eve[dim2_col].isin(mapping_eve["cc_tci"])).values
                      & ~(data_eve[contract_type_col].isin(mapping_eve["cc_tci_excl"])).values)

        filtre_cap_floor = data_eve[contract_type_col].str.contains("HB-CAP|HB-FLOOR").values.astype(bool)

        filtre_lmn_ftp = (data_eve[pa.NC_PA_IND03] == lmn_ftp_ind).values
        filtre_lmn_ftp_gprg = (data_eve[pa.NC_PA_IND03] == "LMN_GPRG_FTP").values
        filtre_lmn_gprg = (data_eve[pa.NC_PA_IND03] == "LMN_GPRG").values
        filtre_lmn = (data_eve[pa.NC_PA_IND03] == lmn_ind).values

        cases_lmn_eve = [filtre_crif & ~filtre_tci & filtre_lmn,
                         filtre_crif & filtre_tci & filtre_lmn_ftp,
                         filtre_cap_floor & filtre_lmn,
                         filtre_tci & filtre_lmn_ftp_gprg,
                         ~filtre_crif & ~filtre_tci & ~filtre_cap_floor & filtre_lmn_gprg]
        values_lmn_eve = [ind_mni] * len(cases_lmn_eve)
        data_eve[pa.NC_PA_IND03] = np.select(cases_lmn_eve, values_lmn_eve,
                                                default=data_eve[pa.NC_PA_IND03].values)
        data_eve = data_eve[data_eve[pa.NC_PA_IND03] == ind_mni].copy()
        data = pd.concat([data, data_eve])

        return data

    def update_recalc_st(self, data_stock, data_recalc):
        global dic_sc_dav, dic_recalc_st_data_liq, dic_recalc_st_data_eve, dic_recalc_st_data_eve_liq
        global is_ra_rn

        data_recalc_ind \
            = data_recalc[data_recalc[pa.NC_PA_IND03].isin([pa.NC_PA_LEF, pa.NC_PA_LEM, pa.NC_PA_TEF, pa.NC_PA_TEM,
                                                            pa.NC_PA_LMN, gp.pa.NC_PA_LMN_EVE])].copy()

        data_recalc_ind["key_update"] = (data_recalc_ind[pa.NC_PA_CLE_OUTPUT + [pa.NC_PA_IND03]]
                                         .apply(lambda row: '_'.join(row.values.astype(str)), axis=1))
        data_recalc_ind["key_update"] = data_recalc_ind["key_update"].str.upper()
        data_recalc_ind = data_recalc_ind.set_index("key_update")
        data_recalc_ind = data_recalc_ind.loc[~data_recalc_ind.index.duplicated(keep='first')]

        if len(data_recalc_ind) > 0:
            filtre_ind = data_recalc_ind[pa.NC_PA_IND03].isin(
                [pa.NC_PA_TEF, pa.NC_PA_TEM, gp.tef_eve_sti, gp.tem_eve_sti])
            filtre_floor = (data_recalc_ind[pa.NC_PA_CONTRACT_TYPE].str.contains('|'.join(["HB-FLOOR", "HB-CAP"])))
            filtre_ind_liq = ~((filtre_ind & filtre_floor))
            data_stock.update(data_recalc_ind[filtre_ind_liq])
            data_stock["UPDATED"] = np.where(
                data_stock.index.isin(data_recalc_ind[filtre_ind_liq].index.unique().tolist()), True, False)
            unfound_recalc = (~data_recalc_ind[filtre_ind_liq].index.isin(data_stock.index.unique().tolist()))
            if unfound_recalc.any():
                logger.warning("Il y a des éléments du stock recalculé qu'on ne retrouve pas dans le stock RCO : %s" %
                               data_recalc_ind[filtre_ind_liq].index[unfound_recalc])
                # raise ValueError("Il y a des éléments du stock recalculé qu'on ne retrouve pas dans le stock RCO : %s" % data_recalc_ind[filtre_ind_liq][unfound_recalc].index.unique().tolist())

        return data_stock



