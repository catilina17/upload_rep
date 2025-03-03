import numpy as np
import pandas as pd
import modules.alim.parameters.general_parameters as gp
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import mappings.mapping_functions as mp
import mappings.general_mappings as gma
import modules.alim.parameters.user_parameters as up
import modules.alim.parameters.RZO_params as rzo_p
import utils.general_utils as gu

np.seterr(divide='ignore', invalid='ignore')

class PN_BILAN_CONSTANT_Formater():
    def __init__(self, cls_usr):
        self.up = cls_usr
            
    def map_and_calculate_duree(self, data_ech):
        mapping_ech = gma.mapping_PN["mapping_CONSO_ECH"]
    
        data_ech = mp.map_data(data_ech.drop(columns=[pa.NC_PA_isECH], axis=1), mapping_ech, keys_data=[pa.NC_PA_CONTRACT_TYPE],
                               name_mapping="DATA STOCK vs. ")
    
        data_ech[pa.NC_PA_ACCRUAL_BASIS] = data_ech[pa.NC_PA_ACCRUAL_BASIS].astype(str)
        data_ech[pa.NC_PA_RELEASING_RULE] = data_ech[pa.NC_PA_RELEASING_RULE].fillna("")
    
        if self.up.current_etab in gp.MTY_DETAILED_ETAB:
            mapping_duree = gma.map_pass_alm["MTY DETAILED"]
            map_MTY = mapping_duree["TABLE"].loc[:, mapping_duree["OUT"]].drop_duplicates()
            data_ech = mp.map_data(data_ech, map_MTY, keys_mapping=[pa.NC_PA_MATUR],
                                   keys_data=[pa.NC_PA_MATUR], \
                                   cols_mapp=[pa.NC_PA_MATURITY_DURATION], option="", no_map_value=0,
                                   name_mapping="DATA STOCK vs. " + mapping_duree["FULL_NAME"])
    
        else:  # DUREE= DURATION
            duree_flux = [int(str(x).replace("M", "")) for x in pa.NC_PA_COL_SORTIE_NUM_ST] + [25 * 12]
            duree_flux = np.array(duree_flux[1:])
            data_lef = np.array(data_ech[pa.NC_PA_COL_SORTIE_NUM_ST])
            flux_lef = data_lef[:, 0:-1] - data_lef[:, 1:]
            flux_lef = np.column_stack([flux_lef, data_lef[:, -1]])
            somme_flux_lef = flux_lef.sum(axis=1)
            somme_flux_lef = np.where(somme_flux_lef == 0, 1, somme_flux_lef)
            filtre_sum0 = (somme_flux_lef == 0)
            duree = np.nan_to_num(
                np.divide((flux_lef * duree_flux.reshape((1, duree_flux.shape[0]))).sum(axis=1), somme_flux_lef), posinf=0,
                neginf=0)
            data_ech[pa.NC_PA_MATURITY_DURATION] = np.where(filtre_sum0, 1, duree)
            data_ech[pa.NC_PA_MATURITY_DURATION] = data_ech[pa.NC_PA_MATURITY_DURATION].mask(data_ech[pa.NC_PA_AMORTIZING_TYPE] != "INFINE",
                                                               data_ech[pa.NC_PA_MATURITY_DURATION] * 2)
            data_ech[pa.NC_PA_MATURITY_DURATION] = [int(min(pa.MAX_MONTHS_ST, max(0, round(x)))) for x in data_ech[pa.NC_PA_MATURITY_DURATION]]
    
        return data_ech
    
    
    def join_fermat_margins(self, dic_evol, data_ech, data_nmd):
        new_margins = None
        new_sp_tx = None
        col_num = [x for x in pa.NC_PA_COL_SORTIE_NUM_PN if x != "M0"]
        if len(data_ech) > 0:
            data_ech["cle_ech"] = data_ech[pa.NC_PA_CLE]
            data_ech = data_ech.set_index(["cle_ech"]).sort_index()
            data_ech_marges = data_ech.loc[data_ech[pa.NC_PA_IND03] == pa.NC_PA_MG_CO, col_num].copy().fillna(0)
            data_ech_encours = data_ech.loc[data_ech[pa.NC_PA_IND03] == pa.NC_PA_DEM, col_num].copy().fillna(0)
            weighted_margins = pd.DataFrame(np.array(data_ech_marges) * np.array(data_ech_encours) \
                                            , index=data_ech_marges.index)
            weighted_margins = weighted_margins.groupby(weighted_margins.index).sum()
            somme_encours = (data_ech_encours).copy().groupby(data_ech_encours.index).sum()
            new_margins = pd.DataFrame((np.array(weighted_margins) / np.array(somme_encours)), \
                                       index=weighted_margins.index, columns=col_num)
            new_margins = new_margins.replace((np.inf, -np.inf), (0, 0)).fillna(0)
    
        if len(data_nmd) > 0:
            data_nmd["cle_nmd"] = data_nmd[pa.NC_PA_CLE]
            data_nmd = data_nmd.set_index(["cle_nmd"]).sort_index()
            new_sp_tx = data_nmd.loc[data_nmd[pa.NC_PA_IND03] == pa.NC_PA_TX_SP, col_num].astype(np.float64).fillna(0)
            if new_sp_tx.index.duplicated().any():
                data_encours = data_nmd.loc[data_nmd[pa.NC_PA_IND03] == pa.NC_PA_DEM_CIBLE, col_num].copy().fillna(0)
                weighted_margins = pd.DataFrame(np.array(new_sp_tx) * np.array(data_encours)
                                                , index=new_sp_tx.index)
                weighted_margins = weighted_margins.groupby(weighted_margins.index).sum()
                somme_encours = (data_encours).copy().groupby(data_encours.index).sum()
                new_sp_tx = pd.DataFrame((np.array(weighted_margins) / np.array(somme_encours)),
                                         index=weighted_margins.index, columns=col_num)
                new_sp_tx = new_sp_tx.replace((np.inf, -np.inf), (0, 0)).fillna(0)
    
        for typo, new_tx, name_ind, order_on in zip(["ECH", "NMD"], [new_margins, new_sp_tx],
                                                    [pa.NC_PA_MG_CO, pa.NC_PA_TX_SP], [pa.NC_PA_COL_SPEC_ECH, []]):
            if len(dic_evol[typo]) > 0 and new_tx is not None:
                DEM = pa.NC_PA_DEM_CIBLE
                data_new_tx = dic_evol[typo].copy()
                data_new_tx[col_num] = np.nan
                data_new_tx["M0"] = 0
                data_new_tx[pa.NC_PA_IND03] = name_ind
                data_new_tx.update(new_tx)
                data_new_tx[col_num] = data_new_tx[col_num].fillna(0)
                dic_evol[typo] = pd.concat([dic_evol[typo], data_new_tx])
                dic_evol[typo] = gu.order_by_indic(dic_evol[typo], [DEM, name_ind], pa.NC_PA_IND03, \
                                                   [pa.NC_PA_BILAN, pa.NC_PA_CLE, pa.NC_PA_INDEX] + order_on)
    
    
    def get_relevant_stock_data(self, data_stock):
        data_stock_evol = data_stock[data_stock[pa.NC_PA_IND03] == "LEF"].copy()
        somme_flux = (data_stock_evol[pa.NC_PA_COL_SORTIE_NUM_ST].values.sum(axis=1)) ** 2
        data_stock_evol = data_stock_evol[somme_flux != 0].copy()
        data_stock_evol["cle_stock"] = data_stock_evol[pa.NC_PA_CLE].copy()
        data_stock_evol = data_stock_evol.set_index("cle_stock").sort_index()
        data_stock_evol[pa.NC_PA_IND03] = pa.NC_PA_DEM_CIBLE
    
        dic_stock_evol = {}
        dic_stock_evol["NMD"] = data_stock_evol[data_stock_evol[pa.NC_PA_isECH] == "N"].copy()
        dic_stock_evol["ECH"] = data_stock_evol[
            (data_stock_evol[pa.NC_PA_isECH] == "O") & (~data_stock_evol[pa.NC_PA_CONTRACT_TYPE].str.startswith("P-PEL"))].copy()

        return dic_stock_evol
    
    
    def add_missing_indics(self, pn_data, type_pn):
        DEM = pa.NC_PA_DEM_CIBLE
        if type_pn == "ECH":
            indic_ordered = [DEM, pa.NC_PA_MG_CO, pa.NC_PA_TX_SP, pa.NC_PA_TX_CIBLE]
            keys_order = [pa.NC_PA_BILAN, pa.NC_PA_CLE, pa.NC_PA_INDEX] + pa.NC_PA_COL_SPEC_ECH
            empty_indics = [pa.NC_PA_MG_CO, pa.NC_PA_TX_SP, pa.NC_PA_TX_CIBLE]
        else:
            indic_ordered = [DEM, pa.NC_PA_TX_SP, pa.NC_PA_TX_CIBLE]
            keys_order = [pa.NC_PA_BILAN, pa.NC_PA_CLE, pa.NC_PA_INDEX]
            empty_indics = [pa.NC_PA_TX_SP, pa.NC_PA_TX_CIBLE]
    
        pn_data = gu.add_empty_indics(pn_data, empty_indics, pa.NC_PA_IND03, \
                                      DEM, pa.NC_PA_COL_SORTIE_NUM_PN, order=True, indics_ordered=indic_ordered, \
                                      keys_order=keys_order)
        return pn_data
    
    
    def calculate_PN_PRCT(self, data_stock, data_ech=[], data_nmd=[]):
        dic_stock_evol = self.get_relevant_stock_data(data_stock)
        num_cols = [x for x in pa.NC_PA_COL_SORTIE_NUM_ST if x != "M0"]
        if len(dic_stock_evol["ECH"]) > 0:
            dic_stock_evol["ECH"] = self.map_and_calculate_duree(dic_stock_evol["ECH"])
    
        if self.up.current_etab not in gp.NON_RZO_ETABS:
            self.join_fermat_margins(dic_stock_evol, data_ech.copy(), data_nmd.copy())
    
        for symb, type_pn in zip(["ECH%", "NMD%"], ["ECH", "NMD"]):
            if len(dic_stock_evol[type_pn]) > 0:
                DEM = pa.NC_PA_DEM_CIBLE
                if ((type_pn == "ECH" and pa.NC_PA_MG_CO in dic_stock_evol[type_pn][pa.NC_PA_IND03].unique()) \
                    or (type_pn == "NMD" and self.up.current_etab not in gp.NON_RZO_ETABS and pa.NC_PA_TX_SP in
                        dic_stock_evol[type_pn][pa.NC_PA_IND03].unique())) \
                        and rzo_p.do_pn:
                    INDEXO = np.array([symb + str(i) for i in range(1, int(dic_stock_evol[type_pn].shape[0] / 2) + 1)])
                    INDEXO = np.repeat(INDEXO, 2)
                    dic_stock_evol[type_pn][pa.NC_PA_INDEX] = INDEXO
                else:
                    dic_stock_evol[type_pn][pa.NC_PA_INDEX] = [symb + str(i) for i in
                                                            range(1, dic_stock_evol[type_pn].shape[0] + 1)]
    
                if type_pn == "ECH":
                    col_to_keep = pa.NC_PA_COL_SORTIE_QUAL_ECH + pa.NC_PA_COL_SORTIE_NUM_PN
                else:
                    col_to_keep = pa.NC_PA_COL_SORTIE_QUAL + pa.NC_PA_COL_SORTIE_NUM_PN
    
                dic_stock_evol[type_pn].loc[dic_stock_evol[type_pn][pa.NC_PA_IND03] == DEM, num_cols] = np.nan
    
                dic_stock_evol[type_pn] = dic_stock_evol[type_pn][col_to_keep]
    
                dic_stock_evol[type_pn] = self.add_missing_indics(dic_stock_evol[type_pn], type_pn)
    
        data_stock = data_stock.drop(columns=[pa.NC_PA_isECH], axis=1, errors='ignore')
    
        return [data_stock] + [dic_stock_evol[type_pn] for type_pn in ["ECH", "NMD"]]
