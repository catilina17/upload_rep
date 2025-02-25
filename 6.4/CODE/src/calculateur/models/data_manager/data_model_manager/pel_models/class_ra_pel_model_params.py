from ....data_manager.data_model_manager.ech_models.class_ra_rn_model_params import Data_RA_RN_Model_Params
from utils import excel_utils as ex, general_utils as gu
import pandas as pd
import dateutil

class Data_RA_PEL_Model_Params(Data_RA_RN_Model_Params):
    def __init__(self, model_wb, nb_mois_proj):
        Data_RA_RN_Model_Params.__init__(self, model_wb, nb_mois_proj)

    def load_pel_model_params(self):
        self.mtx_tx_struct_dict = {}

        self.NAME_RANGE_CONTRACTS_MAPPINGS = ["_CONTRAT_MTX_TX_STRUCT_PEL_CAT", "_CONTRAT_MTX_TX_STRUCT_PEL"]
        self.NAME_RANGE_LIST_CONTRACTS = "_LIST_CONTRATS"

        self.NAME_RANGE_TX_STRUCT = ["MTX_TX_STRUCT_PEL_4A", "MTX_TX_STRUCT_PEL_4AP", "MTX_TX_STRUCT_PELC_4A",
                                     "MTX_TX_STRUCT_PELC_4A10A", "MTX_TX_STRUCT_PELC_10A"]

        self.NC_CLE_MTS = "CLE"
        self.NC_AGE_MTS = "AGE"
        self.NC_LN_TX_STRUCT_MTS = "TAUX_STRUCTUREL"
        self.NC_LN_TX_STRUCT_APS_MTS = "TAUX_STRUCTUREL_APRES_STRIKE"
        self.NC_LN_TX_STRUCT_AVS_MTS = "TAUX_STRUCTUREL_AVT_STRIKE"
        self.NC_LN_TX_STRUCT_SM2_MTS = "TAUX_STRUCTUREL_STRIKE_MOINS_2PTS"
        self.NC_LN_TX_STRUCT_SM1_MTS = "TAUX_STRUCTUREL_STRIKE_MOINS_1PT"
        self.NC_BETA = "BETA"
        self.NC_U = "U"
        self.NC_MU = "MU"
        self.NC_AGE = "AGE"
        self.NC_CONTRAT = "CONTRAT"

        self.NC_CONTRAT_CC = "CONTRAT"
        self.NC_CLE_CC = "CLE"
        self.NC_CONTRAT_PEL_CC = "CONTRAT PEL"

        """ PARAMS FIXES"""
        self.tx_cot_soc = ex.get_value_from_named_ranged(self.model_wb, "_TX_COT_SOC")
        self.tx_fisc = ex.get_value_from_named_ranged(self.model_wb, "_TX_FISC")
        self.tx_survie = ex.get_value_from_named_ranged(self.model_wb, "_TX_SURVIE")

        self.curve_name_arbi = ex.get_value_from_named_ranged(self.model_wb, "_INDEX_ARBITRAGE_PEL")
        self.tenor_arbi = ex.get_value_from_named_ranged(self.model_wb, "TENOR_ARBITRAGE_PEL")

        self.date_pfu = ex.get_value_from_named_ranged(self.model_wb, "_DATE_PFU")
        self.date_pfu = dateutil.parser.parse(str(self.date_pfu)).replace(tzinfo=None)
        self.date_pfu = (self.date_pfu.year * 12 + self.date_pfu.month)
        self.limit_age_pfu = 144

        self.mu_default = ex.get_value_from_named_ranged(self.model_wb, "_mu_default")
        self.u_default = ex.get_value_from_named_ranged(self.model_wb, "_u_default")
        self.beta_default = ex.get_value_from_named_ranged(self.model_wb, "_beta_default")

        self.strike_1pt = 0.1
        self.strike_2pt = 0.2

        """ PEL% ECOULEMENT"""
        self.nb_contrat_total = 100000000 / 5000
        self.nominal_total = 100000000
        self.EM_moyen_t0 = 5000
        self.IC_moyen_t0 = 0

        self.age_limite_pel_sup = 15 * 12
        self.annee_limit = 2011

        """ LISTE CONTRATS"""
        self.get_contracts_pel_list(self.model_wb)

        """ MATRICES CONTRATS"""
        self.mapping_contrats_cle = self.get_mapping_contracts(self.model_wb)

        """ MATRICES VERSEMENTS et TX STRUCTURELS """
        self.mtx_tx_struct = self.get_matrices(self.mtx_tx_struct_dict, self.model_wb)

        """ PEL ANCIEN """
        self.PEL_ANCIEN = ['P-PEL-C-5,25', 'P-PEL-C-4,25', 'P-PEL-C-4', 'P-PEL-C-3,6', 'P-PEL-C-4,5', 'P-PEL-C-3,5',
                      'P-PEL-ANCIEN', 'P-PEL-6', 'P-PEL-5,25', 'P-PEL-4,25', 'P-PEL-4', 'P-PEL-3,60', 'P-PEL-3,6',
                      'P-PEL-4,50', 'P-PEL-4,5', 'P-PEL-3,50', 'P-PEL-3,5']

        """ NB PN """
        self.max_pn = 60


    def get_contracts_pel_list(self, model_wb):
        global liste_contrats
        liste_contrats = ex.get_dataframe_from_range(model_wb, self.NAME_RANGE_LIST_CONTRACTS)


    def get_mapping_contracts(self, model_wb):
        names_range = self.NAME_RANGE_CONTRACTS_MAPPINGS
        mapp_contrats_cle = None
        for i in range(0, len(names_range)):
            df = ex.get_dataframe_from_range(model_wb, names_range[i])
            if not mapp_contrats_cle is None:
                mapp_contrats_cle = pd.concat([mapp_contrats_cle, df.copy()])
            else:
                mapp_contrats_cle = df.copy()

        mapp_contrats_cle[self.NC_CLE_CC] = mapp_contrats_cle[self.NC_CLE_CC].mask(mapp_contrats_cle[self.NC_CLE_CC].isnull(),
                                                mapp_contrats_cle[self.NC_CONTRAT_PEL_CC])

        mapp_contrats_cle = gu.strip_and_upper(mapp_contrats_cle,
                                            [self.NC_CONTRAT_CC, self.NC_CONTRAT_PEL_CC, self.NC_CLE_CC])
        mapp_contrats_cle = mapp_contrats_cle.set_index([self.NC_CONTRAT_CC, self.NC_CONTRAT_PEL_CC])

        return mapp_contrats_cle


    def get_matrices(self, mtx_tx_struct_dict, model_wb):
        names_mtx = self.NAME_RANGE_TX_STRUCT
        names_range = ["_MTX_TX_STRUCT_4ANS_PEL", "_MTX_TX_STRUCT_4ANS_PLUS_PEL", \
                   "_MTX_TX_STRUCT_4ANS_PEL_CAT", "_MTX_TX_STRUCT_4_10ANS_PEL_CAT", "_MTX_TX_STRUCT_10ANS_PEL_CAT"]

        for i in range(0, len(names_mtx)):
            mtx_tx_struct_dict[names_mtx[i]] = ex.get_dataframe_from_range(model_wb, names_range[i])

        mtx_tx_struct_temp = None

        for key, mtx in mtx_tx_struct_dict.items():
            mtx[self.NC_CONTRAT] = "P-PEL-C" if "PELC" in key else "P-PEL"
            if "_4AP" in key or "_10A" in key:
                mtx[self.NC_U] = mtx[self.NC_U].mask(mtx[self.NC_U].isnull(), self.u_default)
                mtx[self.NC_MU] = mtx[self.NC_MU].mask(mtx[self.NC_MU].isnull(), self.mu_default)
                mtx[self.NC_BETA] = mtx[self.NC_BETA].mask(mtx[self.NC_BETA].isnull(), self.beta_default)

            if mtx_tx_struct_temp is None:
                mtx_tx_struct_temp = mtx
            else:
                mtx_tx_struct_temp = pd.concat([mtx_tx_struct_temp, mtx])

        mtx_tx_struct_temp[self.NC_CLE_MTS] = mtx_tx_struct_temp[self.NC_CLE_MTS].fillna("*")
        mtx_tx_struct_temp[self.NC_AGE] = [str(int(x)) for x in mtx_tx_struct_temp[self.NC_AGE]]
        cle_mtx = [self.NC_CONTRAT, self.NC_AGE_MTS, self.NC_CLE_MTS]
        mtx_tx_struct_temp = gu.strip_and_upper(mtx_tx_struct_temp, cle_mtx)
        mtx_tx_struct_temp['new_key'] = mtx_tx_struct_temp[cle_mtx].apply(lambda row: '_'.join(row.values.astype(str)),
                                                                          axis=1)
        mtx_tx_struct_temp = mtx_tx_struct_temp.set_index('new_key').drop(columns=cle_mtx, axis=1)

        return mtx_tx_struct_temp
