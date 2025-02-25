import pandas as pd
from utils import excel_utils as ex
import numpy as np
from calculateur.models.utils import utils as ut
from utils import general_utils as gu

class Data_Deblocage_Model_Params():
    def __init__(self, model_wb, cls_fields, name_product):
        self.name_product = name_product
        self.model_wb = model_wb
        self.cls_fields = cls_fields
        self.err_msg_absent_contract = "Il y a des contrats sans modèles de déblocage mais qui ont une releasing rule non nulle"

    def load_deblocage_params(self):
        self.NR_MD_DEBLOCAGE = "_REGLES_DEBLOCAGE"
        self.NC_CANCEL_RATE = 'Cancellation rate'
        self.NC_DEBLOCAGE_RULE = "Déblocage"
        self.NC_MOD_CTRT = "CONTRACT_TYPE"
        self.NC_MOD_MARCHE = "MARCHE"
        if self.name_product not in ["nmd_st", "nmd_pn"]:
            self.get_deblocage_mtx_per_prod()
        else:
            self.nb_mois_deblocage = 0

    def get_deblocage_mtx_per_prod(self):
        mtx_deblocages = {}
        """ MATRICES DE DEBLOCAGE """
        deblocage_rules = ex.get_dataframe_from_range(self.model_wb, self.NR_MD_DEBLOCAGE)
        deblocage_rules = ut.explode_dataframe(deblocage_rules , [self.NC_MOD_CTRT, self.NC_MOD_MARCHE])
        deblocage_rules = deblocage_rules.reset_index(drop=True)
        mtx_deblocage = np.array(deblocage_rules.loc[:,'M1':].fillna(0)).cumsum(axis=1)
        mtx_deblocage = ut.remove_leading_zeros(mtx_deblocage)
        s = mtx_deblocage.shape[0]
        self.nb_mois_deblocage = mtx_deblocage.shape[1]
        mtx_deblocage = np.hstack([np.zeros((s, 1)), mtx_deblocage])
        mtx_deblocage = pd.DataFrame(mtx_deblocage, columns=["M" + str(i) for i in range(0, mtx_deblocage.shape[1])])
        self.mtx_deblocages = pd.concat([deblocage_rules.loc[:, :"M1"].iloc[:, :-1], mtx_deblocage], axis=1)

        cle = [self.NC_MOD_CTRT, self.NC_MOD_MARCHE , self.NC_DEBLOCAGE_RULE]
        self.mtx_deblocages[self.NC_DEBLOCAGE_RULE] = self.mtx_deblocages[self.NC_DEBLOCAGE_RULE].astype(int).astype(str)
        self.mtx_deblocages['new_key'] = self.mtx_deblocages[cle].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        self.mtx_deblocages = self.mtx_deblocages.drop_duplicates(subset=cle).set_index('new_key').drop(columns=cle, axis=1).copy()


    def get_deblocage_mtx(self, data_num_rule):
        _n = data_num_rule.shape[0]
        self.mtx_deblocage_all = np.zeros((_n, self.nb_mois_deblocage + 1))
        self.cancel_rates_all = np.zeros((_n, 1))

        cles_a_combiner = [self.cls_fields.NC_LDP_CONTRACT_TYPE, self.cls_fields.NC_LDP_MARCHE,
                           self.cls_fields.NC_LDP_RELEASING_RULE]

        data_num_rule2 = data_num_rule[cles_a_combiner].copy()
        data_num_rule2[self.cls_fields.NC_LDP_RELEASING_RULE] = data_num_rule2[self.cls_fields.NC_LDP_RELEASING_RULE].astype(int).astype(str)

        mtx_deblocage = gu.map_with_combined_key(data_num_rule2, self.mtx_deblocages, cles_a_combiner,
                                                      symbol_any="*", no_map_value=np.nan, filter_comb=True,
                                                      necessary_cols=1, error=True, error_message=self.err_msg_absent_contract)
        mtx_deblocage = mtx_deblocage.drop(cles_a_combiner,axis=1)
        self.mtx_deblocage_all = np.array(mtx_deblocage.drop([self.NC_CANCEL_RATE], axis=1))
        self.cancel_rates_all = np.array(mtx_deblocage[self.NC_CANCEL_RATE]).reshape(_n, 1)




