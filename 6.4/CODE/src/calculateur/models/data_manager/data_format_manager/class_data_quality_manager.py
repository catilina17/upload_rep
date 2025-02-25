import numpy as np
import pandas as pd
import numexpr as ne
from calculateur.models.utils import utils as ut
from calculateur.models.mappings.products_perimeters import Perimeters

class Data_Quality_Manager():
    """
    Chargement des données, des pharamètres de modèle et formatage.
    Sous-calculator : Data_Loader, et Data_Formater
    """
    def __init__(self, cls_debloc_params, cls_data, cls_hz_params, cls_format):
        self.cls_debloc_params = cls_debloc_params
        self.cls_data = cls_data
        self.dar_usr = cls_hz_params.dar_usr
        self.cls_format = cls_format
        self.contracts_updated = []

        self.perimetre_PEL = Perimeters.perimetre_PEL
        self.perimetre_P_CAT = Perimeters.perimetre_P_CAT
        self.perimetre_cap_floor = Perimeters.perimetre_cap_floor_rco
        self.perimetre_A_INTBQ = Perimeters.perimetre_A_INTBQ
        self.perimetre_P_INTBQ = Perimeters.perimetre_P_INTBQ
        self.perimetre_ptz_rco = Perimeters.perimetre_ptz_rco
        self.perimetre_swap = Perimeters.perimetre_swap
        self.perimetre_titres = Perimeters.perimetre_titres
        self.default_date = cls_format.default_date

    def update_amor_if_ech_greater_than_ourstanding(self, data_hab):
        dar_mois = ut.nb_months_in_date(self.cls_data.dar_usr)
        value_date_month = data_hab[self.cls_data.NC_LDP_VALUE_DATE]
        cond = data_hab[self.cls_data.NC_LDP_ECHEANCE_VAL].fillna(0) > data_hab[self.cls_data.NC_LDP_OUTSTANDING].fillna(0)
        cond = cond & (value_date_month <= dar_mois)
        data_hab[self.cls_data.NC_LDP_TYPE_AMOR] = np.where(cond, "F", data_hab[self.cls_data.NC_LDP_TYPE_AMOR].values)
        return data_hab

    def update_mat_and_val_date_for_deblocage(self, data):
        # Mise à jour de la value date pour les contrats de déblocage avec outstanding
        # Conservation de l'ancienne value_date pour les calcul de durée d'amortissement et de durée de vie
        # Adaptation à RCO
        n = data.shape[0]
        dar_mois = self.cls_format.nb_months_in_date(self.dar_usr)
        num_rule = (np.nan_to_num(np.array(data[self.cls_data.NC_LDP_RELEASING_RULE]), nan=-1).astype(int)).reshape(n, 1)
        is_release_rule = (num_rule != -1).reshape(n, 1)
        is_forward = np.array(data[self.cls_data.NC_LDP_VALUE_DATE] > dar_mois).reshape(n, 1)
        no_capitalization = np.array(~(data[self.cls_data.NC_LDP_CAPI_MODE] == 'T')).reshape(n, 1)
        outstd_crd = np.array(data[self.cls_data.NC_LDP_OUTSTANDING]).reshape(n, 1)
        nominal = np.array(data[self.cls_data.NC_LDP_NOMINAL]).reshape(n, 1)

        # PRODUIT
        list_products = ["HB-NS-CR-PTZ", "A-PTZ+", "HB-NS-PR-PER", "HB-NS-CR-HBN", "HB-NS-CR-HAB", "HB-NS-CR-EQ", "HB-NS-CR-EQA"]
        list_products = (list_products + self.perimetre_PEL + self.perimetre_P_CAT + self.perimetre_A_INTBQ + self.perimetre_P_INTBQ
                         + self.perimetre_cap_floor + self.perimetre_swap + self.perimetre_titres)
        contract_type_cond = np.array(~data[self.cls_data.NC_LDP_CONTRACT_TYPE].isin(list_products)).reshape(n,1)

        cond_modif = is_forward & is_release_rule & no_capitalization & contract_type_cond

        data_num_rule = data.loc[cond_modif, [self.cls_data.NC_LDP_CLE, self.cls_data.NC_LDP_RELEASING_RULE, self.cls_data.NC_LDP_CONTRACT_TYPE,
                 self.cls_data.NC_LDP_MARCHE]].copy()

        self.cls_debloc_params.get_deblocage_mtx(data_num_rule)
        mtx_deblocage_all, cancel_rate = self.cls_debloc_params.mtx_deblocage_all, self.cls_debloc_params.cancel_rates_all

        p = data_num_rule.shape[0]

        if p > 0:
            _outstd_crd = outstd_crd[cond_modif].reshape(p, 1)
            _nominal = nominal[cond_modif].reshape(p, 1)
            _num_rule = num_rule[cond_modif].reshape(p, 1)
    
            mtx_deblocage_all1 = mtx_deblocage_all[:, 1:].copy()
            mtx_deblocage_all[:, 1:] = ne.evaluate("_outstd_crd + mtx_deblocage_all1 * _nominal - _nominal")
            new_value_date = ut.first_sup_strict_zero(mtx_deblocage_all, 1, 0, invalid_val=-1)
            if (new_value_date != -1).any():
                value_date = np.array(data.loc[cond_modif, self.cls_data.NC_LDP_VALUE_DATE])
                has_new_value = (new_value_date != -1)
                filtero = cond_modif.reshape(n)
    
                data.loc[filtero, self.cls_data.NC_LDP_VALUE_DATE] = \
                    np.where(has_new_value, (dar_mois + new_value_date).astype(int),
                             data.loc[filtero, self.cls_data.NC_LDP_VALUE_DATE].values)
    
                months_adj = dar_mois + new_value_date - value_date

                data.loc[filtero, self.cls_data.NC_LDP_MATUR_DATE] = \
                    np.where(has_new_value, (data.loc[filtero, self.cls_data.NC_LDP_MATUR_DATE] + months_adj).astype(int),
                             data.loc[filtero, self.cls_data.NC_LDP_MATUR_DATE].values)

                real_val_date = data.loc[filtero, self.cls_data.NC_LDP_VALUE_DATE + "_REAL"] + months_adj.astype("timedelta64[M]")
    
                real_mat_date = data.loc[filtero, self.cls_data.NC_LDP_MATUR_DATE + "_REAL"]+ months_adj.astype("timedelta64[M]")
    
                data.loc[filtero, self.cls_data.NC_LDP_VALUE_DATE + "_REAL"] = \
                    np.where(has_new_value, real_val_date,
                             data.loc[filtero, self.cls_data.NC_LDP_VALUE_DATE + "_REAL"].values)
    
                data.loc[filtero, self.cls_data.NC_LDP_MATUR_DATE + "_REAL"] = \
                    np.where(has_new_value, real_mat_date,
                             data.loc[filtero, self.cls_data.NC_LDP_MATUR_DATE + "_REAL"].values)
    
                self.contracts_updated = data.loc[filtero][has_new_value][[self.cls_data.NC_LDP_CLE]].copy()
                self.contracts_updated["MONTH_ADJ"] = months_adj[has_new_value]
                self.contracts_updated = self.contracts_updated.set_index(self.cls_data.NC_LDP_CLE)

        return data

    def update_mat_and_val_date_for_already_realeased_cap(self, data_hab):
        # Mise à jour de la value date pour les contrats forward dont le outstanding est égal au nominal
        # Conservation de l'ancienne value_date pour les calcul de durée d'amortissement et de durée de vie
        # Adaptation à RCO
        contracts_updated_old = self.contracts_updated
        n = data_hab.shape[0]
        dar_mois = self.cls_format.nb_months_in_date(self.dar_usr)
        num_rule = (np.nan_to_num(np.array(data_hab[self.cls_data.NC_LDP_RELEASING_RULE]), nan=-1).astype(int)).reshape(n, 1)
        not_release_rule = (num_rule == -1).reshape(n, 1)
        is_forward = np.array(data_hab[self.cls_data.NC_LDP_VALUE_DATE] > dar_mois).reshape(n, 1)
        no_capitalization = np.array(~(data_hab[self.cls_data.NC_LDP_CAPI_MODE] == 'T')).reshape(n, 1)
        outstd_crd = np.array(data_hab[self.cls_data.NC_LDP_OUTSTANDING]).reshape(n, 1)
        nominal = np.array(data_hab[self.cls_data.NC_LDP_NOMINAL]).reshape(n, 1)
        no_releasing_date = np.array((data_hab[self.cls_data.NC_LDP_RELEASING_DATE] == 0)).reshape(n, 1)
        outstanding_up_nom = (np.abs(outstd_crd) >= np.abs(nominal)) & (np.abs(nominal) > 0)

        # PRODUIT
        list_products = ["HB-NS-CR-PTZ", "HB-NS-PR-PER", "HB-NS-CR-HBN", "HB-NS-CR-HAB", "HB-NS-CR-EQ", "HB-NS-CR-EQA"]
        list_products = (list_products + self.perimetre_PEL + self.perimetre_P_CAT + self.perimetre_A_INTBQ
                         + self.perimetre_P_INTBQ + self.perimetre_cap_floor +  self.perimetre_swap + self.perimetre_titres)
        contract_type_cond = np.array(~data_hab[self.cls_data.NC_LDP_CONTRACT_TYPE].isin(list_products)).reshape(n,1)

        cond_modif = is_forward & not_release_rule & no_capitalization & outstanding_up_nom & no_releasing_date & contract_type_cond

        data_modif = data_hab.loc[cond_modif, [self.cls_data.NC_LDP_CLE]].copy()
        p = data_modif.shape[0]

        if p > 0:
            _outstd_crd = outstd_crd[cond_modif].reshape(p, 1)
            _nominal = nominal[cond_modif].reshape(p, 1)
            value_date = np.array(data_hab.loc[cond_modif, self.cls_data.NC_LDP_VALUE_DATE])
            cond_modif = cond_modif.reshape(n)

            data_hab.loc[cond_modif, self.cls_data.NC_LDP_VALUE_DATE] = dar_mois

            months_adj = dar_mois - value_date

            data_hab.loc[cond_modif, self.cls_data.NC_LDP_MATUR_DATE] = data_hab.loc[cond_modif, self.cls_data.NC_LDP_MATUR_DATE] + months_adj
            data_hab.loc[cond_modif, self.cls_data.NC_LDP_FIRST_AMORT_DATE] = data_hab.loc[
                                                                       cond_modif, self.cls_data.NC_LDP_FIRST_AMORT_DATE] + months_adj
            data_hab.loc[cond_modif, self.cls_data.NC_LDP_RELEASING_DATE] = data_hab.loc[
                                                                     cond_modif, self.cls_data.NC_LDP_RELEASING_DATE] + months_adj

            real_val_date = data_hab.loc[cond_modif, self.cls_data.NC_LDP_VALUE_DATE + "_REAL"] + months_adj.astype("timedelta64[M]")

            real_mat_date = data_hab.loc[cond_modif, self.cls_data.NC_LDP_MATUR_DATE + "_REAL"] + months_adj.astype("timedelta64[M]")

            real_amor_date = data_hab.loc[cond_modif, self.cls_data.NC_LDP_FIRST_AMORT_DATE + "_REAL"] + months_adj.astype("timedelta64[M]")

            real_releasing_date = data_hab.loc[cond_modif, self.cls_data.NC_LDP_RELEASING_DATE + "_REAL"] + months_adj.astype("timedelta64[M]")

            data_hab.loc[cond_modif, self.cls_data.NC_LDP_VALUE_DATE + "_REAL"] = real_val_date

            data_hab.loc[cond_modif, self.cls_data.NC_LDP_MATUR_DATE + "_REAL"] = real_mat_date

            data_hab.loc[cond_modif, self.cls_data.NC_LDP_FIRST_AMORT_DATE + "_REAL"] = real_amor_date

            data_hab.loc[cond_modif, self.cls_data.NC_LDP_RELEASING_DATE + "_REAL"] = real_releasing_date

            contracts_updated = data_hab.loc[cond_modif][[self.cls_data.NC_LDP_CLE]].copy()
            contracts_updated["MONTH_ADJ"] = months_adj
            contracts_updated = contracts_updated.set_index(self.cls_data.NC_LDP_CLE)
            if len(contracts_updated_old) != 0:
                contracts_updated = pd.concat([contracts_updated, contracts_updated_old])

        else:
            contracts_updated = contracts_updated_old

        self.contracts_updated  = contracts_updated
        return data_hab

    def update_capitalization_status_for_contracts_with_releasing_dates(self, data_hab):
        """
        Only for PTZ deals
        """
        n = data_hab.shape[0]
        dar_mois = self.cls_format.nb_months_in_date(self.dar_usr)
        num_rule = (np.nan_to_num(np.array(data_hab[self.cls_data.NC_LDP_RELEASING_RULE]), nan=-1).astype(int)).reshape(n, 1)
        #not_release_rule = (num_rule == -1).reshape(n, 1)
        is_forward = np.array(data_hab[self.cls_data.NC_LDP_VALUE_DATE] > dar_mois).reshape(n, 1)
        no_capitalization = np.array(~(data_hab[self.cls_data.NC_LDP_CAPI_MODE] == 'T')).reshape(n, 1)
        outstd_crd = np.array(data_hab[self.cls_data.NC_LDP_OUTSTANDING]).reshape(n, 1)
        nominal = np.array(data_hab[self.cls_data.NC_LDP_NOMINAL]).reshape(n, 1)
        with_releasing_date = np.array((data_hab[self.cls_data.NC_LDP_RELEASING_DATE + "_REAL"] != self.default_date)).reshape(n, 1)
        #past_releasing_date = np.array((data_hab[self.cls_data.NC_LDP_RELEASING_DATE] < dar_mois)).reshape(n, 1)
        #outstanding_up_nom = outstd_crd >= nominal
        contract_type_cond = np.array(~data_hab[self.cls_data.NC_LDP_CONTRACT_TYPE].isin(["HB-NS-CR-PTZ"])).reshape(n, 1)
        contract_type_cond = contract_type_cond & np.array(data_hab[self.cls_data.NC_LDP_CONTRACT_TYPE].isin(self.perimetre_ptz_rco)).reshape(n, 1)

        cond_modif = is_forward & no_capitalization & with_releasing_date & contract_type_cond

        data_hab.loc[cond_modif.reshape(n), [self.cls_data.NC_LDP_CAPI_MODE]] = "T"
        return data_hab
