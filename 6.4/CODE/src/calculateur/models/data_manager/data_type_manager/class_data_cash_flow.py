import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import numexpr as ne
import logging

nan = np.nan
logger = logging.getLogger(__name__)

class Data_Cash_Flow():
    """
    Formate les données
    """
    def __init__(self, cls_format, cls_hz_params, cls_fields, data_cf_cpfl):
        self.cls_fields = cls_fields
        self.cls_hz_params = cls_hz_params
        self.data_cf_cpfl = data_cf_cpfl
        self.cls_format = cls_format

    def format_cash_flow_file(self):
        self.NC_CF_VALUE =  "new_value_eur".upper()
        self.NC_CF_DATE = "cf_date".upper()
        self.NC_CF_CONTRACT = "base_contract_ref".upper()
        self.NC_CF_ETAB = "etab".upper()

        self.NC_DATE_END_MONTH = "cf_date_end_month".upper()
        self.NC_JOUR_TOMBEE = "jour_tombee".upper()

        data_cf_cpfl = self.data_cf_cpfl.copy()
        if len(data_cf_cpfl) > 0:
            logging.debug('    Lecture du fichier CASH FLOW')
            data_cf_cpfl = self.cls_format.upper_columns_names(data_cf_cpfl)
            data_cf_cpfl = data_cf_cpfl.drop_duplicates([self.NC_CF_CONTRACT,self.NC_CF_ETAB, self.NC_CF_DATE, self.NC_CF_VALUE])
            data_cf_cpfl = data_cf_cpfl[[self.NC_CF_CONTRACT, self.NC_CF_DATE, self.NC_CF_VALUE]].copy()

            data_cf_cpfl, data_cf_cpfl_tombee = self.format_montants_RZO(data_cf_cpfl)
            self.data_cf_cpfl = data_cf_cpfl
            self.data_cf_cpfl_tombee = data_cf_cpfl_tombee

        else:
            self.num_cols = ["M" + str(i) for i in range(1, self.cls_hz_params.max_projection + 1)]
            self.data_cf_cpfl = pd.DataFrame([],
                                             columns=[self.NC_CF_CONTRACT, self.NC_CF_DATE, self.NC_CF_VALUE] + self.num_cols)
            self.data_cf_cpfl_tombee = pd.DataFrame([], columns=[self.NC_CF_CONTRACT, self.NC_CF_DATE,
                                                                 self.NC_CF_VALUE] + self.num_cols)

    def format_montants_RZO(self, data):
        data[self.NC_CF_VALUE] = data[self.NC_CF_VALUE].fillna(0)
        data[self.NC_CF_VALUE] = data[self.NC_CF_VALUE].astype(np.float64)
        qual_vars = [x for x in data.columns if x != self.NC_CF_VALUE and x != self.NC_CF_DATE]
        data[qual_vars] = data[qual_vars].fillna("-")
        data[self.NC_CF_DATE] = data[self.NC_CF_DATE].str.strip().str.replace(".","01/01/1990")
        data[self.NC_DATE_END_MONTH] = pd.to_datetime(data[self.NC_CF_DATE], format='%d/%m/%Y') + pd.offsets.MonthEnd(0)
        data = data[data[self.NC_DATE_END_MONTH] > self.cls_hz_params.dar_usr].copy()
        data_num = data.pivot_table(index=qual_vars, columns=[self.NC_DATE_END_MONTH], values=self.NC_CF_VALUE, aggfunc="sum",
                                fill_value=0)
        data_tombee = data.copy()
        data_tombee[self.NC_JOUR_TOMBEE] = pd.to_datetime(data[self.NC_CF_DATE], format='%d/%m/%Y').dt.day
        data_tombee = data_tombee.pivot_table(index=qual_vars, columns=[self.NC_DATE_END_MONTH], values=self.NC_JOUR_TOMBEE, aggfunc="max",
                                fill_value=np.nan)

        data_num = data_num.reset_index().copy()
        data_tombee = data_tombee.reset_index().copy()
        data_num = self.add_missing_num_cols(data_num, [x for x in data_num.columns if x != self.NC_CF_CONTRACT])
        data_tombee  = self.add_missing_num_cols(data_tombee,
                                                 [x for x in data_tombee.columns if x != self.NC_CF_CONTRACT], fill_value=np.nan)

        data_num = data_num.set_index(self.NC_CF_CONTRACT)
        data_tombee = data_tombee.set_index(self.NC_CF_CONTRACT)

        return data_num, data_tombee

    def add_missing_num_cols(self, data, num_cols, fill_value=0):
        necessary_cols = [pd.Timestamp((self.cls_hz_params.dar_usr + relativedelta(months=x)
                                        + relativedelta(day=31)).date()) for x in range(1, self.cls_hz_params.max_projection+ 1)]
        missing_cols = [x for x in necessary_cols if x not in num_cols]
        data_num =  pd.concat([data[num_cols].copy(), pd.DataFrame(fill_value, columns = missing_cols, index=data.index)], axis=1)
        data_num = data_num.reindex(sorted(data_num.columns), axis=1)
        self.num_cols = ["M" + str(i) for i in range(1, len(data_num.columns.tolist()) + 1)]
        data_num.columns = self.num_cols
        return pd.concat([data[[self.NC_CF_CONTRACT]].copy(), data_num], axis=1)

    ###################@profile
    def get_capital_with_cash_flows(self, data_ldp, cls_format, cls_proj):
        if len(self.data_cf_cpfl) > 0:
            data_cible = data_ldp.reset_index(drop=True).copy()
            if True:#cls_proj.name_product in ["a-security-tf", "a-security-tv", "p-security-tf", "p-security-tv"]:
                data_cible = data_cible[data_cible[self.cls_fields.NC_PROFIL_AMOR] == "CASHFLOWS"].copy()

            data_cible = data_cible[[self.cls_fields.NC_LDP_CONTRAT, self.cls_fields.NC_LDP_CONTRACT_TYPE,
                                     self.cls_fields.NC_PRODUCT_TYPE]].copy()

            data_cash_flows = (data_cible.join(self.data_cf_cpfl, on=self.cls_fields.NC_LDP_CONTRAT, how="inner"))

            if len(data_cash_flows) > 0:
                data_cash_flows = cls_format.format_value_passif(data_cash_flows, self.num_cols)

                data_cash_flows_tombee = (data_cible.join(self.data_cf_cpfl_tombee,
                                                          on=self.cls_fields.NC_LDP_CONTRAT, how="inner"))
                #n = data_cash_flows_tombee.shape[0]
                #is_cap_floor = (data_cash_flows_tombee[self.cls_fields.NC_LDP_CONTRACT_TYPE].isin(pp.Perimeters.perimetre_cap_floor_rco)).values.reshape(n, 1)
                #vals_cash_flow = data_cash_flows_tombee.loc[:, "M1":].values
                #data_cash_flows_tombee[self.num_cols] = ne.evaluate("where(is_cap_floor, nan, vals_cash_flow)").astype(np.float64)

                self.data_cash_flows = data_cash_flows.drop([self.cls_fields.NC_LDP_CONTRACT_TYPE,
                                                             self.cls_fields.NC_PRODUCT_TYPE],axis=1).copy()
                self.data_cash_flows_tombee = data_cash_flows_tombee.drop([self.cls_fields.NC_LDP_CONTRACT_TYPE,
                                                                           self.cls_fields.NC_PRODUCT_TYPE],axis=1).copy()
            else:
                self.data_cash_flows = []
                self.data_cash_flows_tombee = []

        else:
            self.data_cash_flows = []
            self.data_cash_flows_tombee = []

    def update_nominal_and_outstanding(self, data_ldp):
        if len(self.data_cf_cpfl) > 0:
            f_cf_prof  = np.array(data_ldp[self.cls_fields.NC_LDP_TYPE_AMOR] == "O")
            if f_cf_prof.any():
                dar_mois = self.cls_hz_params.dar_mois
                value_date_month = data_ldp.loc[f_cf_prof, self.cls_fields.NC_LDP_VALUE_DATE].values
                num_rule = (np.nan_to_num(np.array(data_ldp.loc[f_cf_prof, self.cls_fields.NC_LDP_RELEASING_RULE]), nan=-1).astype(int))
                is_release_rule = ne.evaluate("num_rule != -1")

                data_cash_flows_all = (data_ldp.loc[f_cf_prof, [self.cls_fields.NC_LDP_CONTRAT, self.cls_fields.NC_PRODUCT_TYPE]].copy()
                                       .reset_index(drop=True).join(self.data_cf_cpfl, on=self.cls_fields.NC_LDP_CONTRAT, how="left"))

                data_cash_flows_all = self.cls_format.format_value_passif(data_cash_flows_all, self.num_cols)

                new_outstanding = np.round(np.array(
                    data_cash_flows_all[[x for x in data_cash_flows_all.columns if x not in
                                         [self.cls_fields.NC_LDP_CONTRAT, self.cls_fields.NC_PRODUCT_TYPE]]]).sum(axis=1),2)
                not_null_cond = ~data_cash_flows_all["M1"].isnull().values
                old_outstanding = data_ldp.loc[f_cf_prof, self.cls_fields.NC_LDP_OUTSTANDING].values
                data_ldp.loc[f_cf_prof, self.cls_fields.NC_LDP_OUTSTANDING] = ne.evaluate("where(not_null_cond & (value_date_month - dar_mois <= 0),"
                                                                         "new_outstanding, old_outstanding)")


                t = self.data_cf_cpfl.shape[1]
                current_month = np.arange(1, t + 1).reshape(1, t)

                val_month = np.array(value_date_month).reshape(value_date_month.shape[0], 1)
                new_nominal = np.array(data_cash_flows_all[[x for x in data_cash_flows_all.columns if x not in
                                                            [self.cls_fields.NC_LDP_CONTRAT, self.cls_fields.NC_PRODUCT_TYPE]]])

                new_nominal = ne.evaluate("where(current_month <= val_month - dar_mois, 0, new_nominal)").sum(axis=1)
                old_nominal = data_ldp.loc[f_cf_prof, self.cls_fields.NC_LDP_NOMINAL].values
                new_nominal = ne.evaluate("where(not_null_cond & (value_date_month - dar_mois > 0) & (~is_release_rule), new_nominal,"
                                          "old_nominal)")
                # PRODUITS MARCHES
                #is_produit_marche = (data_ldp.loc[f_cf_prof, self.cls_fields.NC_LDP_CONTRACT_TYPE].isin(pp.Perimeters.produits_marche)).values
                #data_ldp.loc[f_cf_prof, self.cls_fields.NC_LDP_NOMINAL] = ne.evaluate("where(is_produit_marche, old_nominal, new_nominal)")

        return data_ldp


    def cancel_data_without_cash_flows(self, data_ldp):
        contract_with_cashflow = data_ldp[data_ldp[self.cls_fields.NC_PROFIL_AMOR]
                                                       == "CASHFLOWS"][[self.cls_fields.NC_LDP_CONTRAT]].copy()

        no_cf_found = contract_with_cashflow[
            ~contract_with_cashflow[self.cls_fields.NC_LDP_CONTRAT].isin(self.data_cf_cpfl.index.values.tolist())]
        if len(no_cf_found) > 0:
            logger.debug("      Certains contrats en 'O' n'ont pas de cashflow dans le fichier cashflow %s "
                         % no_cf_found[self.cls_fields.NC_LDP_CONTRAT].values.tolist())
            logger.debug("      Ils ne seront pas projetés")
            data_ldp.loc[
                data_ldp[self.cls_fields.NC_LDP_CONTRAT].isin(no_cf_found[self.cls_fields.NC_LDP_CONTRAT].values.tolist()), self.cls_fields.NC_PROFIL_AMOR] = "A"
            data_ldp.loc[
                data_ldp[self.cls_fields.NC_LDP_CONTRAT].isin(no_cf_found[self.cls_fields.NC_LDP_CONTRAT].values.tolist()), self.cls_fields.NC_LDP_OUTSTANDING] = 0
            data_ldp.loc[
                data_ldp[self.cls_fields.NC_LDP_CONTRAT].isin(no_cf_found[self.cls_fields.NC_LDP_CONTRAT].values.tolist()), self.cls_fields.NC_LDP_NOMINAL] = 0

        return data_ldp