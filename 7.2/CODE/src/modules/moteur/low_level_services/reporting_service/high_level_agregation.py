import numpy as np
from modules.moteur.low_level_services.reporting_service import mapping_module as map
import utils.general_utils as ut
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import logging
from mappings import general_mappings as gmp

logger = logging.getLogger(__name__)

class Agregate_to_PASSALM_Level():

    def __init__(self, key_vars_dic, etab, horizon, product, results_ag):
        self.INFO_DATA_STOCK = [pa.NC_PA_LCR_TIERS_SHARE, pa.NC_PA_LCR_TIERS, pa.NC_PA_INDEX,\
                           pa.NC_PA_SCOPE, pa.NC_PA_BOOK, pa.NC_PA_CUST, pa.NC_PA_PERIMETRE, pa.NC_PA_TOP_MNI, pa.NC_PA_DIM2, pa.NC_PA_DIM3,
                           pa.NC_PA_DIM4, pa.NC_PA_DIM5, pa.NC_PA_INDEX_AGREG, pa.NC_PA_POSTE, pa.NC_PA_Affectation_Social,
                           pa.NC_PA_Affectation_Social_2, pa.NC_PA_DIM_NSFR_1, \
                           pa.NC_PA_DIM_NSFR_2, pa.NC_PA_Regroupement_1, pa.NC_PA_Regroupement_2, \
                           pa.NC_PA_Regroupement_3, pa.NC_PA_Bilan_Cash, pa.NC_PA_Bilan_Cash_Detail, \
                           pa.NC_PA_Bilan_Cash_CTA, pa.NC_PA_METIER, pa.NC_PA_SOUS_METIER, pa.NC_PA_ZONE_GEO, \
                           pa.NC_PA_SOUS_ZONE_GEO]

        self.cle_join_stock = [pa.NC_PA_BASSIN, pa.NC_PA_ETAB, pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_MATUR, pa.NC_PA_DEVISE,
                          pa.NC_PA_RATE_CODE, pa.NC_PA_MARCHE, pa.NC_PA_GESTION, pa.NC_PA_PALIER]

        self.key_vars_dic = key_vars_dic
        self.etab = etab
        self.horizon = horizon
        self.product = product

        self.cols_month = [x for x in ["M" + str(i) for i in range(0, self.horizon + 1)] if x in results_ag]


    def generate_agregated_data(self, results_ag, data_stock):
        stock_data, is_not_det, is_not_det_gestion = self.load_stock_data(data_stock)
        results_ag = self.filter_dim_and_market_ag_data(results_ag)
        results_ag = self.finalize_agregation(results_ag, is_not_det, is_not_det_gestion)
        results_ag = self.mapp_results(results_ag)
        results = self.join_with_stock_data(results_ag, stock_data)
        results_f = self.finalize_formating(results)
        results_ag = []
        return results_f

    def filter_dim_and_market_ag_data(self, data):
        if pa.NC_PA_DIM6 in data.columns:
            filter_rco = (data[pa.NC_PA_DIM6].isnull()) | (data[pa.NC_PA_DIM6].str.upper() != "FCT")
            filter_rco = filter_rco & (data[self.key_vars_dic["MARCHE"]].str.upper() != "MDC")
            data = data[filter_rco].copy()
        return data

    def finalize_agregation(self, results_ag, is_not_det, is_not_det_gestion):
        results_ag = map.format_paliers(results_ag, self.key_vars_dic["PALIER"])
        results_ag[self.key_vars_dic["MARCHE"]] = results_ag[self.key_vars_dic["MARCHE"]].fillna("VIDE")
        if is_not_det_gestion:
            results_ag[self.key_vars_dic["GESTION"]] = "-"

        if is_not_det:
            results_ag[self.key_vars_dic["ETAB"]] = self.etab

        ag_vars = list(self.key_vars_dic.values()) + [pa.NC_PA_IND03]
        results_ag = results_ag[ag_vars + self.cols_month].copy().groupby(ag_vars, dropna=False, as_index=False).sum()
        results_ag = results_ag.sort_values(list(self.key_vars_dic.values()) + [pa.NC_PA_IND03])
        results_ag = results_ag.rename(columns={self.key_vars_dic["ETAB"]: pa.NC_PA_ETAB})
        results_ag[pa.NC_PA_BASSIN] = self.etab
        return results_ag


    def join_with_stock_data(self, results, stock_data):
        stock_data = self.filter_stock_data(stock_data, results[pa.NC_PA_CONTRACT_TYPE].unique().tolist())
        results["TEMP_KEY"] = results[self.cle_join_stock].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        results["TEMP_KEY"] = results["TEMP_KEY"].str.upper()
        stock_data.index = stock_data.index.str.upper()
        results_joined = results.join(stock_data, on=["TEMP_KEY"], rsuffix="_right")

        self.signal_not_found_stock_products(results_joined, results)

        lcr_share = (results_joined[pa.NC_PA_LCR_TIERS_SHARE].astype(np.float64).fillna(100).values). \
                        reshape(results_joined.shape[0], 1) / 100
        results_joined[self.cols_month] = results_joined[self.cols_month].values * lcr_share
        results_joined[pa.NC_PA_CLE] = results_joined[pa.NC_PA_CLE_OUTPUT].apply(lambda row: '_'.join(row.values.astype(str)),
                                                                 axis=1)
        return results_joined

    def signal_not_found_stock_products(self, results_joined, results):
        not_found = results_joined[pa.NC_PA_LCR_TIERS_SHARE].isnull()
        if not_found.any():
            list = results_joined.loc[not_found, "TEMP_KEY"].unique().tolist()
            msg = "Il y a des produits dans les fichiers contrats qu'on ne retrouve pas dans les contrats de l'ALIM : %s" % list
            logger.warning(msg)
        if len(results_joined) > len(results):
            msg = "Il y a des produits dupliqués dans les contrats de l'ALIM"
            logger.error(msg)
            raise ValueError(msg)

    def finalize_formating(self, results):
        results = results.sort_values([pa.NC_PA_CLE, pa.NC_PA_IND03]).copy()
        results = results[pa.NC_PA_COL_SORTIE_QUAL + self.cols_month].copy()
        results = results.groupby(pa.NC_PA_COL_SORTIE_QUAL, dropna=False, as_index=False).sum()
        results[pa.NC_PA_IND03] = results[pa.NC_PA_IND03].str[2:]
        results["PRODUIT"] = self.product.upper()
        return results

    def mapp_results(self, results):
        if self.product not in ["a-swap-tf", "a-swap-tv", "p-swap-tf", "p-swap-tv", "cap_floor",
                                "a-change-tf", "a-change-tv", "p-change-tf", "p-change-tv"] :
            contrat = results[self.key_vars_dic["CONTRACT_TYPE"]]
            cases = [contrat.str[:2] == "A-", contrat.str[:2] == "P-",
                     (contrat.str[:5] == "HB-NS") | ((contrat.str[:2] == "HB") & (contrat.str[-2:] == "-A")),
                     ((contrat.str[:2] == "HB") & (contrat.str[-2:] == "-P"))]
            results[pa.NC_PA_BILAN] = np.select(cases, ["B ACTIF", "B PASSIF", "HB ACTIF", "HB PASSIF"])

        elif self.product in ["a-swap-tf", "a-swap-tv", "p-swap-tf", "p-swap-tv", "cap_floor",
                              "a-change-tf", "a-change-tv", "p-change-tf", "p-change-tv"]:
            results[pa.NC_PA_BILAN] = np.where(results[self.key_vars_dic["BUY_SELL"]] == "S", "HB PASSIF", "HB ACTIF")

        if self.product in ["a-change-tf", "a-change-tv", "p-change-tf", "p-change-tv"]:
            results[self.key_vars_dic["RATE CODE"]] = results[self.key_vars_dic["RATE CODE"]].fillna("AUTRES").replace("", "AUTRES")
        else:
            results[self.key_vars_dic["RATE CODE"]] = results[self.key_vars_dic["RATE CODE"]].fillna("FIXE").astype(str).replace("", "FIXE")
        results[self.key_vars_dic["MATUR"]] = results[self.key_vars_dic["MATUR"]].fillna("-").replace("", "-")
        results[self.key_vars_dic["PALIER"]] = results[self.key_vars_dic["PALIER"]].fillna("-").replace("", "-")
        results = ut.force_integer_to_string(results, self.key_vars_dic["PALIER"])
        results = results.rename(columns={self.key_vars_dic["MARCHE"]: pa.NC_PA_MARCHE,
                                          self.key_vars_dic["MATUR"]: self.key_vars_dic["MATUR"] + "_TEMP",
                                          self.key_vars_dic["GESTION"]: self.key_vars_dic["GESTION"] + "_TEMP",
                                          self.key_vars_dic["PALIER"]: self.key_vars_dic["PALIER"] + "_TEMP",
                                          self.key_vars_dic["RATE CODE"]:pa.NC_PA_RATE_CODE,
                                          self.key_vars_dic["CUR"]: pa.NC_PA_DEVISE})

        results = map.map_data_with_general_mappings(results, gmp.map_pass_alm, self.key_vars_dic)

        ag_vars = [x for x in results.columns.tolist() if x not in self.cols_month]
        results = results[ag_vars + self.cols_month].copy().groupby(ag_vars, dropna=False, as_index=False).sum()

        return results


    def load_stock_data(self, stock_data):
        is_not_det = len(stock_data[pa.NC_PA_ETAB].unique()) == 1
        is_not_det_gestion = (len(stock_data[pa.NC_PA_GESTION].unique()) == 1) & (
                stock_data[pa.NC_PA_GESTION].unique().tolist() == ["-"])
        return stock_data, is_not_det, is_not_det_gestion


    def filter_stock_data(self, stock_data, contrats):
        stock_data = stock_data[stock_data[pa.NC_PA_CONTRACT_TYPE].isin(contrats)].copy()
        stock_data = stock_data[stock_data[pa.NC_PA_IND03] == pa.NC_PA_LEM]
        stock_data["TEMP_KEY"] = stock_data[self.cle_join_stock].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        stock_data = stock_data.set_index("TEMP_KEY")[self.INFO_DATA_STOCK].copy()
        if len(stock_data) == 0:
            msg = "Les contrats suivants ne se retrovuent pas les contrats de l'ALIM : %s" % contrats
            logger.error(msg)
            raise ValueError(msg)
        return stock_data
