from params.sources_params import SourcesParams
from calculateur.services.projection_services.ech import run_calculator_stock_ech as cs
from calculateur.services.projection_services.nmd import run_calculator_stock_nmd as st_nmd
from calculateur.services.projection_services.nmd import run_nmd_spreads as nmd_sp
from modules.moteur.low_level_services.reporting_service import high_level_agregation as hla
from utils import excel_openpyxl as ex
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Stock_Calculator():

    def __init__(self, cls_calc_params, etab, scenario_name, cls_usr):
        self.sources_cls = SourcesParams()
        self.params = cls_calc_params
        self.calculated_stock = []
        self.data_stock_nmd_dt = []
        self.etab = etab
        self.scenario_name = scenario_name
        self.up = cls_usr

    def run_stock_calculator(self, data_stock, cls_pn_loader):
        scenarii = self.get_scenarios(self.etab, self.scenario_name, self.params.out_of_recalc_st_etabs)
        if len(scenarii) ==0 :
            return

        type_products = scenarii["TYPE PRODUIT"].values.tolist()
        list_products = self.get_models_list_from_products(type_products)

        compiled_indics = None
        key_vars_dic = {}
        calculated_stock = []

        for produit, type_data in list_products:
            nom_produit = self.params.products_map_nomenclature[produit]
            logger.info("          RECALCUL des %s"
                        % (nom_produit + " " + " ".join(type_data.split("_")).upper()))
            source_data = {}
            source_data["MODELS"] = {}
            if type_data == "stock_ech":
                source_data["STOCK"] = self.sources_cls.get_contract_sources_paths(self.etab, nom_produit)
                source_data = self.add_source_models(self.scenario_name, "ECH", source_data)

                cls_ag_st \
                    = cs.run_calculator_stock_ech(self.params.dar, self.params.horizon, source_data,
                                                  produit, tx_params=self.params.tx_params.copy(),
                                                  exit_indicators_type=["GPLIQ", "GPTX", "TCI"],
                                                  batch_size = self.params.batch_size_ech, output_mode="dataframe")

                compiled_indics = cls_ag_st.compiled_indics
                key_vars_dic = cls_ag_st.keep_vars_dic

            elif produit == "nmd_st":
                source_data["PN"] = self.params.get_source_data_pn(cls_pn_loader, "nmd")
                source_data["STOCK"] = self.sources_cls.get_contract_sources_paths(self.etab, nom_produit)
                source_data = self.add_source_models(self.scenario_name, "NMD", source_data)
                if self.up.nom_etab == "ONEY":
                    source_data["STOCK"]["LDP"]["CHEMIN"] = self.up.stock_nmd_template_file_path
                    source_data["STOCK"]["LDP"]["DELIMITER"] = ";"
                    source_data["STOCK"]["LDP"]["DECIMAL"] = ","

                cls_nmd_tmp = self.params.cls_nmd_tmp

                cls_nmd_spreads = nmd_sp.run_nmd_spreads(self.etab, self.params.horizon, source_data, cls_nmd_tmp)

                cls_ag_st  = st_nmd.run_calculator_nmd_stock(self.params.dar, self.params.horizon, source_data,
                                                             produit, cls_nmd_spreads=cls_nmd_spreads,
                                                             tx_params=self.params.tx_params.copy(),
                                                             exit_indicators_type=["GPLIQ", "GPTX", "TCI"],
                                                             agregation_level="NMD_TEMPLATE",
                                                             with_dyn_data=True, with_pn_data=True,
                                                             batch_size=self.params.batch_size_nmd,
                                                             tci_contract_perimeter=self.params.nmd_tci_perimeter.copy(),
                                                             output_mode="dataframe")

                compiled_indics = cls_ag_st.compiled_indics
                self.data_stock_nmd_dt = cls_ag_st.compiled_indics
                key_vars_dic = cls_ag_st.keep_vars_dic

                ex.close_workbook(source_data["MODELS"]["NMD"]["DATA"])

            if not compiled_indics is None and len(compiled_indics) >0 :
                cls_agg = hla.Agregate_to_PASSALM_Level(key_vars_dic, self.etab, self.params.horizon, produit, compiled_indics)
                calc_st_tmp = cls_agg.generate_agregated_data(compiled_indics, data_stock)
                if len(calculated_stock) > 0:
                    calculated_stock = pd.concat([calculated_stock, calc_st_tmp])
                else:
                    calculated_stock = calc_st_tmp.copy()
                compiled_indics = None

        self.calculated_stock = calculated_stock


    def add_source_models(self, scenario_name, type, source_data):
        source_model = self.params.get_source_models(scenario_name, type)
        source_data["MODELS"][type] = {}
        source_data["MODELS"][type]["DATA"] = source_model
        return source_data

    def get_models_list_from_products(self, type_products):
        list_products = []
        for products in type_products:
            for product in self.params.products_name_map[products]:
                if "ECH" in products or "CAP & FLOOR" in products:
                    if not product + "_stock_ech" in list_products:
                        list_products.append((product, "stock_ech"))
                elif "STOCK PEL" == products:
                    list_products.append((product, "stock_pel"))

                elif "STOCK NMD" in products:
                    list_products.append((product, "st"))

        return list_products

    def get_scenarios(self, etab, scenario_name, out_of_recalc_st_etabs):
        if etab in out_of_recalc_st_etabs:
            logger.info('        Pas d\'application du calculateur : {}'.format(etab))
            return []
        scenarii = self.up.scenarios_params[etab][scenario_name]
        if scenarii is None:
            return []
        scenarii = scenarii[scenarii["TYPE PRODUIT"].str.contains("STOCK")].copy()
        if not "STOCK NMD" in scenarii["TYPE PRODUIT"].unique().tolist():
            scenarii = pd.concat([scenarii, pd.DataFrame([["", "STOCK NMD"]], columns = ["NOM SCENARIO", "TYPE PRODUIT"])])

        if self.up.nom_etab == "ONEY":
            scenarii = scenarii[scenarii["TYPE PRODUIT"].str.contains("PN NMD|STOCK NMD")].copy()

        return scenarii


