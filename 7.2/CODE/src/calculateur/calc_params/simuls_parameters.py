import dateutil
import datetime
import os

import pandas as pd

from mappings import mapping_products as mpr
from calculateur.rates_transformer.rate_calc_params import Calculator_RateParams_Manager
from pathlib import Path
import calculateur.utils.logger_config as log_conf
from calculateur.calc_params import model_params as mod
import logging
from utils import excel_openpyxl as ex
from params.sources_params import SourcesParams

logger = logging.getLogger(__name__)


class SimulParameters():
    def set_simul_parameters(self, etabs_list, horizon, dar, rate_scenario_list,
                             products_names=[], max_pn=60, model_ech_list = [],
                             model_nmd_pel_list = [],
                             exit_indicators_type = ["GPLIQ"], cr_immo_cal_mode = "quick",
                             tci_contract_perimeter = [], agregation_level = "AG", batch_sizes = {},
                             output_folder = "", interpolate_curves=True, rate_file_list=[],
                             zc_file_paths = [], tci_rates_file_path = [], liq_rates_file_path =[],
                             level_logger=logging.DEBUG, name_run="", exec_mode="simul",
                             rate_file_formatted=True, ech_pn_path = "", nmd_pn_path = "", nmd_pn_calage_path = ""):

        self.sources_cls = SourcesParams()
        self.dic_rates = {}
        self.set_horizon(horizon)
        self.set_dar(dar)
        self.set_max_pn(max_pn)
        self.set_products_list(products_names)
        self.set_etab_models_list(etabs_list)
        self.set_rate_scenarios(rate_scenario_list)
        self.set_ech_models_list(model_ech_list)
        self.set_rate_file_list(rate_file_list, rate_file_formatted)
        self.set_zc_file_list(zc_file_paths)
        self.set_tci_rate_file_list(tci_rates_file_path)
        self.set_liq_rate_file_list(liq_rates_file_path)
        self.set_nmd_pel_models_list(model_nmd_pel_list)
        self.set_tci_params(tci_contract_perimeter)
        self.set_exit_indicators_type(exit_indicators_type)
        self.set_cr_immo_cal_mode(cr_immo_cal_mode)
        self.set_agregation_level(agregation_level)
        self.set_batch_sizes(batch_sizes)
        self.set_output_folder(output_folder)
        self.set_sources_dic()
        self.set_pn_sources(ech_pn_path, nmd_pn_path, nmd_pn_calage_path)
        self.get_config(level_logger = level_logger)
        self.interpolate_curves = interpolate_curves
        self.set_name_run(name_run)
        self.set_exec_mode(exec_mode)

    def set_pn_sources(self, ech_pn_path, nmd_pn_path, nmd_pn_calage_path):
        self.ech_pn_path = ech_pn_path
        self.nmd_pn_path = nmd_pn_path
        self.nmd_pn_calage_path = nmd_pn_calage_path

    def set_exec_mode(self, exec_mode):
        self.exec_mode = exec_mode

    def set_name_run(self, name_run):
        self.name_run = name_run
    def set_sources_dic(self):
        self.sources_dic = {}
        self.sources_dic["STOCK"] = {}
        self.sources_dic["PN"] = {}
        self.sources_dic["MODELS"] = {}
        self.sources_dic["MODELS"]["ECH"] = {}
        self.sources_dic["MODELS"]["PEL"] = {}
        self.sources_dic["MODELS"]["NMD"] = {}

    def set_batch_sizes(self, batch_sizes):
        self.batch_size_by_product = mod.batch_size_by_products
        self.batch_size_by_product.update(batch_sizes)

    def set_agregation_level(self, agregation_level):
        self.agregation_level = agregation_level

    def set_cr_immo_cal_mode(self, cr_immo_cal_mode):
        self.cr_immo_cal_mode = cr_immo_cal_mode

    def set_exit_indicators_type(self, exit_indicators_type):
        self.exit_indicators_type = exit_indicators_type

    def set_rco_benchmark_file(self, rco_benchmark_file):
        self.rco_benchmark_file = rco_benchmark_file

    def set_tci_params(self, tci_contract_perimeter):
        self.tci_perimeter = tci_contract_perimeter

    def set_ech_models_list(self, model_ech_list):
        self.model_ech_list = model_ech_list

    def set_rate_file_list(self, rate_file_list, rate_file_formatted):
        self.rate_file_list = rate_file_list
        self.rate_file_formatted = rate_file_formatted

    def set_zc_file_list(self, zc_file_list):
        if zc_file_list == []:
            zc_file_list = [""] * len(self.rate_file_list )
        self.zc_file_list = zc_file_list

    def set_tci_rate_file_list(self, tci_rate_file_list):
        if tci_rate_file_list == []:
            tci_rate_file_list = [""] * len(self.rate_file_list)
        self.tci_rate_file_list = tci_rate_file_list

    def set_liq_rate_file_list(self,liq_rate_file_list):
        if liq_rate_file_list == []:
            liq_rate_file_list = [""] * len(self.rate_file_list)
        self.liq_rate_file_list = liq_rate_file_list

    def set_etab_models_list(self, etabs_list):
        self.etabs_list = etabs_list

    def set_nmd_pel_models_list(self, model_nmd_pel_list):
        self.model_nmd_pel_list = model_nmd_pel_list

    def set_rate_scenarios(self, rate_scenario_list):
        self.rate_scenario_list = rate_scenario_list

    def set_products_list(self, products_list):
        if products_list == []:
            self.products_list = mod.models_names_list
        else:
            self.products_list = products_list

    def set_max_pn(self, max_pn):
        self.max_pn = max_pn

    def set_horizon(self, horizon):
        self.horizon = min(360, horizon)

    def set_dar(self, dar):
        self.dar = dateutil.parser.parse(str(dar)).replace(tzinfo=None)

    def set_output_folder(self, output_folder):
        #now = datetime.datetime.now()
        #now = now.strftime("%Y%m%d.%H%M.%S")
        self.output_folder = output_folder
        #os.path.join(output_folder, "SIMUL_EXEC-%s" % str(now))
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

    def set_current_simul_params(self, rate_sc, ech_model_file, nmd_model_file, etab, product):
        self.set_current_params(rate_sc, etab, product, ech_model_file, nmd_model_file)
        self.get_simul_sources(ech_model_file, nmd_model_file)

    def get_scenario_rates_data(self, rate_file_path, rate_scenario, zc_file, tci_rate_file, liq_rate_file):
        if not (rate_file_path + "_" + rate_scenario) in self.dic_rates:
            self.rates_file = os.path.basename(rate_file_path)
            crm = Calculator_RateParams_Manager(rate_file_path = rate_file_path, interpolate_curves=self.interpolate_curves,
                                                scenario_name=rate_scenario, zc_curves_path=zc_file,
                                                tci_file_path=tci_rate_file, liq_file_path=liq_rate_file,
                                                is_formated=self.rate_file_formatted, is_scenario=True)
            self.tx_params = crm.load_rate_params()
            self.dic_rates[rate_file_path + "_" + rate_scenario] = self.tx_params.copy()
        else:
            self.tx_params = self.dic_rates[rate_file_path + "_" + rate_scenario].copy()

    def set_current_params(self, rate_sc, etab, product, ech_model_file, nmd_model_file):
        self.rate_scenario = rate_sc
        self.etab = etab
        self.product = product
        if self.product in mod.models_nmd_pn:
            self.exit_indicators_type = self.exit_indicators_type + ["PN_NMD"]

        if self.product in mod.models_ech_st + mod.models_ech_pn:
            self.model =  os.path.basename(ech_model_file)
        elif self.product in mod.models_nmd_st + mod.models_nmd_pn:
            self.model = os.path.basename(nmd_model_file)

    def get_simul_sources(self, ech_model_file, nmd_model_file):
        if self.product in mod.models_ech_st:
            self.sources_dic["STOCK"]\
                = self.sources_cls.get_contract_sources_paths(self.etab, mpr.products_map[self.product])

        if self.product in mod.models_nmd_st + mod.models_nmd_pn:
            self.sources_dic["STOCK"]\
                = self.sources_cls.get_contract_sources_paths(self.etab, mpr.products_map["nmd_st"])

        if self.product in mod.models_ech_st + mod.models_ech_pn:
            self.sources_dic["MODELS"]["ECH"]["DATA"] = ex.load_workbook_openpyxl(ech_model_file, read_only=True)
            if self.product in mod.models_ech_pn:
                self.sources_dic["PN"]["LDP"] = {}
                self.sources_dic["PN"]["LDP"]["CHEMIN"] = self.ech_pn_path
                self.sources_dic["PN"]["LDP"]["DELIMITER"] = ";"
                self.sources_dic["PN"]["LDP"]["DECIMAL"] = ","

        elif self.product in mod.models_nmd_st + mod.models_nmd_pn:
            self.sources_dic["MODELS"]["NMD"]["DATA"] = ex.load_workbook_openpyxl(nmd_model_file, read_only=True)
            if self.product in mod.models_nmd_pn:
                self.sources_dic["PN"]["LDP"] = {}
                self.sources_dic["PN"]["LDP"]["CHEMIN"] = self.nmd_pn_path
                self.sources_dic["PN"]["LDP"]["DELIMITER"] = ";"
                self.sources_dic["PN"]["LDP"]["DECIMAL"] = ","
                self.sources_dic["PN"]["CALAGE"] = {}
                self.sources_dic["PN"]["CALAGE"]["CHEMIN"] = self.nmd_pn_calage_path
                self.sources_dic["PN"]["CALAGE"]["DELIMITER"] = ";"
                self.sources_dic["PN"]["CALAGE"]["DECIMAL"] = ","


    def get_config(self, level_logger=logging.DEBUG):
        import getpass
        username = str(getpass.getuser()).upper()
        now = datetime.datetime.now()
        now = now.strftime("%Y%m%d.%H%M.%S")
        log_path = os.path.join(self.output_folder, "LOG_%s_%s.txt" % (username, now))
        logging.getLogger().setLevel(logging.INFO)
        log_conf.load_logger(log_path, logger, level_logger=level_logger)
