import dateutil
import datetime
import os
from pathlib import Path
import calculateur.utils.logger_config as log_conf
from calculateur.simul_params import model_params as mod
import logging
from ..mappings import mapping_params as mp

logger = logging.getLogger(__name__)


class SimulParameters():
    def set_simul_parameters(self, etabs_list, horizon, dar, rate_scenario_list=["@FORWARD"],
                             products_names=[], max_pn=60, model_ech_list = ["DEFAULT"],
                             model_nmd_pel_list=[("DEFAULT", "DEFAULT")], sources_folder = "",
                             exit_indicators_type = ["GPLIQ"], cr_immo_cal_mode = "quick",
                             tci_contract_perimeter = [], agregation_level = "AG", batch_sizes = {},
                             output_folder = "", interpolate_curves=True, rate_file_list=["DEFAULT"],
                             level_logger=logging.DEBUG, name_run = ""):
        self.set_horizon(horizon)
        self.set_dar(dar)
        self.set_max_pn(max_pn)
        self.set_sources_folder(sources_folder)
        self.set_products_list(products_names)
        self.set_etab_models_list(etabs_list)
        self.set_rate_scenarios(rate_scenario_list)
        self.set_ech_models_list(model_ech_list)
        self.set_rate_file_list(rate_file_list)
        self.set_nmd_pel_models_list(model_nmd_pel_list)
        self.set_tci_params(tci_contract_perimeter)
        self.set_exit_indicators_type(exit_indicators_type)
        self.set_cr_immo_cal_mode(cr_immo_cal_mode)
        self.set_agregation_level(agregation_level)
        self.set_batch_sizes(batch_sizes)
        self.set_output_folder(output_folder)
        self.set_sources_dic()
        self.get_config(level_logger = level_logger)
        self.interpolate_curves = interpolate_curves
        self.set_name_run(name_run)

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

    def set_rate_file_list(self, rate_file_list):
        self.rate_file_list = rate_file_list

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

    def set_sources_folder(self, sources_folder):
        self.sources_folder = sources_folder

    def set_output_folder(self, output_folder):
        now = datetime.datetime.now()
        now = now.strftime("%Y%m%d.%H%M.%S")
        self.output_folder = os.path.join(output_folder, "SIMUL_EXEC-%s" % str(now))
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

    def set_current_simul_params(self, rate_sc, model, etab, product):
        self.set_current_params(rate_sc, model, etab, product)
        self.get_simul_sources()

    def get_list_models(self, product):
        if product in mod.models_ech_pn + mod.models_ech_st:
            self.models_list = self.model_ech_list
        elif product in mod.models_nmd_st + mod.models_nmd_pn:
            self.models_list = self.model_nmd_pel_list

    def get_scenario_rates_data(self, rates_file, rate_scenario, interpolate_curves):
        rate_file_path = self.get_simul_rate_sources(rates_file)
        self.tx_params = mp.load_rate_params(rate_file_path, rate_scenario, self.sources_folder, interpolate_curves)

    def set_current_params(self, rate_sc, model, etab, product):
        self.rate_scenario = rate_sc
        if product in mod.models_ech_pn + mod.models_ech_st:
            self.ech_model = model
        if product in mod.models_nmd_st + mod.models_nmd_pn:
            self.nmd_model = model[0]
            self.pel_model = model[1]
        self.etab = etab
        self.product = product
        if self.product in mod.models_nmd_pn:
            self.exit_indicators_type = self.exit_indicators_type + ["PN_NMD"]

    def get_simul_sources(self):
        if self.product in mod.models_ech_st + mod.models_nmd_st :
            self.sources_dic["STOCK"] = mp.get_contract_sources_paths(self.etab, self.product, self.sources_folder)

        if self.product in mod.models_ech_pn + mod.models_nmd_pn :
            self.sources_dic["PN"] = mp.get_contract_sources_paths(self.etab, self.product, self.sources_folder)
            if self.product in mod.models_nmd_pn:
                self.sources_dic["STOCK"] = mp.get_contract_sources_paths(self.etab, "nmd_st", self.sources_folder)

        if self.product in mod.models_ech_st + mod.models_ech_pn:
            if self.ech_model == "DEFAULT":
                ech_model_file_path = mp.get_model_file_path("MODELE_ECH", self.sources_folder)
                self.ech_model = ech_model_file_path.split("\\")[-1]
            else:
                ech_model_file_path = os.path.join(self.sources_folder, "MODELES", self.ech_model)

            self.sources_dic["MODELS"]["ECH"]["DATA"] = mp.get_model_wb(ech_model_file_path)
            self.model = self.ech_model

        elif self.product in mod.models_nmd_st + mod.models_nmd_pn:
            if self.nmd_model == "DEFAULT":
                nmd_model_file_path = mp.get_model_file_path("MODELE_NMD", self.sources_folder)
                self.nmd_model = nmd_model_file_path.split("\\")[-1]
            else:
                nmd_model_file_path = os.path.join(self.sources_folder, "MODELES", self.nmd_model)
            self.sources_dic["MODELS"]["NMD"]["DATA"] = mp.get_model_wb(nmd_model_file_path)

            if self.pel_model == "DEFAULT":
                pel_model_file_path = mp.get_model_file_path("MODELE_PEL", self.sources_folder)
                self.pel_model = pel_model_file_path.split("\\")[-1]
            else:
                pel_model_file_path = os.path.join(self.sources_folder, "MODELES", self.pel_model)
            self.sources_dic["MODELS"]["PEL"]["DATA"] = mp.get_model_wb(pel_model_file_path)
            self.model = self.nmd_model + " & " + self.pel_model

    def get_simul_rate_sources(self, rates_file):
        if rates_file == "DEFAULT":
            rates_file_path = mp.get_rate_file_path("RATE-INPUT", self.sources_folder)
            self.rates_file = rates_file_path.split("\\")[-1]
        else:
            rates_file_path = os.path.join(self.sources_folder, "RATE_INDEX_DATA", rates_file)
            self.rates_file = rates_file

        return rates_file_path

    def get_config(self, level_logger=logging.DEBUG):
        username = str(os.getlogin())
        now = datetime.datetime.now()
        now = now.strftime("%Y%m%d.%H%M.%S")
        log_path = self.output_folder + "\\\\" + "LOG_%s_%s.txt" % (username, now)
        logging.getLogger().setLevel(logging.INFO)
        log_conf.load_logger(log_path, logger, level_logger=level_logger)
