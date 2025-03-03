import traceback
import gc
from calculateur.calc_params import model_params as mod
import calculateur.services.projection_services.ech.run_calculator_stock_ech as calc_stock
import calculateur.services.projection_services.ech.run_calculator_pn_ech as calc_pn
import calculateur.services.projection_services.nmd.run_calculator_stock_nmd as calc_nmd_st
import calculateur.services.projection_services.nmd.run_calculator_pn_nmd as calc_nmd_pn
import calculateur.services.projection_services.nmd.run_nmd_template as nmd_tmp
import calculateur.services.projection_services.nmd.run_nmd_spreads as nmd_sp
from calculateur.reports.main_report import Reports
from calculateur.calc_params.simuls_parameters import SimulParameters
import utils.excel_openpyxl as ex
from mappings import general_mappings as gmp
import logging

logger = logging.getLogger(__name__)


class Simulator():
    def __init__(self, etabs_list, horizon, dar,  products_list, rate_file_list, rate_scenario_list,
                 model_ech_list, model_nmd_pel_list, zc_file_paths = [], tci_rates_file_path =[],
                 liq_rates_file_path =[],  exit_indicators_type = ["GPLIQ"], is_tci = False, max_pn = 60, tci_contract_perimeter = [],
                 agregation_level = "AG", cr_immo_cal_mode = "quick", batch_sizes = {}, output_folder = "",
                 benchmark_file = "", rco_benchmark_file = "", is_liq_report = False, is_tx_report = False,
                 model_report = False, det_report = False, agreg_report = False, name_run = "RUN",
                 interpolate_curves = True, rco_like_report=False, level_logger=logging.DEBUG,
                 exec_mode = "simul", rate_file_formatted=True, ech_pn_path = "",
                 nmd_pn_path = "", nmd_pn_calage_path = ""):

        logger.info("LOADING MAPPINGS")
        self.sp = SimulParameters()
        self.sp.set_simul_parameters(etabs_list = etabs_list, horizon=horizon, dar=dar, max_pn=max_pn,
                                     products_names=products_list, rate_scenario_list=rate_scenario_list,
                                     model_ech_list = model_ech_list, model_nmd_pel_list = model_nmd_pel_list,
                                     tci_contract_perimeter = tci_contract_perimeter,
                                     exit_indicators_type=exit_indicators_type, cr_immo_cal_mode = cr_immo_cal_mode,
                                     agregation_level = agregation_level, batch_sizes = batch_sizes,
                                     output_folder=output_folder,
                                     interpolate_curves=interpolate_curves, rate_file_list=rate_file_list,
                                     zc_file_paths = zc_file_paths, tci_rates_file_path = tci_rates_file_path,
                                     liq_rates_file_path = liq_rates_file_path,
                                     level_logger = level_logger, name_run=name_run, exec_mode = exec_mode,
                                     rate_file_formatted=rate_file_formatted, ech_pn_path = ech_pn_path,
                                     nmd_pn_path = nmd_pn_path, nmd_pn_calage_path = nmd_pn_calage_path)

        self.rp = Reports(sim_params = self.sp, benchmark_file = benchmark_file,
                          rco_benchmark_file = rco_benchmark_file, is_liq_report = is_liq_report,
                          is_tx_report = is_tx_report, model_report = model_report,
                          det_report = det_report, agreg_report = agreg_report, name_run = name_run,
                          rco_like_report = rco_like_report, horizon = horizon, is_tci = is_tci)


    def launch_simulator(self):
        logger.info("DEBUT DE LA SIMUL '%s'" % (self.sp.name_run))
        try:
            for (rates_file, rate_sc, zc_file, tci_rate_file, liq_rate_file, product, ech_model_file, nmd_model_file,
                 etab) in zip(self.sp.rate_file_list, self.sp.rate_scenario_list, self.sp.zc_file_list,
                              self.sp.tci_rate_file_list, self.sp.liq_rate_file_list, self.sp.products_list,
                              self.sp.model_ech_list, self.sp.model_nmd_pel_list, self.sp.etabs_list):
                self.sp.get_scenario_rates_data(rates_file, rate_sc, zc_file, tci_rate_file, liq_rate_file)
                self.sp.set_current_simul_params(rate_sc, ech_model_file, nmd_model_file, etab, product)
                logger.info("   SIMULATION DU SCENARIO: %s * %s * %s * %s * %s"
                            % (self.sp.rates_file, rate_sc, product, self.sp.model, etab))
                cls_ag_data_list = self.launch_simulation()
                self.rp.generate_report(cls_ag_data_list)
        except:
            logger.error(traceback.format_exc())
            root = logging.getLogger()
            root.handlers = []

        finally:
            self.wrap_up_simulation()

        logger.info("FIN DE LA SIMUL '%s'" % (self.sp.name_run))

    def launch_simulation(self):
        if self.sp.product in mod.models_ech_st:
            cls_ag_data\
                = calc_stock.run_calculator_stock_ech(self.sp.dar, self.sp.horizon, self.sp.sources_dic,
                                                      self.sp.product , tx_params=self.sp.tx_params.copy(),
                                                      exit_indicators_type=self.sp.exit_indicators_type,
                                                      batch_size=self.sp.batch_size_by_product[self.sp.product],
                                                      agregation_level = self.sp.agregation_level,
                                                      cr_immo_cal_mode=self.sp.cr_immo_cal_mode,
                                                      exec_mode = self.sp.exec_mode)

            return {"STOCK" : cls_ag_data}

        elif self.sp.product in mod.models_ech_pn:
            cls_ag_data \
                = calc_pn.run_calculator_pn_ech(self.sp.dar, self.sp.horizon, self.sp.sources_dic,
                                                self.sp.product,
                                                tx_params=self.sp.tx_params.copy(),
                                                exit_indicators_type=self.sp.exit_indicators_type,
                                                map_bassins=gmp.map_pass_alm["BASSINS"]["TABLE"], max_pn=self.sp.max_pn,
                                                batch_size=self.sp.batch_size_by_product[self.sp.product],
                                                agregation_level = self.sp.agregation_level,
                                                cr_immo_cal_mode=self.sp.cr_immo_cal_mode,
                                                exec_mode = self.sp.exec_mode)

            return {"PN" : cls_ag_data}

        elif self.sp.product in mod.models_nmd_st:
            cls_ag_data \
                = calc_nmd_st.run_calculator_nmd_stock(self.sp.dar, self.sp.horizon, self.sp.sources_dic,
                                                       self.sp.product , tx_params=self.sp.tx_params.copy(),
                                                       with_dyn_data=False,
                                                       batch_size=self.sp.batch_size_by_product[self.sp.product],
                                                       agregation_level=self.sp.agregation_level,
                                                       exit_indicators_type=self.sp.exit_indicators_type,
                                                       tci_contract_perimeter=self.sp.tci_perimeter,
                                                       exec_mode = self.sp.exec_mode)

            return {"STOCK" : cls_ag_data}

        elif self.sp.product  in mod.models_nmd_pn:
            cls_nmd_tmp = nmd_tmp.run_nmd_template_getter(self.sp.sources_dic, self.sp.etab, self.sp.dar)

            cls_nmd_spreads = nmd_sp.run_nmd_spreads(self.sp.etab, self.sp.horizon, self.sp.sources_dic,
                                                     cls_nmd_tmp)

            cls_ag_st_nmd \
                = calc_nmd_st.run_calculator_nmd_stock(self.sp.dar, self.sp.horizon, self.sp.sources_dic,
                                                       self.sp.product, cls_nmd_spreads=cls_nmd_spreads,
                                                       tx_params=self.sp.tx_params.copy(),
                                                       exit_indicators_type=["GPLIQ"] + self.sp.exit_indicators_type,
                                                       agregation_level="NMD_TEMPLATE",
                                                       with_dyn_data=True, with_pn_data=True,
                                                       batch_size=self.sp.batch_size_by_product[self.sp.product],
                                                       output_mode="all", tci_contract_perimeter=self.sp.tci_perimeter,
                                                       exec_mode = self.sp.exec_mode)

            cls_ag_pn_nmd \
                = calc_nmd_pn.run_calculator_pn_nmd(self.sp.dar, self.sp.horizon, self.sp.sources_dic, self.sp.product,
                                                    self.sp.etab, cls_nmd_tmp, cls_ag_st_nmd.compiled_indics,
                                                    tx_params=self.sp.tx_params.copy(), cls_nmd_spreads=cls_nmd_spreads,
                                                    exit_indicators_type=self.sp.exit_indicators_type,
                                                    agregation_level=self.sp.agregation_level, with_dyn_data=True,
                                                    batch_size=self.sp.batch_size_by_product[self.sp.product],
                                                    tci_contract_perimeter=self.sp.tci_perimeter,
                                                    max_pn=self.sp.max_pn, exec_mode = self.sp.exec_mode)

            return {"STOCK" : cls_ag_st_nmd, "PN" : cls_ag_pn_nmd}

        logger.info("   End of Simulation for %s of ETAB %s" % (self.sp.product, self.sp.etab))


    def wrap_up_simulation(self):
        try:
            for name_model in self.sp.sources_dic["MODELS"]:
                try:
                    ex.close_workbook(self.sp.sources_dic["MODELS"][name_model])
                except:
                    pass
        except:
            pass
        gc.collect()
