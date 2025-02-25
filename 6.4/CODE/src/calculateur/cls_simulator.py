import traceback
import gc
from calculateur.simul_params import model_params as mod
import calculateur.services.projection_services.ech.run_calculator_stock_ech as calc_stock
import calculateur.services.projection_services.ech.run_calculator_pn_ech as calc_pn
import calculateur.services.projection_services.nmd.run_calculator_stock_nmd as calc_nmd_st
import calculateur.services.projection_services.nmd.run_calculator_pn_nmd as calc_nmd_pn
import calculateur.services.projection_services.nmd.run_nmd_template as nmd_tmp
import calculateur.services.projection_services.nmd.run_nmd_spreads as nmd_sp
from calculateur.reports.main_report import Reports
from calculateur.simul_params.simuls_parameters import SimulParameters
import utils.excel_utils as ex
from .mappings import mapping_params as mp
import logging

logger = logging.getLogger(__name__)


class Simulator():
    def __init__(self, sources_folder, etabs_list, horizon, dar,  products_list, rate_scenario_list = ["@FORWARD"],
                 rate_file_list = ["DEFAULT"], model_ech_list = ["DEFAULT"],
                 model_nmd_pel_list = [("DEFAULT", "DEFAULT")],
                 exit_indicators_type = ["GPLIQ"], is_tci = False, max_pn = 60, tci_contract_perimeter = [],
                 agregation_level = "AG", cr_immo_cal_mode = "quick", batch_sizes = {}, output_folder = "",
                 benchmark_file = "", rco_benchmark_file = "", is_liq_report = False, is_tx_report = False,
                 model_report = False, det_report = False, agreg_report = False, name_run = "RUN",
                 interpolate_curves = True, rco_like_report=False, level_logger=logging.DEBUG):

        mp.load_mappings(sources_folder, dar)
        self.sp = SimulParameters()
        self.sp.set_simul_parameters(etabs_list = etabs_list, horizon=horizon, dar=dar, max_pn=max_pn,
                                     products_names=products_list, rate_scenario_list=rate_scenario_list,
                                     model_ech_list = model_ech_list, model_nmd_pel_list = model_nmd_pel_list,
                                     tci_contract_perimeter = tci_contract_perimeter,
                                     exit_indicators_type=exit_indicators_type, cr_immo_cal_mode = cr_immo_cal_mode,
                                     agregation_level = agregation_level, batch_sizes = batch_sizes,
                                     sources_folder = sources_folder, output_folder=output_folder,
                                     interpolate_curves=interpolate_curves, rate_file_list=rate_file_list,
                                     level_logger = level_logger, name_run= name_run)

        self.rp = Reports(sim_params = self.sp, benchmark_file = benchmark_file,
                          rco_benchmark_file = rco_benchmark_file, is_liq_report = is_liq_report,
                          is_tx_report = is_tx_report, model_report = model_report,
                          det_report = det_report, agreg_report = agreg_report, name_run = name_run,
                          rco_like_report = rco_like_report, horizon = horizon, is_tci = is_tci)



    def launch_simulator(self):
        logger.info("RUNNING SIMUL '%s'" % (self.sp.name_run))
        try:
            for rates_file in self.sp.rate_file_list:
                for rate_sc in self.sp.rate_scenario_list:
                    self.sp.get_scenario_rates_data(rates_file, rate_sc, self.sp.interpolate_curves)
                    for product in self.sp.products_list:
                        self.sp.get_list_models(product)
                        for model in self.sp.models_list:
                            for etab in self.sp.etabs_list:
                                self.sp.set_current_simul_params(rate_sc, model, etab, product)
                                logger.info("   SIMULATING SCENARIO: %s * %s * %s * %s * %s"
                                            % (self.sp.rates_file, rate_sc, product, self.sp.model, etab))
                                cls_ag_data_list = self.launch_simulation()
                                self.rp.generate_report(cls_ag_data_list)
        except:
            logger.error(traceback.format_exc())
            root = logging.getLogger()
            root.handlers = []

        finally:
            self.wrap_up_simulation()
        logger.info("RUNNING SIMUL '%s'" % (self.sp.name_run))

    def launch_simulation(self):
        if self.sp.product in mod.models_ech_st:
            cls_ag_data\
                = calc_stock.run_calculator_stock_ech(self.sp.dar, self.sp.horizon, self.sp.sources_dic,
                                                      self.sp.product , tx_params=self.sp.tx_params.copy(),
                                                      exit_indicators_type=self.sp.exit_indicators_type,
                                                      batch_size=self.sp.batch_size_by_product[self.sp.product],
                                                      agregation_level = self.sp.agregation_level,
                                                      cr_immo_cal_mode=self.sp.cr_immo_cal_mode)

            return {"STOCK" : cls_ag_data}

        elif self.sp.product in mod.models_ech_pn:
            cls_ag_data \
                = calc_pn.run_calculator_pn_ech(self.sp.dar, self.sp.horizon, self.sp.sources_dic,
                                                self.sp.product,
                                                tx_params=self.sp.tx_params.copy(),
                                                exit_indicators_type=self.sp.exit_indicators_type,
                                                map_bassins=mp.mp_bassins, max_pn=self.sp.max_pn,
                                                batch_size=self.sp.batch_size_by_product[self.sp.product],
                                                agregation_level = self.sp.agregation_level,
                                                cr_immo_cal_mode=self.sp.cr_immo_cal_mode)

            return {"PN" : cls_ag_data}

        elif self.sp.product in mod.models_nmd_st:
            cls_ag_data \
                = calc_nmd_st.run_calculator_nmd_stock(self.sp.dar, self.sp.horizon, self.sp.sources_dic,
                                                       self.sp.product , tx_params=self.sp.tx_params.copy(),
                                                       with_dyn_data=False,
                                                       batch_size=self.sp.batch_size_by_product[self.sp.product],
                                                       agregation_level=self.sp.agregation_level,
                                                       exit_indicators_type=self.sp.exit_indicators_type,
                                                       tci_contract_perimeter=self.sp.tci_perimeter)

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
                                                       output_mode="all", tci_contract_perimeter=self.sp.tci_perimeter)

            cls_ag_pn_nmd \
                = calc_nmd_pn.run_calculator_pn_nmd(self.sp.dar, self.sp.horizon, self.sp.sources_dic, self.sp.product,
                                                    self.sp.etab, cls_nmd_tmp, cls_ag_st_nmd.compiled_indics,
                                                    tx_params=self.sp.tx_params.copy(), cls_nmd_spreads=cls_nmd_spreads,
                                                    exit_indicators_type=self.sp.exit_indicators_type,
                                                    agregation_level=self.sp.agregation_level, with_dyn_data=True,
                                                    batch_size=self.sp.batch_size_by_product[self.sp.product],
                                                    tci_contract_perimeter=self.sp.tci_perimeter,
                                                    max_pn=self.sp.max_pn)

            return {"STOCK" : cls_ag_st_nmd, "PN" : cls_ag_pn_nmd}

        logger.info("   End of Simulation for %s of ETAB %s" % (self.sp.product, self.sp.etab))


    def wrap_up_simulation(self):
        try:
            for name_model in self.sp.sources_dic["MODELS"]:
                try:
                    self.sp.sources_dic["MODELS"][name_model].Close(False)
                except:
                    pass
        except:
            pass
        try:
            ex.able_Excel(True)
        except:
            pass
        try:
            for wb in ex.xl.Workbooks:
                wb.Close(False)
            ex.xl.Quit()
            ex.xl = None
            ex.kill_excel()
        except:
            try:
                ex.xl.Quit()
                ex.kill_excel()
            except:
                pass
        gc.collect()
        input("Type any key...")
