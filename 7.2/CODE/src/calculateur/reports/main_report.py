from ..reports.class_custom_reports import Custom_Reports
from ..reports.class_model_report import Models_Reports
from ..reports.class_rco_like_report import RcoLikeReport
from calculateur.models.data_manager.data_format_manager.class_fields_manager import Data_Fields
import platform
from utils import excel_utils as ex2

class Reports():
    def __init__(self, sim_params, rco_benchmark_file,
               benchmark_file = "", is_liq_report = True, is_tx_report = False, horizon = 300,
               model_report=False, det_report=False, agreg_report = False, name_run = "", rco_like_report=False,
               is_tci = False):

        self.output_folder = sim_params.output_folder
        self.model_report = model_report
        self.det_report = det_report
        self.agreg_report = agreg_report
        self.rco_like_report = rco_like_report
        self.benchmark_file = benchmark_file
        self.is_liq_report = is_liq_report
        self.is_tx_report = is_tx_report
        self.sim_params = sim_params
        self.name_run = name_run
        self.rco_benchmark_file_def = rco_benchmark_file
        self.horizon = horizon
        self.is_tci = is_tci

    def generate_report(self, list_cls_agreg):
        self.tx_params = self.sim_params.tx_params

        self.rco_benchmark_file\
            = (self.rco_benchmark_file_def.replace('@ETAB@', self.sim_params.etab)
               .replace('@RATE_SCENARIO@', self.sim_params.rate_scenario))

        cls_fields = Data_Fields()

        cls_custom_report = Custom_Reports(cls_fields, list_cls_agreg ,self.output_folder, self.sim_params,
                                           self.sim_params.dar, self.sim_params.product,
                                           self.is_liq_report | self.model_report, self.is_tx_report | self.model_report,
                                           self.det_report, self.agreg_report, self.model_report,
                                           self.tx_params, is_tci=self.is_tci, name_run = self.name_run)

        cls_custom_report.generate_agregated_report()

        cls_custom_report.generate_detailed_report()

        if (self.agreg_report or self.model_report):
            cls_custom_report.add_rco_data(self.rco_benchmark_file, self.sim_params.product, self.agreg_report,
                                           self.model_report)

            if self.model_report and platform.system().upper() == "WINDOWS":
                ex2.load_excel()
                cls_report_model = Models_Reports(self.sim_params, cls_custom_report, self.benchmark_file, self.is_tci)
                cls_report_model.generate_models_synthesis_report(self.sim_params.product, self.sim_params.etab)


        if self.rco_like_report:
            cls_rco_like_report = RcoLikeReport(cls_fields, self.sim_params, list_cls_agreg, self.horizon,
                                                self.output_folder, self.name_run)
            cls_rco_like_report.generate_rco_like_reports()