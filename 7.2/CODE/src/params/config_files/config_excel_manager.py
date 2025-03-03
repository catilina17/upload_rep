from pathlib import Path
import os
import utils.excel_openpyxl as ex
import params.config_files.config_param_names_excel as names_ex
class ExcelConfigManager():
    def __init__(self, args):
        self.config_file = ex.load_workbook_openpyxl(args.ref_file_path, read_only=True, data_only=True)
        self.config_path = args.ref_file_path
        self.config_name = os.path.basename(args.ref_file_path)
        self.config_base_path = Path(self.config_path).parent
        self.load_config_main_params_names()
        self.import_config_type()

    def import_config_type(self):
        self.ex = ex

    def load_config_main_params_names(self):
        self.names_ex = names_ex

    def get_value_from_named_ranged(self, name):
        return self.ex.get_value_from_named_ranged(self.config_file, name)

    def get_dataframe_from_named_ranged(self, name, header=True):
        return self.ex.get_dataframe_from_range(self.config_file, name, header=header)

