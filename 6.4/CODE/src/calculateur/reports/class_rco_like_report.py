import numpy as np
import pandas as pd
import pickle
import os
import utils.general_utils as gu
from calculateur.models.agregation.class_agregation import Agregation
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
from calculateur.simul_params import model_params as mod
from pathlib import Path


class RcoLikeReport():
    def __init__(self, cls_fields, sim_params, list_cls_agreg, horizon, output_folder, name_run):
        self.output_folder = output_folder
        self.sim_params = sim_params
        self.cls_fields = cls_fields
        self.list_cls_agreg = list_cls_agreg
        self.liquidity_report_indics = {"A$LEF": "GAP-LIQ-EF", "B$LEM" : "GAP-LIQ-EM",
                                        "E$LMN":"MNI-LIQ", "C$TEF":"GAP-TX-EF", "D$TEM" : "GAP-TX-EM"}
        self.all_cols_num_calc = ["M" + str(i) for i in range(0, 361)]
        self.all_cols_num = ["M" + str(i) for i in range(0, 121)] + ["M" + str(i) for i in range(132, 300 + 1, 12)]
        self.horizon = horizon
        self.name_run = name_run
        self.NC_OUTPUT_SCENARIO = "SCENARIO_ID"
        self.NC_DARNUM = "DARNUM"
        self.NC_OUTPUT_BILAN = "category_code".upper()
        self.NC_OUTPUT_CUR = "ccy_code".upper()
        self.NC_OUTPUT_GESTION = "int_gest".upper()
        self.NC_OUTPUT_PALIER = "palier_conso".upper()
        self.NC_OUTPUT_MATUR = "matur_ini".upper()
        self.NC_DATE_PERIOD = "datefin_period".upper()
        self.NC_MONTANT = "montant".upper()

        self.renames_dic = {self.cls_fields.NC_LDP_CURRENCY: self.NC_OUTPUT_CUR,
                            self.cls_fields.NC_LDP_GESTION: self.NC_OUTPUT_GESTION,
                            self.cls_fields.NC_LDP_PALIER: self.NC_OUTPUT_PALIER,
                            self.cls_fields.NC_LDP_MATUR: self.NC_OUTPUT_MATUR}

        self.output_vars = [self.cls_fields.NC_LDP_BASSIN, self.cls_fields.NC_LDP_ETAB, self.NC_DARNUM,
                            self.NC_OUTPUT_SCENARIO,
                            self.NC_OUTPUT_BILAN, self.cls_fields.NC_LDP_CONTRACT_TYPE, self.NC_OUTPUT_MATUR,
                            self.NC_OUTPUT_CUR,
                            self.cls_fields.NC_LDP_RATE_CODE, self.cls_fields.NC_LDP_MARCHE, self.NC_OUTPUT_GESTION,
                            self.NC_OUTPUT_PALIER]

    def generate_rco_like_reports(self):
        etab = self.sim_params.etab
        Path(os.path.join(self.output_folder, etab)).mkdir(parents=True, exist_ok=True)
        compiled_data = self.load_report_data()
        if len(compiled_data) > 0:
            compiled_data = self.add_missing_cols(compiled_data)
            for indic, name_indic in self.liquidity_report_indics.items():
                formated_data_indic = self.format_report_data(compiled_data, indic)
                self.save_data_indic(formated_data_indic, name_indic)

    def load_report_data(self):
        data_report_all = []
        for type_data, cls_agreg in self.list_cls_agreg.items():
            self.qual_cols = cls_agreg.ag_vars
            self.num_cols = cls_agreg.num_cols
            for i in range(0, len(cls_agreg.temp_files_name)):
                name_file = os.path.join(cls_agreg.temp_dir.name, cls_agreg.temp_files_name[i])
                with open(name_file, 'rb') as handle:
                    data_report = pickle.load(handle)
                if len(data_report_all) > 0:
                    data_report_all = pd.concat([data_report_all, data_report])
                else:
                    data_report_all = data_report.copy()

        return data_report_all

    def format_report_data(self, compiled_data, indic):
        compiled_data_indic = compiled_data[compiled_data[pa.NC_PA_IND03] == indic].copy()
        compiled_data_indic = self.format_qual_cols(compiled_data_indic)
        formated_data_indic, cols_unpivot = self.format_num_cols(compiled_data_indic, indic)
        return formated_data_indic

    def format_num_cols(self, compiled_data_indic, indic):
        num_cols = ["M" + str(i) for i in range(0, self.horizon + 1)]
        compiled_data_indic = compiled_data_indic[self.output_vars + num_cols].copy()
        compiled_data_indic = self.change_num_cols_post_year_10(compiled_data_indic, indic)
        compiled_data_indic, cols_unpivot = self.rename_date_columns(compiled_data_indic, num_cols)
        cols_unpivot = cols_unpivot[:121] + cols_unpivot[132:self.horizon + 1:12]
        compiled_data_indic = compiled_data_indic[self.output_vars + cols_unpivot].copy()
        cols_keep = [x for x in compiled_data_indic.columns if x not in cols_unpivot]

        compiled_data_indic = self.make_passif_negative(compiled_data_indic, cols_unpivot)

        formated_data_indic = pd.melt(compiled_data_indic, id_vars=cols_keep, value_vars=cols_unpivot,
                                      var_name=self.NC_DATE_PERIOD, value_name=self.NC_MONTANT.upper())
        return formated_data_indic, cols_unpivot

    def change_num_cols_post_year_10(self, data, indic):
        if indic in ["A$LEF", "B$LEM", "C$TEF", "D$TEM"]:
            for i in range(132, 301, 12):
                data["M%s" % i] = data.loc[:, "M%s" % (i - 11):"M%s" % i].values.mean(axis=1)
        elif indic in ["F$LMN"]:
            for i in range(132, 301, 12):
                data["M%s" % i] = data.loc[:, "M%s" % (i - 11):"M%s" % i].values.sum(axis=1)
        return data

    def make_passif_negative(self, data, num_cols):
        is_passif = data[self.NC_OUTPUT_BILAN].str.contains("PASSIF").values.reshape(data.shape[0], 1)
        num_vals = data[num_cols].values
        data[num_cols] = np.where(is_passif, -1 * num_vals, num_vals)
        return data

    def save_data_indic(self, formated_data_indic, name_indic):
        file_name = self.sim_params.etab.upper() + "_" + self.sim_params.dar.strftime("%Y-%m-%d") + "_" + name_indic + ".tab"
        file_path = os.path.join(self.output_folder, self.sim_params.etab, file_name)
        if not os.path.isfile(file_path):
            formated_data_indic.to_csv(file_path, sep="\t", index=False)
        else:
            formated_data_indic.to_csv(file_path, sep="\t", index=False, mode="a", header=False)

    def rename_date_columns(self, data, num_cols):
        start = self.sim_params.dar
        period = self.horizon + 1
        end_month_dates = pd.DataFrame(pd.date_range(start=start, periods=period, freq="ME")).iloc[:, 0].dt.strftime(
            '%d/%m/%Y').values.tolist()
        data = data.rename(columns={x: y for x, y in zip(num_cols, end_month_dates)})
        return data, end_month_dates

    def add_missing_cols(self, data):
        data[self.cls_fields.NC_LDP_RATE_CODE] \
            = data[self.cls_fields.NC_LDP_RATE_CODE].fillna("FIXE").replace("", "FIXED")

        contract_type_pref = data[self.cls_fields.NC_LDP_CONTRACT_TYPE].str[:2]
        sens = data[self.cls_fields.NC_LDP_BUY_SELL].str.upper()
        cases = [contract_type_pref == "A-", contract_type_pref == "P-",
                 (contract_type_pref == "HB") & np.array((self.sim_params.product[:2] == "a-")),
                 (contract_type_pref == "HB") & np.array((self.sim_params.product[:2] == "p-")),
                 (contract_type_pref == "HB") & (sens == "S"), (contract_type_pref == "HB") & (sens != "S")]
        values = ["ACTIF", "PASSIF", "HB ACTIF","HB PASSIF", "HB PASSIF", "HB ACTIF"]
        data[self.NC_OUTPUT_BILAN] = np.select(cases, values, default="ACTIF")

        return data

    def format_qual_cols(self, data):
        data[self.NC_OUTPUT_SCENARIO] = self.name_run
        data[self.NC_DARNUM] = self.sim_params.dar.strftime("%Y%m%d")
        data = gu.force_integer_to_string(data, self.cls_fields.NC_LDP_PALIER)
        data = data.rename(columns=self.renames_dic)
        return data
