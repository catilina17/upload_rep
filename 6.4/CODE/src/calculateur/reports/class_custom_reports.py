import numpy as np
import pandas as pd
import pickle
import os
import dateutil
from calculateur.models.agregation.class_agregation import Agregation
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
from calculateur.simul_params import model_params as mod


class Custom_Reports():
    """
    Formate les données
    """

    def __init__(self, cls_fields, list_cls_agreg, output_folder, sim_params, dar, produit, is_liquidity_report,
                 is_taux_report, gen_detailed_report, gen_agreg_report, gen_model_report,
                 tx_params, is_tci=False, name_run=""):

        self.list_cls_agreg = list_cls_agreg
        self.cls_fields = cls_fields
        self.data_report = []
        self.name_product = produit
        self.is_liquidity_report = is_liquidity_report
        self.is_taux_report = is_taux_report
        self.gen_detailed_report = gen_detailed_report
        self.gen_model_report = gen_model_report
        self.gen_agreg_report = gen_agreg_report
        self.output_folder = output_folder
        self.is_tci = is_tci
        self.dar = dar
        self.output_file_name = self.get_report_file_name(sim_params)
        self.name_simul = self.get_version_name(sim_params, name_run)
        if not self.is_tci:
            self.liquidity_report_indics = ["lem", "mni"]
        else:
            self.liquidity_report_indics = ["lem", "mni_tci"]

        self.liquidity_report_det_indics = ["lef", "lem", "mni", "mni_tci", "lem_statique", "lem_renego",
                                            "mni_statique", "mni_renego", "sc_rates_statique", "sc_rates_reneg",
                                            "sc_rates", "sc_rates_tci", "effet_RA", "effet_RN", "tx_RA", "tx_RN"]
        self.taux_report_indics = ["tem"]
        self.taux_report_det_indics = ["tef", "tem", "mni_gptx"]
        self.liquidity_report_rco_indics = [pa.NC_PA_LEM, pa.NC_PA_LMN, pa.NC_PA_DEM_RCO, pa.NC_PA_DMN_RCO]
        self.taux_report_rco_indics = [pa.NC_PA_TEM]
        self.list_tf = ["a-creq", "a-crctz", "a-crctf", "a-intbq-tf", "p-intbq-tf", "p-cat-tf",
                        "a-autres-tf", "p-autres-tf", "a-crif", "a-swap-tf", "p-swap-tf", "a-security-tf",
                        "p-security-tf"]
        self.all_cols_num_calc = ["M" + str(i) for i in range(0, 361)]
        self.all_cols_num = ["M" + str(i) for i in range(0, 121)] + ["M" + str(i) for i in range(132, 300 + 1, 12)]
        self.tx_params = tx_params
        self.pass_alm_rco_map_indics = Agregation.pass_alm_rco_map_indics

    def get_report_file_name(self, sim_params):
        if sim_params.product in mod.models_ech_pn + mod.models_ech_st:
            model_tag = os.path.splitext(sim_params.ech_model)[0]
        else:
            model_tag = os.path.splitext(sim_params.nmd_model)[0] + "_" + os.path.splitext(sim_params.pel_model)[0]

        dar_stamp = str(sim_params.dar.year) + '{:02d}'.format(sim_params.dar.month) + str(sim_params.dar.day)

        return "REPORT_%s_%s_%s_%s_%s_%s%s.csv" % (dar_stamp, model_tag, sim_params.rates_file, sim_params.rate_scenario, sim_params.etab,
                                               sim_params.product, "_TCI" if self.is_tci else "")

    def get_version_name(self, sim_params, name_run):
        if sim_params.product in mod.models_ech_pn + mod.models_ech_st:
            model_tag = os.path.splitext(sim_params.ech_model)[0]
        else:
            model_tag = os.path.splitext(sim_params.nmd_model)[0] + "_" + os.path.splitext(sim_params.pel_model)[0]

        return "RUN_%s_%s_%s%s" % (name_run, model_tag, sim_params.rate_scenario, "_TCI" if self.is_tci else "")

    def generate_agregated_report(self):
        if self.gen_agreg_report or self.gen_model_report:
            j = 0
            for type_data, cls_agreg in self.list_cls_agreg.items():
                self.qual_cols = cls_agreg.ag_vars
                self.num_cols = cls_agreg.num_cols

                for i in range(0, len(cls_agreg.temp_files_name)):
                    j = j + 1
                    name_file = os.path.join(cls_agreg.temp_dir.name, cls_agreg.temp_files_name[i])
                    with open(name_file, 'rb') as handle:
                        data_report = pickle.load(handle)

                    data_report = self.select_report_indics(data_report)

                    data_report = self.add_data_qual_report(data_report, type_data)

                    data_report = self.agregate_report(data_report)

                    data_report = self.add_annual_indics(data_report)

                    if len(self.data_report) > 0:
                        self.data_report = pd.concat([self.data_report, data_report])
                    else:
                        self.data_report = data_report.copy()

                    output_file_name = "AGREG_" + self.output_file_name

                    if self.gen_agreg_report:
                        if j == 1:
                            data_report.to_csv(os.path.join(self.output_folder, output_file_name), decimal=",", sep=";",
                                               index=False, mode='w')
                        else:
                            data_report.to_csv(os.path.join(self.output_folder, output_file_name), decimal=",", sep=";",
                                               index=False, mode='a', header=False)

    def generate_detailed_report(self):
        if self.gen_detailed_report:
            j = 0
            for type_data, cls_agreg in self.list_cls_agreg.items():
                self.qual_cols = cls_agreg.ag_vars
                self.num_cols = cls_agreg.num_cols
                for i in range(0, len(cls_agreg.temp_files_name)):
                    j = j + 1
                    name_file = os.path.join(cls_agreg.temp_dir.name, cls_agreg.temp_files_name[i])
                    with open(name_file, 'rb') as handle:
                        data_report = pickle.load(handle)

                    data_report = self.select_det_report_indics(data_report)

                    data_report = self.add_data_qual_report(data_report, type_data)

                    data_report = self.add_annual_indics(data_report)

                    output_file_name = "DETAILED_" + self.output_file_name

                    if j == 1:
                        data_report.to_csv(os.path.join(self.output_folder, output_file_name), decimal=",", sep=";",
                                           index=False,
                                           mode='w')
                    else:
                        data_report.to_csv(os.path.join(self.output_folder, output_file_name), decimal=",", sep=";",
                                           index=False,
                                           mode='a', header=False)

    def filter_dim_and_market_ag_data(self, data):
        if pa.NC_PA_DIM6 in data.columns:
            filter_rco = (data[pa.NC_PA_DIM6].isnull()) | (data[pa.NC_PA_DIM6].astype(str).str.upper() != "FCT")
            filter_rco = filter_rco & (data[pa.NC_PA_MARCHE].str.upper() != "MDC")
            data = data[filter_rco].copy()
        return data

    def select_report_indics(self, data_report):
        list_indics = (self.liquidity_report_indics * self.is_liquidity_report
                       + self.taux_report_indics * self.is_taux_report)
        display_names = [self.pass_alm_rco_map_indics[x] for x in list_indics]
        data_report = data_report[data_report[pa.NC_PA_IND03].isin(display_names)].copy()

        data_report[pa.NC_PA_IND03] = data_report[pa.NC_PA_IND03].str.split("$", expand=True)[1]

        if self.is_tci:
            data_report[pa.NC_PA_IND03] = data_report[pa.NC_PA_IND03].replace("LMN_FTP", pa.NC_PA_LMN)

        return data_report

    def select_det_report_indics(self, data_report):
        list_indics = (self.liquidity_report_det_indics * self.is_liquidity_report
                       + self.taux_report_det_indics * self.is_taux_report)
        display_names = [self.pass_alm_rco_map_indics[x] for x in list_indics]
        data_report = data_report[data_report[pa.NC_PA_IND03].isin(display_names)].copy()

        data_report[pa.NC_PA_IND03] = data_report[pa.NC_PA_IND03].str.split("$", expand=True)[1]

        return data_report

    def add_data_qual_report(self, data_report, data_type):

        self.cols_num = [x for x in self.all_cols_num_calc if x in data_report]
        if self.cls_fields.NC_LDP_CONTRAT in data_report:
            data_report["CONTRAT_REF"] = data_report[self.cls_fields.NC_LDP_CONTRAT]
        else:
            data_report["CONTRAT_REF"] = "-"

        # PRODUIT
        if not self.name_product in mod.models_ech_pn:
            if (self.name_product == "cap_floor"):
                data_report[pa.NC_PA_CONTRACT_TYPE] = np.where(data_report[self.cls_fields.NC_LDP_BUY_SELL] == "B",
                                                               "A" + data_report[self.cls_fields.NC_LDP_CONTRACT_TYPE],
                                                               "P" + data_report[self.cls_fields.NC_LDP_CONTRACT_TYPE])

            elif self.name_product in ["a-swap-tf", "a-swap-tv"]:
                data_report[pa.NC_PA_CONTRACT_TYPE] = np.where(
                    data_report[self.cls_fields.NC_LDP_CONTRACT_TYPE].str.contains("HB"),
                    "A" + data_report[self.cls_fields.NC_LDP_CONTRACT_TYPE],
                    data_report[self.cls_fields.NC_LDP_CONTRACT_TYPE])

            elif self.name_product in ["p-swap-tf", "p-swap-tv"]:
                data_report[pa.NC_PA_CONTRACT_TYPE] = np.where(
                    data_report[self.cls_fields.NC_LDP_CONTRACT_TYPE].str.contains("HB"),
                    "P" + data_report[self.cls_fields.NC_LDP_CONTRACT_TYPE],
                    data_report[self.cls_fields.NC_LDP_CONTRACT_TYPE])
            else:
                data_report[pa.NC_PA_CONTRACT_TYPE] = np.where(
                    data_report[self.cls_fields.NC_LDP_CONTRACT_TYPE].str.contains("HB"),
                    "A" + data_report[self.cls_fields.NC_LDP_CONTRACT_TYPE],
                    data_report[self.cls_fields.NC_LDP_CONTRACT_TYPE])

        else:
            data_report[pa.NC_PA_CONTRACT_TYPE] = data_report[self.cls_fields.NC_LDP_CONTRACT_TYPE].values

        data_report["VERSION"] = self.name_simul

        data_report[pa.NC_PA_RATE_CODE] = data_report[self.cls_fields.NC_LDP_RATE_CODE]

        data_report[pa.NC_PA_ETAB] = data_report[self.cls_fields.NC_LDP_ETAB]

        data_report[pa.NC_PA_MARCHE] = data_report[self.cls_fields.NC_LDP_MARCHE]

        data_report[pa.NC_PA_DEVISE] = data_report[self.cls_fields.NC_LDP_CURRENCY]

        data_report[pa.NC_PA_DIM6] = data_report[self.cls_fields.NC_LDP_DIM6]

        data_report["TYPE"] = data_type

        data_report = self.filter_dim_and_market_ag_data(data_report)

        if False:  # True:
            data_report = self.add_crif_special_vars(data_report)
            data_report = data_report[
                ["VERSION", pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_MARCHE, pa.NC_PA_ETAB, pa.NC_PA_DEVISE,
                 pa.NC_PA_RATE_CODE, "CONTRAT_REF", "MAT_BUCKET",
                 "DRAC", "TYPE", pa.NC_PA_IND03] + self.cols_num + self.delta_num_cols].copy()
        else:
            data_report = data_report[
                ["VERSION", pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_MARCHE, pa.NC_PA_ETAB, pa.NC_PA_DEVISE, pa.NC_PA_RATE_CODE,
                 "CONTRAT_REF", "TYPE",
                 pa.NC_PA_IND03] + self.cols_num].copy()

        return data_report

    def add_crif_special_vars(self, data_report):
        dar = dateutil.parser.parse(str(self.dar)).replace(tzinfo=None)
        dar_mois = 12 * dar.year + dar.month
        mat_date_month = data_report[self.cls_fields.NC_LDP_MATUR_DATE]
        cases = [mat_date_month - dar_mois <= 12, mat_date_month - dar_mois <= 5 * 12,
                 mat_date_month - dar_mois <= 10 * 12, mat_date_month - dar_mois <= mat_date_month]
        vals = ["< 1 an", "1-5ans", "5-10ans", "10-20 ans"]
        data_report["MAT_BUCKET"] = np.select(cases, vals, default="> 20ans")

        val_date_month = data_report[self.cls_fields.NC_LDP_VALUE_DATE]
        cases = [(mat_date_month - val_date_month) <= 0,
                 ((mat_date_month - dar_mois) / np.maximum(0.0001, mat_date_month - val_date_month)) <= 0.5,
                 ((mat_date_month - dar_mois) / np.maximum(0.0001, mat_date_month - val_date_month)) <= 0.8]
        vals = ["DRAC 50", "DRAC 50", "DRAC 80"]
        data_report["DRAC"] = np.select(cases, vals, default="DRAC 100")

        ag_vars = ["VERSION", pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_MARCHE, pa.NC_PA_ETAB, pa.NC_PA_DEVISE,
                   pa.NC_PA_RATE_CODE, "MAT_BUCKET",
                   "DRAC", "TYPE", pa.NC_PA_IND03]

        data_report = (data_report.groupby(ag_vars, dropna=False, as_index=False).sum(numeric_only=True))

        data_report["CONTRAT_REF"] = "-"

        for i in range(1, len(self.num_cols) - 1):
            data_report["DM%s" % i] = data_report["M%s" % (i)] - data_report["M%s" % (i - 1)]

        self.delta_num_cols = ["DM%s" % i for i in range(1, len(self.num_cols) - 1)]

        return data_report

    def agregate_report(self, data_report):

        data_report = data_report[
            ["VERSION", pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_MARCHE, pa.NC_PA_DEVISE, pa.NC_PA_ETAB, pa.NC_PA_RATE_CODE,
             "TYPE", pa.NC_PA_IND03] + self.cols_num] \
            .groupby(["VERSION", pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_MARCHE, pa.NC_PA_ETAB, pa.NC_PA_DEVISE,
                      pa.NC_PA_RATE_CODE, "TYPE", pa.NC_PA_IND03], dropna=False,
                     as_index=False).sum()

        data_report["CONTRAT_REF"] = "-"

        data_report = data_report[
            ["VERSION", pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_MARCHE, pa.NC_PA_ETAB, pa.NC_PA_DEVISE,
             pa.NC_PA_RATE_CODE, "CONTRAT_REF", "TYPE", pa.NC_PA_IND03] + self.cols_num].copy()

        missing_col_nums = [x for x in self.all_cols_num_calc if x not in self.cols_num]
        data_report = pd.concat([data_report, pd.DataFrame(0, columns=missing_col_nums, index=data_report.index)],
                                axis=1)

        return data_report

    def add_annual_indics(self, data_report):
        # Interessant pour la MNI
        is_em = data_report[pa.NC_PA_IND03].isin([pa.NC_PA_LEM, pa.NC_PA_TEM])
        for i in range(1, 26):
            if "M%s" % ((i - 1) * 12 + 1) in data_report.columns and "M%s" % (i * 12) in data_report.columns:
                data_report["A%s" % i] = data_report.loc[:, "M%s" % ((i - 1) * 12 + 1):"M%s" % (i * 12)].sum(axis=1)
                data_report.loc[is_em, "A%s" % i] = data_report.loc[is_em,
                                                    "M%s" % ((i - 1) * 12 + 1):"M%s" % (i * 12)].mean(axis=1)

        return data_report

    def add_rco_data(self, fichier_rco, name_prod, is_ag_report, is_model_report):
        if fichier_rco != "":
            data_rco = pd.read_csv(fichier_rco, sep=";", decimal=",", encoding="ISO-8859-1")
            data_rco = data_rco[[pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_ETAB, pa.NC_PA_IND03, pa.NC_PA_MARCHE,
                                 pa.NC_PA_DEVISE, pa.NC_PA_RATE_CODE, pa.NC_PA_INDEX_AGREG,
                                 pa.NC_PA_PALIER] + self.all_cols_num].copy()
            list_indics = (self.liquidity_report_rco_indics * self.is_liquidity_report
                           + self.taux_report_rco_indics * self.is_taux_report)
            if self.name_product in mod.models_ech_pn:
                data_rco = data_rco[data_rco[pa.NC_PA_IND03].isin(
                    [pa.NC_PA_LEM, pa.NC_PA_DEM_RCO, pa.NC_PA_DMN_RCO, pa.NC_PA_LMN])].copy()
                filtre_dem = data_rco[pa.NC_PA_IND03].isin([pa.NC_PA_DEM_RCO])
                filtre_lem = data_rco[pa.NC_PA_IND03].isin([pa.NC_PA_LEM])
                filtre_dmn = data_rco[pa.NC_PA_IND03].isin([pa.NC_PA_DMN_RCO])
                filtre_lmn = data_rco[pa.NC_PA_IND03].isin([pa.NC_PA_LMN])
                data_rco.loc[filtre_dem, self.all_cols_num] = data_rco.loc[filtre_dem, self.all_cols_num].values - \
                                                              data_rco.loc[filtre_lem, self.all_cols_num].values
                data_rco.loc[filtre_dmn, self.all_cols_num] = data_rco.loc[filtre_dmn, self.all_cols_num].values - \
                                                              data_rco.loc[filtre_lmn, self.all_cols_num].values
                data_rco = data_rco[data_rco[pa.NC_PA_IND03].isin([pa.NC_PA_DEM_RCO, pa.NC_PA_DMN_RCO])].copy()
                data_rco[pa.NC_PA_IND03] = data_rco[pa.NC_PA_IND03].replace(pa.NC_PA_DEM_RCO, pa.NC_PA_LEM).replace(
                    pa.NC_PA_DMN_RCO, pa.NC_PA_LMN)
                del_ajust = (data_rco[pa.NC_PA_RATE_CODE] == "ESTR") & data_rco[pa.NC_PA_CONTRACT_TYPE].isin(
                    ["A-PR-INTBQ", "P-EMP-INTBQ"]) & (data_rco[pa.NC_PA_MARCHE] == "VIDE") & (
                                    data_rco[pa.NC_PA_PALIER] == "-")
                data_rco = data_rco[~del_ajust].copy()
            elif self.name_product in mod.models_nmd_pn:
                data_rco = data_rco[data_rco[pa.NC_PA_IND03].isin(
                    [x for x in list_indics if x in [pa.NC_PA_DEM_RCO, pa.NC_PA_DMN_RCO]])].copy()
                data_rco[pa.NC_PA_IND03] = data_rco[pa.NC_PA_IND03].replace(pa.NC_PA_DEM_RCO, pa.NC_PA_LEM).replace(
                    pa.NC_PA_DMN_RCO, pa.NC_PA_LMN)
            else:
                data_rco = data_rco[data_rco[pa.NC_PA_IND03].isin(
                    [x for x in list_indics if x not in [pa.NC_PA_DEM_RCO, pa.NC_PA_DMN_RCO]])].copy()

            # tf-tv
            data_rco = self.filter_tf_tv(data_rco)
            data_rco = self.filter_by_contract(name_prod, data_rco)

            data_rco = data_rco[
                [pa.NC_PA_CONTRACT_TYPE, "ETAB", pa.NC_PA_IND03, pa.NC_PA_MARCHE, pa.NC_PA_DEVISE, pa.NC_PA_RATE_CODE]
                + self.all_cols_num].copy()
            data_rco["VERSION"] = "RCO"
            data_rco["CONTRAT_REF"] = "-"
            data_rco["TYPE"] = ""

            data_rco = data_rco[
                ["VERSION", pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_MARCHE, pa.NC_PA_ETAB, pa.NC_PA_DEVISE, pa.NC_PA_RATE_CODE,
                 "CONTRAT_REF", "TYPE",
                 pa.NC_PA_IND03] + self.all_cols_num].copy()

            # Utile pour la MNI
            is_encours = data_rco[pa.NC_PA_IND03].isin(
                [pa.NC_PA_LEM, pa.NC_PA_TEM, "LEF", pa.NC_PA_TEF, pa.NC_PA_DEM_RCO])
            data_rco.loc[:, "M1":"M300"] = data_rco.loc[:, "M1":"M300"].astype(np.float64)
            for i in range(1, 26):
                if i <= 10:
                    data_rco["A%s" % i] = np.zeros(data_rco.shape[0])
                    data_rco.loc[~is_encours, "A%s" % i] = data_rco.loc[~is_encours,
                                                           "M%s" % ((i - 1) * 12 + 1):"M%s" % (i * 12)].sum(axis=1)
                    data_rco.loc[is_encours, "A%s" % i] = data_rco.loc[is_encours,
                                                          "M%s" % ((i - 1) * 12 + 1):"M%s" % (i * 12)].mean(axis=1)
                else:
                    data_rco["A%s" % i] = data_rco.loc[:, "M%s" % (i * 12)]

            output_file_name = "AGREG_" + self.output_file_name

            if is_ag_report:
                data_rco.to_csv(os.path.join(self.output_folder, output_file_name), decimal=",", sep=";",
                                index=False, mode='a', header=False)

            if is_model_report:
                if len(self.data_report) > 0:
                    self.data_report = pd.concat([self.data_report, data_rco])
                else:
                    self.data_report = data_rco.copy()

    def filter_tf_tv(self, data_rco):
        # tf-tv
        if self.name_product not in ["all_ech_pn", "nmd_st", "nmd_pn"]:
            if (self.name_product in self.list_tf):
                data_rco = data_rco[data_rco[pa.NC_PA_RATE_CODE].str.contains("FIXE")].copy()
            else:
                data_rco = data_rco[~data_rco[pa.NC_PA_RATE_CODE].str.contains("FIXE")].copy()

        return data_rco

    def filter_by_contract(self, name_prod, data_rco):
        # PRODUIT
        if (name_prod == "a-crctz"):
            data_rco = data_rco[data_rco[pa.NC_PA_CONTRACT_TYPE].isin(["A-PTZ", "A-PTZ+", "AHB-NS-CR-PTZ"])].copy()

        elif (name_prod in ["p-cat-tf", "p-cat-tv"]):
            data_rco = data_rco[data_rco[pa.NC_PA_CONTRACT_TYPE].str.startswith("P-CAT")].copy()

        elif (name_prod in ["a-crctf", "a-crctv"]):
            data_rco = data_rco[data_rco[pa.NC_PA_CONTRACT_TYPE].isin(["A-PR-PERSO", "AHB-NS-PR-PER"])].copy()

        elif (name_prod in ["a-crif", "a-criv"]):
            perimetre_habitat = ["A-CR-HAB-LIS", "A-CR-HAB-STD", "A-CR-HAB-MOD", "A-CR-HAB-AJU", "A-PR-STARDEN",
                                 "AHB-NS-CR-HAB", "A-CR-HAB-BON", "AHB-NS-CR-HBN", "A-CR-REL-HAB"]
            filtre = (data_rco[pa.NC_PA_CONTRACT_TYPE].isin(perimetre_habitat))
            data_rco = data_rco[filtre].copy()

        elif (name_prod in ["a-creqtv", "a-creq"]):
            data_rco = data_rco[
                data_rco[pa.NC_PA_CONTRACT_TYPE].isin(["A-CR-EQ-STD", "A-CR-EQ-STR", "A-CR-EQ-AIDE", "A-CR-EQ-CPLX",
                                                       "A-CR-EQ-MUL", "AHB-NS-CR-EQ", "AHB-NS-CR-EQA"])].copy()
            data_rco = data_rco[data_rco[pa.NC_PA_CONTRACT_TYPE].isin(["A-CR-EQ-STD",
                                                                       "A-CR-EQ-AIDE",
                                                                       "A-CR-EQ-STR",
                                                                       "A-CR-EQ-CPLX",
                                                                       "A-CR-EQ-MUL",
                                                                       "AHB-NS-CR-EQ",
                                                                       "AHB-NS-CR-EQA"])].copy()

        elif (name_prod in ["a-intbq-tf", "a-intbq-tv"]):
            list_contrats = ["A-PR-INTBQ"]
            data_rco = data_rco[data_rco[pa.NC_PA_CONTRACT_TYPE].isin(list_contrats)].copy()

        elif (name_prod in ["p-intbq-tf", "p-intbq-tv"]):
            list_contrats = ["P-EMP-INTBQ"]
            data_rco = data_rco[data_rco[pa.NC_PA_CONTRACT_TYPE].isin(list_contrats)].copy()

        elif (name_prod == "cap_floor"):
            data_rco = data_rco[
                data_rco[pa.NC_PA_CONTRACT_TYPE].isin(["AHB-CAP", "PHB-CAP", "AHB-FLOOR", "PHB-FLOOR"])].copy()

        elif (name_prod == "p-pel"):
            data_rco = data_rco[data_rco[pa.NC_PA_CONTRACT_TYPE].str.contains("P-PEL")].copy()

        elif (name_prod in ["a-autres-tf", "a-autres-tv"]):
            list_contrats = ["A-CRE-AV", "A-EFT-CRECOM", "A-PR-PATRI", "A-CR-TRESO",
                             "A-CR-BAIL", "A-CR-LBO", "A-PR-CEL", "A-LIGNE-TRES",
                             "A-PR-PEL", "A-PR-LDD", "A-EMP-SUB", "A-PR-PGE1", "A-PR-PGE2", "HB-NS-CR-TR",
                             "A-PR-FDS", "A-PR-OMM", "AHB-SW-A", "AHB-SW-D-A"]
            data_rco = data_rco[data_rco[pa.NC_PA_CONTRACT_TYPE].isin(list_contrats)].copy()

        elif (name_prod in ["p-autres-tf", "p-autres-tv"]):
            list_contrats = ['P-REFI-BEI', 'P-TCN', 'P-REFI-CRH', 'P-EMP-CLI', 'P-REFI-SFH', 'P-PEP-STD',
                             'P-REFI-CDC', 'P-BON', 'P-PEP-PROG', 'P-PEP-RENTE', 'P-EMP-SUB-DI', 'P-PEP-IND',
                             'P-EMP-BILAN', 'P-EMP-SUB-DD', 'P-EMP-OMM', 'PHB-SW-D-P', "PHB-SW-P"]
            filtre = (data_rco[pa.NC_PA_CONTRACT_TYPE].isin(list_contrats))
            data_rco = data_rco[filtre].copy()

        elif (name_prod in ["all_ech_pn"]):
            list_contrats = ['P-REFI-BEI', 'P-TCN', 'P-REFI-CRH', 'P-EMP-CLI', 'P-REFI-SFH', 'P-PEP-STD',
                             'P-REFI-CDC', 'P-BON', 'P-PEP-PROG', 'P-PEP-RENTE', 'P-EMP-SUB-DI', 'P-PEP-IND',
                             'P-EMP-BILAN', 'P-EMP-SUB-DD', 'P-EMP-OMM', "A-CRE-AV", "A-EFT-CRECOM", "A-PR-PATRI",
                             "A-CR-TRESO",
                             "A-CR-BAIL", "A-CR-LBO", "A-PR-CEL", "A-LIGNE-TRES",
                             "A-PR-PEL", "A-PR-LDD", "A-EMP-SUB", "A-PR-PGE1", "A-PR-PGE2", "HB-NS-CR-TR",
                             "A-PR-FDS", "A-PR-OMM", "AHB-CAP", "PHB-CAP", "AHB-FLOOR", "PHB-FLOOR", "P-EMP-INTBQ",
                             "A-PR-INTBQ", "A-CR-EQ-STD", "A-CR-EQ-STR", "A-CR-EQ-AIDE", "A-CR-EQ-CPLX",
                             "A-CR-EQ-MUL", "AHB-NS-CR-EQ", "AHB-NS-CR-EQA",
                             "A-CR-HAB-LIS", "A-CR-HAB-STD", "A-CR-HAB-MOD", "A-CR-HAB-AJU", "A-PR-STARDEN",
                             "AHB-NS-CR-HAB", "A-CR-HAB-BON", "AHB-NS-CR-HBN", "A-CR-REL-HAB", "A-PR-PERSO",
                             "AHB-NS-PR-PER", "P-CAT", "A-PTZ", "A-PTZ+", "AHB-NS-CR-PTZ", "AHB-SW-A", "AHB-SW-D-A",
                             "A-OBLIG-1", "A-OBLIG-2", "A-OBLIG-3", "A-OBLIG-4", "A-OBLIG-5", "A-OBLIG-6", "PHB-SW-D-P",
                             "PHB-SW-P", "P-CAT-CORP", "P-CAT-PELP",
                             "P-CAT-PROG", "P-CAT-STD", "P-MISE-PENS", "P-PENS-BILAN", "P-TIT-OBLIG", "P-TCN-FI",
                             "A-ACTION-PN"]

            filtre = (data_rco[pa.NC_PA_CONTRACT_TYPE].isin(list_contrats))
            data_rco = data_rco[filtre].copy()

        elif (name_prod in ["nmd_st", "nmd_pn"]):
            list_contrats = ['P-CEL', 'P-ICNE', 'P-LIV-ORD', 'A-ICNE', 'P-LIV-DD-BP', 'P-CPT-CRD', 'A-CTX', 'A-CPT-DEB',
                             'P-PEP-ECHU', 'A-CR-IMM-PRO', 'A-PROV', 'P-PROV-RISCH', 'A-IMPREPECH', 'P-DAV-CORP',
                             'A-FGD', 'P-REGUL', 'P-DAV-PART', 'A-ESC-VAL', 'A-T-PART', 'A-PROV-PTF', 'A-PROVPAR',
                             'P-LIV-A-BP', 'A-CAISSE', 'A-OPC-ACTION', 'A-IMMO', 'P-FRBG', 'P-LIV-EP', 'P-RESERVES',
                             'A-REGUL', 'P-PROV-EL', 'P-CC-ENT-DEB', 'P-LIV-AUT', 'P-CC-CLI-CRD', 'P-LIV-JEUNE',
                             'A-FCPR',
                             'A-LEP-CENTR', 'A-CC-CLI-DEB', 'P-LIV-CONDI', 'A-CR-REV', 'P-CASDEN2', 'P-RAN',
                             'A-CC-ENT-CRD',
                             'A-CC-CAS', 'P-RSLT-FORM', 'P-CAPITAL', 'A-PR-RO', 'A-ACC-PART', 'A-APPEL-MRG',
                             'A-LIVA-C-BP',
                             'P-APPEL-MRG', 'A-CASDEN', 'A-CASDEN2', "A-LIVA-C-CEP", "A-OPC-ALTER", "A-OPC-DIV",
                             'A-OPC-INC', 'A-OPC-MON', 'P-BON-ECHU', "P-CASDEN", "P-LIV-A-CEP", "P-LIV-DD-CEP",
                             "P-LIV-PREF", "P-RSLT-AFF", "A-CPT-BC", "A-RES-OBLIG", "P-HYBRIDES", "P-LIV-ASSO",
                             "P-LIV-ER-10", "P-LIV-ER-11", "P-LIV-ER-12", "P-LIV-ER-13", "P-LIV-ER-14", "P-LIV-ER-15",
                             "P-LIV-ER-5", "P-LIV-ER-6", "P-LIV-ER-7", "P-LIV-ER-8", "P-LIV-ER-9", "P-LIV-PROMO",
                             "P-LIV-SOC", "P-PEP-RELAIS", "P-PROV-CGR", "P-SLE", "A-DIFF-CB", "A-LIGNE-INT",
                             'P-RSLT-FORM1', 'P-RSLT-FORM2', 'P-RSLT-FORM3', 'P-RSLT-FORM4', 'P-RSLT-FORM5',
                             'P-RSLT-FORM6', 'P-DAV-GCNOP', 'P-DAV-GCOP']

            filtre_pel = data_rco[pa.NC_PA_CONTRACT_TYPE].str.contains("P-PEL")
            filtre = (data_rco[pa.NC_PA_CONTRACT_TYPE].isin(list_contrats)) | filtre_pel
            data_rco = data_rco[filtre].copy()

        elif (name_prod in ["p-swap-tv", "p-swap-tf"]):
            data_rco = data_rco[
                data_rco[pa.NC_PA_CONTRACT_TYPE].isin(
                    ["PHB-SW-SIMPLE", "PHB-SW-ASS", "PHB-SW-STRUCT", "PHB-SW-DEV"])].copy()

        elif (name_prod in ["a-swap-tv", "a-swap-tf"]):
            data_rco = data_rco[
                data_rco[pa.NC_PA_CONTRACT_TYPE].isin(
                    ["AHB-SW-SIMPLE", "AHB-SW-ASS", "AHB-SW-STRUCT", "AHB-SW-DEV"])].copy()

        elif (name_prod in ["a-security-tf", "a-security-tv"]):
            data_rco = data_rco[
                data_rco[pa.NC_PA_CONTRACT_TYPE].isin(['A-OBLIG-2', 'A-OBLIG-1', 'A-OBLIG-3', 'A-OBLIG-4',
                                                       'A-T-SUB-DD', 'A-OBLIG-6', 'A-OBLIG-5', 'A-T-SUB-DI',
                                                       'A-ACTION'])].copy()

        elif (name_prod in ["p-security-tf", "p-security-tv"]):
            data_rco = data_rco[data_rco[pa.NC_PA_CONTRACT_TYPE].isin(['P-TIT-OBLIG', "P-TCN-FI", "P-EMP-OBL"])].copy()

        elif (name_prod in ["a-change-tv", "a-change-tf"]):
            data_rco = data_rco[data_rco[pa.NC_PA_CONTRACT_TYPE].isin(['AHB-CHANGE-SW', "AHB-CHANGE"])].copy()

        elif (name_prod in ["p-change-tv", "p-change-tf"]):
            data_rco = data_rco[data_rco[pa.NC_PA_CONTRACT_TYPE].isin(['PHB-CHANGE-SW', "PHB-CHANGE"])].copy()

        else:
            exit("Produit non défini")

        return data_rco
