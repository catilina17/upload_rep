import pandas as pd
from utils import excel_utils as ex
import numpy as np
from calculateur.models.utils import utils as ut
from dateutil.relativedelta import relativedelta
from calculateur.models.rates.class_nmd_ftp_rates import NMD_FTP_Rate_Calculator
import datetime
import string
import re
import logging

logger = logging.getLogger(__name__)


class Data_NMD_Model_Params():

    def __init__(self, model_wb, dar_usr, cls_fields, cls_hz_params):
        self.model_wb = model_wb
        self.dar_usr = dar_usr
        self.load_range_names()
        self.cls_fields = cls_fields
        self.cls_hz_params = cls_hz_params

    def load_flow_model_cols_names(self):
        self.NC_FLOW_DEF_ID = "FLOW_DEF_ID"
        self.NC_DATE_TYPE = "DATE_TYPE"
        self.NC_YEAR = "YEAR"
        self.NC_MONTH = "MONTH"
        self.NC_DAY = "DAY"
        self.NC_FLOW_DATE = "FLOW_DATE"
        self.FLOW_VAL = "PERCENTAGE"

    def load_flow_formula_cols_names(self):
        self.NC_NAME_VAR = "NAME VARIABLE"
        self.NC_FORMULA = "FORMULA"

    def load_flow_definition_cols_names(self):
        self.NC_FLOW_CODE = "FLOW_CODE"
        self.NC_FLOW_DEF_ID = "FLOW_DEF_ID"

    def load_flow_model_histo_cols_names(self):
        self.NC_FLOW_CODE = "FLOW_CODE"
        self.NC_MONTH = "MONTH"
        self.NC_ETAB = "COMPANY_CODE"
        self.FLOW_VAL = "PERCENTAGE"

    def load_range_names(self):
        self.NR_FLOW_MAPPING = "_FLOW_MAP"
        self.NR_FLOW_MAPPING_FORMULA = "_FLOW_MAP_FORMULA"
        self.NR_FLOW_MAPPING_HISTO = "_FLOW_MAP_HISTO"
        self.NR_FLOW_MODEL = "_FLOW_MODEL"
        self.NR_FLOW_MODEL_HISTO = "_FLOW_MODEL_HISTO"
        self.NR_FLOW_FORMULAS = "_FLOW_FORMULAS"
        self.NR_FLOW_MODEL_FORMULA = "_FLOW_MODEL_FORMULA"

    def load_model_flows(self):
        self.load_flow_model_cols_names()
        self.flows_model = ex.get_dataframe_from_range(self.model_wb, self.NR_FLOW_MODEL)
        self.flows_model[self.NC_FLOW_DATE] = self.flows_model[self.NC_FLOW_DATE].dt.tz_convert(None)
        self.flows_model[self.NC_FLOW_DEF_ID] = self.flows_model[self.NC_FLOW_DEF_ID].astype(int)

        self.flows_model_formula = ex.get_dataframe_from_range(self.model_wb, self.NR_FLOW_MODEL_FORMULA)
        self.flows_model_formula[self.NC_FLOW_DEF_ID] = self.flows_model_formula[self.NC_FLOW_DEF_ID].astype(int)

        self.flows_model_histo = ex.get_dataframe_from_range(self.model_wb, self.NR_FLOW_MODEL_HISTO)
        self.flows_model_histo[self.NC_FLOW_DEF_ID] = self.flows_model_histo[self.NC_FLOW_DEF_ID].astype(int)

        self.flows_definitions[self.NC_FLOW_DEF_ID] = self.flows_definitions[self.NC_FLOW_DEF_ID].astype(int)
        self.flows_definitions_formula[self.NC_FLOW_DEF_ID] = self.flows_definitions_formula[
            self.NC_FLOW_DEF_ID].astype(int)
        self.flows_definitions_histo[self.NC_FLOW_DEF_ID] = self.flows_definitions_histo[self.NC_FLOW_DEF_ID].astype(
            int)

        self.flows_model = self.flows_model.set_index(self.NC_FLOW_DEF_ID)
        self.flows_model_formula = self.flows_model_formula.set_index(self.NC_FLOW_DEF_ID)
        self.flows_model_histo = self.flows_model_histo.set_index(self.NC_FLOW_DEF_ID)

        self.flows_definitions = self.flows_definitions.set_index(self.NC_FLOW_CODE)
        self.flows_definitions_formula = self.flows_definitions_formula.set_index(self.NC_FLOW_CODE)
        self.flows_definitions_histo = self.flows_definitions_histo.set_index(self.NC_FLOW_CODE)

    def load_model_flows_formulas(self):
        self.load_flow_formula_cols_names()
        self.flows_formula = ex.get_dataframe_from_range(self.model_wb, self.NR_FLOW_FORMULAS)

    def load_flow_definitions(self):
        self.load_flow_definition_cols_names()
        self.flows_definitions = ex.get_dataframe_from_range(self.model_wb, self.NR_FLOW_MAPPING)
        self.flows_definitions_formula = ex.get_dataframe_from_range(self.model_wb, self.NR_FLOW_MAPPING_FORMULA)
        self.flows_definitions_histo = ex.get_dataframe_from_range(self.model_wb, self.NR_FLOW_MAPPING_HISTO)

    def recursive_formula(self, levels_dic, formula, flows_model, var_name, level=0):
        dependent_vars = list(set(re.findall('\[+(.*?)\]', formula)))
        if len(dependent_vars) > 0:
            for var in dependent_vars:
                formula_lev = self.flows_formula[self.flows_formula[self.NC_NAME_VAR] == var][self.NC_FORMULA]
                if len(formula_lev) > 0:
                    levels_dic = self.recursive_formula(levels_dic, formula_lev.iloc[0], flows_model, var,
                                                        level=level + 1)
        if level in levels_dic:
            levels_dic[level].append(var_name)
        else:
            levels_dic[level] = [var_name]

        return levels_dic

    def get_python_formula(self, formula, dependent_vars, flow_data_name):
        s = len(dependent_vars)
        name_vars = list(string.ascii_lowercase)
        for dep_var, i in zip(dependent_vars, range(0, s)):
            formula = formula.replace("[" + dep_var + "]", "[" + name_vars[i] + "]")
        formula = formula.replace("[", "").replace("]", "")
        formula = formula.replace("@F_DATE", "datetime.datetime")
        py_formula = "["
        py_formula = py_formula + formula
        py_formula = py_formula + " for " + ",".join(list(string.ascii_lowercase)[:s]) + " in "
        if s > 1:
            py_formula = py_formula + "zip("
        for dep_var, i in zip(dependent_vars, range(0, s)):
            py_formula = py_formula + ("," if i > 0 else "") + '%s["%s"]' % (flow_data_name, dep_var)
        py_formula = py_formula + (")" if s > 1 else "") + "]"
        return py_formula

    def parse_flow_model_vals(self, flows_model_formula):
        for formula_label in flows_model_formula[self.FLOW_VAL].unique():
            formula = self.flows_formula[self.flows_formula[self.NC_NAME_VAR] == formula_label][self.NC_FORMULA].iloc[0]
            levels_dic = {}
            levels_dic = self.recursive_formula(levels_dic, formula, flows_model_formula, formula_label, level=0)
            for i in range(len(levels_dic) - 1, -1, -1):
                for var in levels_dic[i]:
                    if not var in flows_model_formula.columns:
                        level_formula = self.flows_formula[self.flows_formula[self.NC_NAME_VAR]
                                                           == var][self.NC_FORMULA].iloc[0]
                        dependent_vars = list(set(re.findall('\[+(.*?)\]', level_formula)))

                        flows_model_formula[var] = eval(self.get_python_formula(level_formula, dependent_vars,
                                                                                "flows_model_formula"))

        flows_model_formula[self.FLOW_VAL] = flows_model_formula[formula_label]

        flows_model_formula[self.FLOW_VAL] = flows_model_formula[self.FLOW_VAL].astype(np.float64)
        return flows_model_formula

    ###@profile
    def format_model_flows(self, flows_model, t, formula=False):
        qual_vars = ["index"]
        flow_date = np.array(flows_model[self.NC_FLOW_DATE].replace(".", datetime.datetime(1900, 1, 1).date())).astype(
            "datetime64[D]")
        val_date_contrats = np.maximum(np.array(self.cls_hz_params.dar_usr).astype("datetime64[D]"),
                                       np.array(flows_model[self.cls_fields.NC_LDP_VALUE_DATE_REAL]))

        mxt_dates_runoffs = ut.add_days_date(val_date_contrats, flows_model[self.NC_DAY].fillna(0))
        mxt_dates_runoffs = ut.add_months_date(mxt_dates_runoffs, flows_model[self.NC_MONTH].fillna(0))
        flows_model[self.NC_FLOW_DATE] = ut.add_years_date(mxt_dates_runoffs, flows_model[self.NC_YEAR].fillna(0))
        flows_model[self.NC_FLOW_DATE] = np.where(flows_model[self.NC_DATE_TYPE] == "A", flow_date,
                                                  flows_model[self.NC_FLOW_DATE].values)
        if formula:
            flows_model = self.parse_flow_model_vals(flows_model)

        flows_model[self.FLOW_VAL] = flows_model[self.FLOW_VAL] / 100

        flows_model["DATE_END_MONTH"] = pd.to_datetime(flows_model[self.NC_FLOW_DATE]) + pd.offsets.MonthEnd(0)
        flows_model["JOUR TOMBEE"] = pd.to_datetime(flows_model[self.NC_FLOW_DATE]).dt.day
        # flows_model = flows_model[flows_model["DATE_END_MONTH"] > self.cls_hz_params.dar_usr].copy()

        flows_months = flows_model.pivot_table(index=qual_vars, columns=["DATE_END_MONTH"],
                                               values=[self.FLOW_VAL, "JOUR TOMBEE"], aggfunc="sum", fill_value=np.nan,
                                               dropna=False)

        flows_months = flows_months[[x for x in flows_months.columns if x[1] > self.dar_usr]].copy()

        flow_days = flows_months[[x for x in flows_months.columns if "JOUR" in x[0]]]
        flow_days.columns = [y for x, y in flow_days.columns]
        flow_days = flow_days.fillna(1)

        flow_months = flows_months[[x for x in flows_months.columns if not "JOUR" in x[0]]]
        flow_months = flow_months.fillna(0)
        flow_months.columns = [y for x, y in flow_months.columns]

        flow_months = self.add_missing_num_cols(flow_months, qual_vars, t, fill_na=0.0)
        flow_days = self.add_missing_num_cols(flow_days, qual_vars, t, fill_na=1)

        flow_days[flow_days.isnull().all(1)] = np.ones(flow_days[flow_days[self.num_cols].isnull().all(1)].shape)

        return flow_months, flow_days

    ##########@profile
    def format_model_flows_histo(self, flows_model, t):
        qual_vars = ["index"]
        flows_model[self.FLOW_VAL] = flows_model[self.FLOW_VAL] / 100
        val_date_contrats = np.array(flows_model[self.cls_fields.NC_LDP_VALUE_DATE_REAL])

        mxt_dates_runoffs = ut.add_days_date(val_date_contrats, flows_model[self.NC_MONTH].fillna(0) * 0 + 1)
        flows_model["DATE"] = ut.add_months_date(mxt_dates_runoffs, flows_model[self.NC_MONTH].fillna(0))

        flows_model["DATE_END_MONTH"] = pd.to_datetime(flows_model["DATE"]) + pd.offsets.MonthEnd(0)
        flows_model["JOUR TOMBEE"] = pd.to_datetime(flows_model["DATE"]).dt.day

        flows_months = flows_model.pivot_table(index=qual_vars, columns=["DATE_END_MONTH"],
                                               values=[self.FLOW_VAL, "JOUR TOMBEE"], aggfunc="sum", fill_value=0,
                                               dropna=False)

        flows_months = flows_months[[x for x in flows_months.columns if x[1] > self.dar_usr]].copy()

        flow_days = flows_months[[x for x in flows_months.columns if "JOUR" in x[0]]]
        flow_days.columns = [y for x, y in flow_days.columns]

        flow_months = flows_months[[x for x in flows_months.columns if not "JOUR" in x[0]]]
        flow_months.columns = [y for x, y in flow_months.columns]

        flow_months = self.add_missing_num_cols(flow_months, qual_vars, t, fill_na=0.0)
        flow_days = self.add_missing_num_cols(flow_days, qual_vars, t, fill_na=1)

        flow_days = flow_days.ffill(axis=1).bfill(axis=1)

        return flow_months, flow_days

    def load_models_params(self):
        self.load_flow_definitions()
        self.load_model_flows()
        self.load_model_flows_formulas()

    def add_missing_num_cols(self, data, qual_vars, t, fill_na=np.nan):
        num_cols = [x for x in data.columns if x not in qual_vars]
        necessary_cols = [pd.Timestamp((self.dar_usr + relativedelta(months=x)
                                        + relativedelta(day=31)).date()) for x in
                          range(1, t + 1)]
        missing_cols = [x for x in necessary_cols if x not in num_cols]
        if len(missing_cols) > 0 and len(num_cols) > 0:
            data_num = pd.concat([data[num_cols].copy(), pd.DataFrame(fill_na, columns=missing_cols, index=data.index)],
                                 axis=1)
        elif len(num_cols) > 0:
            data_num = data[num_cols].copy()
        else:
            data_num = pd.DataFrame(fill_na, columns=missing_cols, index=data.index)
        data_num = data_num.reindex(sorted(data_num.columns), axis=1)
        self.num_cols = ["M" + str(i) for i in range(1, len(data_num.columns.tolist()) + 1)]
        data_num.columns = self.num_cols
        return data_num

    ###@profile
    def get_flows(self, cls_proj, data, t):
        self.monthly_flow_all, self.monthly_flow, self.day_tombee_all, self.day_tombee = self.get_std_flows(data, t, self.cls_fields.NC_LDP_FLOW_MODEL_NMD)
        self.monthly_flow_gptx_all, self.monthly_flow_gptx, self.day_tombee_gptx_all, self.day_tombee_gptx =  self.get_gptx_flows(data, t)

        if cls_proj.calculate_tci:
            is_tci_diff = (data[self.cls_fields.NC_LDP_FLOW_MODEL_NMD] != data[self.cls_fields.NC_LDP_FLOW_MODEL_NMD_TCI]).values
            self.monthly_flow_tci_all = self.monthly_flow_all.copy()
            self.monthly_flow_tci = self.monthly_flow.copy()
            self.day_tombee_tci = self.day_tombee.copy()
            self.day_tombee_tci_all = self.day_tombee_all.copy()

            (self.monthly_flow_tci_all[is_tci_diff], self.monthly_flow_tci[is_tci_diff],self.day_tombee_tci_all[is_tci_diff],
             self.day_tombee_tci[is_tci_diff])\
                = self.get_std_flows(data[is_tci_diff].copy(), t, self.cls_fields.NC_LDP_FLOW_MODEL_NMD_TCI,
                                     max_flow=self.monthly_flow_all.shape[1])

    ##@profile
    def get_std_flows(self, data, t, flow_model_col, max_flow=None):
        monthly_flow_data_formula_all = []
        monthly_flow_data_histo_all = []
        data[self.cls_fields.NC_LDP_VALUE_DATE_REAL]\
            = np.array(data[self.cls_fields.NC_LDP_VALUE_DATE_REAL]).astype("datetime64[D]")
        data["ETAB_FLOW"] = data[self.cls_fields.NC_LDP_ETAB] + "_" + data[flow_model_col]

        key_join = ["ETAB_FLOW", self.cls_fields.NC_LDP_VALUE_DATE_REAL, flow_model_col]

        data_flow = data[key_join].drop_duplicates()

        data_flow = data_flow.reset_index(drop=True).reset_index().copy()

        monthly_flow_data = data_flow.join(self.flows_definitions, on=["ETAB_FLOW"]).drop(["ETAB_FLOW"], axis=1)

        no_model1 = np.isnan(monthly_flow_data[self.NC_FLOW_DEF_ID])
        if no_model1.any():
            monthly_flow_data2 = (data_flow[no_model1][[flow_model_col, "index", self.cls_fields.NC_LDP_VALUE_DATE_REAL]]
                .join(self.flows_definitions, on=[flow_model_col])).drop([flow_model_col], axis=1)
            monthly_flow_data.loc[no_model1, self.NC_FLOW_DEF_ID] = monthly_flow_data2[self.NC_FLOW_DEF_ID]

        monthly_flow_data_all = monthly_flow_data.join(self.flows_model, on=self.NC_FLOW_DEF_ID)

        no_model2 = np.isnan(monthly_flow_data[self.NC_FLOW_DEF_ID])
        if no_model2.any():
            monthly_flow_data_formula = (data_flow[no_model2][[flow_model_col, "index", self.cls_fields.NC_LDP_VALUE_DATE_REAL]]
                .join(self.flows_definitions_formula, on=[flow_model_col]).drop([flow_model_col], axis=1))
            formula_model = monthly_flow_data_formula[self.NC_FLOW_DEF_ID].notnull()
            if formula_model.any():
                monthly_flow_data_formula_all = monthly_flow_data_formula[formula_model].join(self.flows_model_formula,on=self.NC_FLOW_DEF_ID)
                monthly_flow_data.loc[no_model2 & formula_model, self.NC_FLOW_DEF_ID] = monthly_flow_data_formula[formula_model][self.NC_FLOW_DEF_ID].values

        no_model3 = np.isnan(monthly_flow_data[self.NC_FLOW_DEF_ID])
        if no_model3.any():
            monthly_flow_data_histo = (data_flow[no_model3][[flow_model_col, "index", self.cls_fields.NC_LDP_VALUE_DATE_REAL]]
                .join(self.flows_definitions_formula, on=[flow_model_col]).drop([flow_model_col], axis=1))

            histo_model = monthly_flow_data_histo[self.NC_FLOW_DEF_ID].notnull()
            if histo_model.any():
                monthly_flow_data_histo_all = monthly_flow_data_histo[histo_model].join(self.flows_model_formula, on=self.NC_FLOW_DEF_ID)
                monthly_flow_data.loc[no_model3 & histo_model, self.NC_FLOW_DEF_ID] = monthly_flow_data_histo[histo_model][self.NC_FLOW_DEF_ID].values

        if monthly_flow_data[self.NC_FLOW_DEF_ID].isnull().any():
            manquants = monthly_flow_data[monthly_flow_data[self.NC_FLOW_DEF_ID].isnull()][flow_model_col].unique()
            logger.error("Certains modèles NMDs sont manquants %s" % manquants)

        if max_flow is None:
            flow_max_proj = self.get_flow_max_projection(monthly_flow_data_all, data_flow, t, formula_data=monthly_flow_data_formula_all,
                                                        histo_data = monthly_flow_data_histo_all)
        else:
            flow_max_proj = max_flow

        monthly_flow_all, day_tombee = self.format_model_flows(monthly_flow_data_all, flow_max_proj)
        monthly_flow_all, day_tombee = np.array(monthly_flow_all), np.array(day_tombee)

        if no_model2.any():
            if formula_model.any():
                monthly_flow_formula, day_tombee_formula = self.format_model_flows(monthly_flow_data_formula_all,
                                                                                   flow_max_proj, formula=True)
                monthly_flow_all[no_model2 & formula_model] = np.array(monthly_flow_formula)
                day_tombee[no_model2 & formula_model] = np.array(day_tombee_formula)

        if no_model3.any():
            if histo_model.any():
                monthly_flow_histo, day_tombee_histo= self.format_model_flows_histo(monthly_flow_data_histo_all,
                                                                                         flow_max_proj)
                monthly_flow_all[no_model3 & histo_model] = np.array(monthly_flow_histo)
                day_tombee[no_model3 & histo_model] = histo_model.array(day_tombee_histo)

        index_flow = data.join(data_flow.set_index(key_join), on=key_join)["index"].values
        monthly_flow_all = monthly_flow_all[index_flow]
        day_tombee = day_tombee[index_flow]

        return monthly_flow_all, monthly_flow_all[:, :t], day_tombee, day_tombee[:, :t]

    def get_flow_max_projection(self, flows_model, data_flow, t, formula_data=[],  histo_data=[]):
        mois_dep = (data_flow[self.cls_fields.NC_LDP_VALUE_DATE_REAL].values.astype("datetime64[M]") -
                    np.array(self.cls_hz_params.dar_usr).astype("datetime64[M]")) / np.timedelta64(1, 'M')
        max_val_date = int(max(0, mois_dep.max()))
        flows_model["NB_MONTHS"] = flows_model[self.NC_MONTH] + flows_model[self.NC_YEAR] * 12 + 1
        if len(formula_data) > 0:
            formula_data["NB_MONTHS"] = formula_data[self.NC_MONTH] + formula_data[self.NC_YEAR] * 12 + 1
            max_formula = formula_data["NB_MONTHS"].max()
        else:
            max_formula = 0
        if len(histo_data) > 0:
            max_histo = histo_data[self.NC_MONTH].max() + 1
        else:
            max_histo = 0
        flow_max_proj = int(max(max(max_histo, flows_model["NB_MONTHS"].max(), max_formula) + max_val_date, t))
        return flow_max_proj

    ###@profile
    def get_gptx_flows(self, data, t):
        monthly_flow_data_formula_all = []
        monthly_flow_data_histo_all = []
        n = data.shape[0]
        data["ETAB_FLOW"] = data[self.cls_fields.NC_LDP_ETAB] + "_" + data[self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX]

        key_join = ["ETAB_FLOW", self.cls_fields.NC_LDP_VALUE_DATE_REAL, self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX]
        data_flow = data[key_join].drop_duplicates()

        data_flow = data_flow.reset_index(drop=True).reset_index().copy()

        monthly_flow_data = data_flow.join(self.flows_definitions, on=["ETAB_FLOW"]).drop(["ETAB_FLOW"], axis=1)

        no_model1 = np.isnan(monthly_flow_data[self.NC_FLOW_DEF_ID])

        if no_model1.any():
            monthly_flow_data2 = (data_flow[no_model1][[self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX, "index", self.cls_fields.NC_LDP_VALUE_DATE_REAL]]
                .join(self.flows_definitions, on=[self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX])).drop([self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX], axis=1)

            monthly_flow_data.loc[no_model1, self.NC_FLOW_DEF_ID] = monthly_flow_data2[self.NC_FLOW_DEF_ID]

        monthly_flow_data_all = monthly_flow_data.join(self.flows_model, on=self.NC_FLOW_DEF_ID)

        no_model2 = np.isnan(monthly_flow_data[self.NC_FLOW_DEF_ID])
        if no_model2.any():
            monthly_flow_data_formula = (data_flow[no_model2][[self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX, "index",
                                                         self.cls_fields.NC_LDP_VALUE_DATE_REAL]].join(self.flows_definitions_formula,
                                               on=[self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX]).drop([self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX], axis=1))
            formula_model = monthly_flow_data_formula[self.NC_FLOW_DEF_ID].notnull()
            if formula_model.any():
                monthly_flow_data_formula_all = monthly_flow_data_formula[formula_model].join(self.flows_model_formula,
                                                                                              on=self.NC_FLOW_DEF_ID)
                monthly_flow_data.loc[no_model2 & formula_model, self.NC_FLOW_DEF_ID] = \
                monthly_flow_data_formula[formula_model][self.NC_FLOW_DEF_ID].values

        no_model3 = np.isnan(monthly_flow_data[self.NC_FLOW_DEF_ID])
        if no_model3.any():
            monthly_flow_data_histo = (data_flow[no_model3][[self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX, "index", self.cls_fields.NC_LDP_VALUE_DATE_REAL]]
                                       .join(self.flows_definitions_histo, on=[self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX]).drop([self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX], axis=1))
            histo_model = monthly_flow_data_histo[self.NC_FLOW_DEF_ID].notnull()
            if histo_model.any():
                monthly_flow_data_histo_all = monthly_flow_data_histo[histo_model].join(self.flows_model_formula,
                                                                                        on=self.NC_FLOW_DEF_ID)
                monthly_flow_data.loc[no_model3 & histo_model, self.NC_FLOW_DEF_ID] = monthly_flow_data_histo[histo_model][
                    self.NC_FLOW_DEF_ID].values

        filter_null = ~(monthly_flow_data[self.NC_FLOW_DEF_ID].notnull() |
                        (monthly_flow_data[self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX] == ""))
        if filter_null.any():
            manquants = monthly_flow_data[filter_null][self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX].unique()
            logger.error("Certains modèles NMDs de GAP de TAUX sont manquants %s" % manquants)

        flow_max_proj = self.get_flow_max_projection(monthly_flow_data_all, data_flow, t,
                                                     formula_data=monthly_flow_data_formula_all,
                                                     histo_data = monthly_flow_data_histo_all)

        monthly_flow_all, day_tombee = self.format_model_flows(monthly_flow_data_all, flow_max_proj)
        monthly_flow_all, day_tombee = np.array(monthly_flow_all), np.array(day_tombee)

        if len(monthly_flow_all) == 0:
            monthly_flow_all = np.full((n, flow_max_proj), np.nan)
            day_tombee = np.full((n, flow_max_proj), np.nan)

        if no_model2.any():
            if formula_model.any():
                monthly_flow_formula, day_tombee_formula = self.format_model_flows(monthly_flow_data_formula_all,
                                                                                   flow_max_proj,
                                                                                   formula=True)
                monthly_flow_all[no_model2 & formula_model] = np.array(monthly_flow_formula)
                day_tombee[no_model2 & formula_model] = np.array(day_tombee_formula)

        if no_model3.any():
            if histo_model.any():
                monthly_flow_formula, day_tombee_formula = self.format_model_flows_histo(monthly_flow_data_histo_all,
                                                                                         flow_max_proj)
                monthly_flow_all[no_model3 & histo_model] = np.array(monthly_flow_formula)
                day_tombee[no_model3 & histo_model] = histo_model.array(day_tombee_formula)

        index_flow = data.join(data_flow.set_index(key_join), on=key_join)["index"].values
        monthly_flow_all = monthly_flow_all[index_flow]
        day_tombee = day_tombee[index_flow]

        return monthly_flow_all, monthly_flow_all[:, :t], day_tombee, day_tombee[:, :t]

# ######@profile
    def get_nmd_maturity(self, cls_proj, monthly_flow):
        t = cls_proj.min_proj
        n = monthly_flow.shape[0]
        maturity = t - ut.first_nonzero(monthly_flow[:, ::-1], axis=1, val=0, invalid_val=-1)
        maturity = (np.where(np.round(monthly_flow.sum(axis=1), 6) == 1, maturity, t))
        maturity = np.minimum(cls_proj.cls_cal.date_fin.shape[1] -1 , maturity)
        mat_date = cls_proj.cls_cal.date_fin[:, maturity].reshape(n).astype("datetime64[D]")
        mat_date = ut.add_months_date(mat_date, 1)
        mat_date_date = pd.to_datetime(mat_date)
        cls_proj.data_ldp[self.cls_fields.NC_LDP_MATUR_DATE + "_REAL"] = mat_date_date
        cls_proj.data_ldp[self.cls_fields.NC_LDP_MATUR_DATE] = mat_date_date.year * 12 + mat_date_date.month


    def calculate_tci_mat_by_flow_model(self, cls_proj):
        dic_tx_swap = cls_proj.cls_data_rate.dic_tx_swap.copy()
        data = cls_proj.data_ldp.reset_index(drop=True).copy()

        data = data[data[self.cls_fields.NC_LDP_TCI_METHOD] == "MATURITY"].copy()
        cols_index = [self.cls_fields.NC_LDP_FLOW_MODEL_NMD_TCI, self.cls_fields.NC_LDP_TCI_FIXED_RATE_CODE,
                      self.cls_fields.NC_LDP_VALUE_DATE, self.cls_fields.NC_LDP_VALUE_DATE_REAL]
        data[self.cls_fields.NC_LDP_VALUE_DATE] = np.maximum(0, data[self.cls_fields.NC_LDP_VALUE_DATE] - self.cls_hz_params.dar_mois)

        data_tci = data[cols_index].drop_duplicates()
        monthly_flow_tci = self.monthly_flow_tci_all[data_tci.index]
        n = monthly_flow_tci.shape[0]
        t = monthly_flow_tci.shape[1]
        flow_dates = cls_proj.cls_cal.date_fin[:, :t] + (self.day_tombee_tci_all[data_tci.index] - 1).astype("timedelta64[D]")
        nb_days_since_emission = (flow_dates
                                  - data_tci[self.cls_fields.NC_LDP_VALUE_DATE_REAL].values.astype('datetime64[D]').reshape(n, 1))
        courbe_vect = data_tci[self.cls_fields.NC_LDP_TCI_FIXED_RATE_CODE].copy()
        index_fwd = data_tci[self.cls_fields.NC_LDP_VALUE_DATE].values

        current_month = np.arange(1, t + 1).reshape(1, t)

        tx_tci_maturity, tx_tci_strate\
            = NMD_FTP_Rate_Calculator.get_batched_tci_maturity_coeff(monthly_flow_tci, nb_days_since_emission,
                                                                     courbe_vect.copy(), index_fwd, dic_tx_swap, current_month,
                                                                     self.cls_hz_params.max_projection, n, t)

        self.tx_tci_maturity = pd.concat([data_tci, pd.DataFrame(tx_tci_maturity,index=data_tci.index)], axis=1).set_index(cols_index)

        self.tx_tci_strate = pd.concat([data_tci, pd.DataFrame(tx_tci_strate,index=data_tci.index)], axis=1).set_index(cols_index)


    def calculate_tci_index(self, cls_proj, cls_format, tx_params):
        is_index_method = (cls_proj.data_ldp[self.cls_fields.NC_LDP_TCI_METHOD] == "INDEX").values

        cols_tci = [self.cls_fields.NC_LDP_TCI_FIXED_RATE_CODE, self.cls_fields.NC_LDP_TCI_FIXED_TENOR_CODE]
        data_curve_tci = cls_proj.data_ldp.loc[is_index_method, cols_tci].copy()
        data_curve_tci = data_curve_tci[cols_tci].drop_duplicates()
        not_null = (data_curve_tci[self.cls_fields.NC_LDP_TCI_FIXED_RATE_CODE].isnull()
                    | data_curve_tci[self.cls_fields.NC_LDP_TCI_FIXED_TENOR_CODE].isnull())
        data_curve_tci = data_curve_tci[~not_null].copy()

        n = data_curve_tci.shape[0]
        calendar_dates = cls_proj.cls_cal.date_deb
        is_tv = np.full(n, True)
        contract_type = pd.DataFrame(np.full((n), ""))
        data_curve_tci["accrual_basis"] = "30/360"
        data_curve_tci = cls_format.generate_freq_curve_tenor(data_curve_tci, self.cls_fields.NC_LDP_TCI_FIXED_TENOR_CODE,
                                                        self.cls_fields.NC_LDP_TCI_FIXED_TENOR_CODE)

        map_accrual_key_cols = {"CURVE_NAME": self.cls_fields.NC_LDP_TCI_FIXED_RATE_CODE,
                                "TENOR": self.cls_fields.NC_TENOR_NUM}
        data_curve_tci = cls_proj.cls_data_rate.get_curve_accruals_col(data_curve_tci, tx_params,
                                                                       map_accrual_key_cols, is_tv)

        tx_tci_index = cls_proj.cls_data_rate.match_data_with_sc_rates(data_curve_tci, tx_params,
                                                                       cls_proj.t_max - 1, cols_tci,
                                                                       raise_error=False, deb_col=0)

        data_curve_tci[cls_proj.cls_data_rate.STANDALONE_INDEX] = "Yield Curve"
        data_curve_tci[cls_proj.cls_data_rate.ACCRUAL_METHOD]\
            = np.where(data_curve_tci[self.cls_fields.NC_LDP_TCI_FIXED_TENOR_CODE] == "1D", "30/360",
                       data_curve_tci[cls_proj.cls_data_rate.ACCRUAL_METHOD])

        tx_tci_index_conv = cls_proj.cls_rate.get_accrual_convention_adjusted_sc_rates(data_curve_tci, tx_tci_index,
                                                                                self.cls_fields.NC_TENOR_NUM, "accrual_basis",
                                                                                contract_type, calendar_dates,
                                                                                n, cls_proj.t_max)

        if np.isnan(tx_tci_index_conv[:, 1]).any():
            list_errs = data_curve_tci[np.isnan(tx_tci_index_conv[:, 1])][cols_tci].drop_duplicates().values.tolist()
            msg_err = "     Certaines courbes du TCI FIXE/LIQ en méthode index sont manquantes dans le rate_input: %s" % list_errs
            logger.warning(msg_err)

        self.tx_tci_index = pd.concat([data_curve_tci[cols_tci], pd.DataFrame(tx_tci_index_conv,index=data_curve_tci.index)], axis=1).set_index(cols_tci)

    def calculate_tci_var(self, cls_proj, cls_format, tx_params):
        is_var_tci = (cls_proj.data_ldp[self.cls_fields.NC_LDP_TCI_VARIABLE_CURVE_CODE].fillna("") != "").values

        cols_tci = [self.cls_fields.NC_LDP_TCI_VARIABLE_CURVE_CODE, self.cls_fields.NC_LDP_TCI_VARIABLE_TENOR_CODE]

        data_curve_tci = cls_proj.data_ldp.loc[is_var_tci, cols_tci].copy()
        data_curve_tci = data_curve_tci[cols_tci].drop_duplicates()
        not_null = (data_curve_tci[self.cls_fields.NC_LDP_TCI_VARIABLE_CURVE_CODE].isnull()
                    | data_curve_tci[self.cls_fields.NC_LDP_TCI_VARIABLE_TENOR_CODE].isnull())
        data_curve_tci = data_curve_tci[~not_null].copy()

        n = data_curve_tci.shape[0]
        calendar_dates = cls_proj.cls_cal.date_deb
        is_tv = np.full(n, True)
        contract_type = pd.DataFrame(np.full((n), ""))
        data_curve_tci["accrual_basis"] = "30/360"
        data_curve_tci = cls_format.generate_freq_curve_tenor(data_curve_tci, self.cls_fields.NC_LDP_TCI_VARIABLE_TENOR_CODE,
                                                        self.cls_fields.NC_LDP_TCI_VARIABLE_TENOR_CODE)

        map_accrual_key_cols = {"CURVE_NAME": self.cls_fields.NC_LDP_TCI_VARIABLE_CURVE_CODE,
                                "TENOR": self.cls_fields.NC_TENOR_NUM}
        data_curve_tci = cls_proj.cls_data_rate.get_curve_accruals_col(data_curve_tci, tx_params,
                                                                       map_accrual_key_cols, is_tv)

        data_curve_tci[cls_proj.cls_data_rate.STANDALONE_INDEX] = "Yield Curve"
        data_curve_tci[cls_proj.cls_data_rate.ACCRUAL_METHOD]\
            = np.where(data_curve_tci[self.cls_fields.NC_LDP_TCI_VARIABLE_TENOR_CODE] == "1D", "30/360",
                       data_curve_tci[cls_proj.cls_data_rate.ACCRUAL_METHOD])

        tx_tci_var = cls_proj.cls_data_rate.match_data_with_sc_rates(data_curve_tci, tx_params,
                                                                       cls_proj.t_max - 1, cols_tci,
                                                                       raise_error=False, deb_col=0)

        tx_tci_var_conv = cls_proj.cls_rate.get_accrual_convention_adjusted_sc_rates(data_curve_tci, tx_tci_var,
                                                                                self.cls_fields.NC_TENOR_NUM, "accrual_basis",
                                                                                contract_type, calendar_dates,
                                                                                n, cls_proj.t_max)

        if np.isnan(tx_tci_var_conv[:, 1]).any():
            list_errs = data_curve_tci[np.isnan(tx_tci_var_conv[:, 1])][cols_tci].drop_duplicates().values.tolist()
            msg_err = "     Certaines courbes du TCI VARIABLE/TX sont manquantes dans le rate_input: %s" % list_errs
            logger.warning(msg_err)

        self.tx_tci_var = pd.concat([data_curve_tci[cols_tci], pd.DataFrame(tx_tci_var_conv,index=data_curve_tci.index)], axis=1).set_index(cols_tci)

