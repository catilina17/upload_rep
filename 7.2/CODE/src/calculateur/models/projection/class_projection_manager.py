import datetime
import numpy as np
from calculateur.models.utils import utils as ut
from calculateur.models.calendar.class_fixing_calendar import Fixing_Calendar_Manager
from calculateur.models.paliers.class_palier_manager import Palier_Manager
from calculateur.models.versements.class_versements_model import Versements_PEL
from calculateur.models.rates.class_rate_manager import Rate_Manager
from calculateur.calc_params.model_params import *

class Projection_Manager():
    """
    Formate les données
    """

    def __init__(self, cls_fields, cls_hz_params, cls_data_rate, cls_cal):
        self.cls_hz_params = cls_hz_params
        self.cls_fields = cls_fields
        self.cls_cal = cls_cal
        self.cls_data_rate = cls_data_rate

    def split_data_for_projection_data(self):
        data_ldp_all = self.data_ldp.rename(columns={'Ï»¿ETAB': self.cls_fields.NC_LDP_ETAB})
        data_ldp_all = data_ldp_all.rename(columns={'Ï»¿BASSIN': self.cls_fields.NC_LDP_BASSIN})
        self.data_ldp = data_ldp_all[self.cls_fields.proj_vars].copy()
        other_vars = [x for x in data_ldp_all.columns if x not in self.cls_fields.proj_vars]
        self.data_optional = data_ldp_all[other_vars].copy()
        ut.clean_df(data_ldp_all)

    def set_min_proj(self, data_ldp):
        self.load_max_shift_fixing(data_ldp)
        t = max(self.len_deblocage, self.cls_hz_params.nb_months_proj)  + self.max_shift
        self.min_proj = min(self.cls_hz_params.max_projection, t)

    def set_projection_dimensions(self, cls_flow_params):
        self.load_dimensions(self.data_ldp)
        self.current_month, self.current_month_max = self.build_current_month()
        if self.name_product in (models_nmd_pn + models_nmd_st):
            cls_flow_params.day_tombee_gptx = cls_flow_params.day_tombee_gptx[:, :self.t]
            cls_flow_params.monthly_flow_gptx = cls_flow_params.monthly_flow_gptx[:, :self.t]
            cls_flow_params.day_tombee = cls_flow_params.day_tombee[:, :self.t]
            cls_flow_params.monthly_flow = cls_flow_params.monthly_flow[:, :self.t]

    def load_dimensions(self, data_ldp):
        dar_mois = self.cls_hz_params.dar_mois
        n = data_ldp.shape[0]
        current_proj_max = np.max(data_ldp[self.cls_fields.NC_LDP_MATUR_DATE].values) - dar_mois + 2
        current_proj_max = max(self.max_mat_non_perf, current_proj_max)
        t = min(self.cls_hz_params.nb_months_proj, current_proj_max)  + self.max_shift
        t = max(self.len_deblocage, t)
        t = min(self.cls_hz_params.max_projection, t)
        self.n = n
        self.t = t
        self.t_max = max(self.cls_hz_params.max_projection, self.t)

    def build_current_month(self):
        current_month = np.arange(1, self.t + 1).reshape(1, self.t)
        current_month = np.vstack([current_month] * self.n)
        current_month_max = np.repeat(np.arange(1, self.t_max + 1).reshape(1, self.t_max), self.n, axis=0)
        return current_month, current_month_max

    def format_proj_params(self, cls_format):
        data_ldp = cls_format.generate_amortissement_profil(self.data_ldp, self.cls_fields.NC_LDP_TYPE_AMOR)
        if self.name_product not in (models_nmd_pn + models_nmd_st):
            data_ldp = cls_format.get_supsension_or_interets_capitalization_var(data_ldp,
                                                                                self.cls_fields.NC_LDP_FREQ_INT,
                                                                                self.cls_hz_params.dar_mois,
                                                                                capi_rate_col=self.cls_fields.NC_LDP_CAPITALIZATION_RATE,
                                                                                capi_freq_col=self.cls_fields.NC_LDP_CAPITALIZATION_PERIOD)
        else:
            data_ldp = cls_format.get_capitalization_var(data_ldp, self.cls_fields.NC_LDP_FREQ_INT,
                                                         self.cls_fields.NC_LDP_CAPITALIZATION_RATE,
                                                         self.cls_fields.NC_LDP_CAPITALIZATION_PERIOD)

        data_ldp = cls_format.generate_freq_int(data_ldp, self.cls_fields.NC_LDP_FREQ_INT)
        data_ldp = cls_format.generate_freq_amor(data_ldp, self.cls_fields.NC_LDP_FREQ_AMOR,
                                                 self.cls_fields.NC_PAL_FREQ_INT)
        data_ldp = cls_format.generate_periode_cap(data_ldp, self.cls_fields.NC_LDP_CAPITALIZATION_PERIOD)
        data_ldp = cls_format.generate_freq_fixing_periodicity(data_ldp, self.cls_fields.NC_LDP_FIXING_PERIODICITY,
                                                               self.cls_fields.NC_LDP_FREQ_INT, self.name_product)
        data_ldp = cls_format.generate_freq_curve_tenor(data_ldp, self.cls_fields.NC_LDP_TENOR,
                                                        self.cls_fields.NC_LDP_FIXING_PERIODICITY)

        self.data_ldp = data_ldp

    ######@profile
    def prepare_ldp_data(self, cls_proj, cls_data_ldp, cls_cash_flow, cls_format, cls_model_params, tx_params):
        self.data_ldp = cls_data_ldp.data_ldp.copy()

        self.data_ldp, self.contracts_updated \
            = cls_data_ldp.format_ldp_data(cls_model_params.cls_debloc_params, self.cls_hz_params,
                                           cls_cash_flow, self.name_product, tx_params)
        self.split_data_for_projection_data()
        self.format_proj_params(cls_format)
        cls_cash_flow.get_capital_with_cash_flows(self.data_ldp, cls_format, cls_proj)
        self.data_ldp = cls_cash_flow.cancel_data_without_cash_flows(self.data_ldp)

    def set_dar_options(self):
        if self.cls_hz_params.dar_usr >= datetime.datetime(2023, 9, 30):
            self.make_dar_pel_daily = False
            self.pel_age_params_begin_at_1 = True
        else:
            self.make_dar_pel_daily = True
            self.pel_age_params_begin_at_1 = True

    ################@profile

    def load_internal_dimensions(self, data_ldp, cls_model_params, cls_format):
        self.len_deblocage = cls_model_params.cls_debloc_params.nb_mois_deblocage + 1
        dar_mois = self.cls_hz_params.dar_mois
        if (data_ldp[self.cls_fields.NC_LDP_PERFORMING] == "T").any():
            data_non_perf = data_ldp[data_ldp[self.cls_fields.NC_LDP_PERFORMING] == "T"].copy()
            mat_date = cls_format.convert_to_datetime(data_non_perf, self.cls_fields.NC_LDP_MATUR_DATE)
            mat_date_month = cls_format.convert_to_nb_months(mat_date)[1]
            self.max_mat_non_perf = np.max(mat_date_month) - dar_mois + 2
            self.max_mat_non_perf = self.max_mat_non_perf + cls_model_params.cls_douteux_params.size_ecoul_douteux
        else:
            self.max_mat_non_perf = 0

    def load_max_shift_fixing(self, data_ldp):
        if self.cls_fields.NC_LDP_FIXING_RULE in data_ldp.columns.tolist():
            filter_fixing_rule = data_ldp[self.cls_fields.NC_LDP_FIXING_RULE] == "A"
            if filter_fixing_rule.any():
                self.max_shift = data_ldp[filter_fixing_rule][self.cls_fields.NC_FIXING_PERIODICITY_NUM].fillna(
                    0).astype(int).max()
            else:
                self.max_shift = 0
        else:
            self.max_shift = 0

    def init_projection_classes(self, cls_data_ldp, cls_format, name_product, cls_model_params, calc_mode):
        self.calc_mode = calc_mode
        self.name_product = name_product
        self.OLD_RENEGO = False
        self.set_dar_options()

        self.cls_palier = Palier_Manager(self.cls_fields, cls_format, self.cls_cal, self.cls_hz_params.nb_months_proj)

        self.cls_fixing_cal = Fixing_Calendar_Manager(self.cls_hz_params, self.cls_fields, name_product, self.cls_cal)

        self.cls_rate = Rate_Manager(self.cls_data_rate, self.cls_fields, self.cls_hz_params, self.cls_cal,
                                     self.cls_fixing_cal, self.name_product)

        self.calculate_tci = True if ("TCI" in self.cls_fields.exit_indicators_type
                                      and self.name_product in (models_nmd_st + models_nmd_pn)) else False

        self.load_internal_dimensions(cls_data_ldp.data_ldp, cls_model_params, cls_format)

        if name_product in (models_nmd_st + models_nmd_pn):
            self.versements_model = Versements_PEL(cls_model_params.cls_versement_params)
        else:
            self.versements_model = None

        if name_product in (models_nmd_st + models_nmd_pn):
            contrat = cls_data_ldp.data_ldp[cls_data_ldp.data_ldp[self.cls_fields.NC_LDP_CONTRAT] ==
                                            cls_data_ldp.data_ldp[self.cls_fields.NC_LDP_CONTRAT].iloc[0]].copy()
            self.nb_rm_groups = contrat.groupby([self.cls_fields.NC_LDP_CONTRAT])[
                self.cls_fields.NC_LDP_RM_GROUP].count().max()
            self.has_volatile_part = cls_data_ldp.data_ldp[self.cls_fields.NC_LDP_RM_GROUP].str.upper().str.contains(
                "VOLATILE").any()
            self.has_stable_part = cls_data_ldp.data_ldp[self.cls_fields.NC_LDP_RM_GROUP].str.upper().str.contains(
                "STABLE").any()
