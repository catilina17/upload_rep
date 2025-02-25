import numpy as np

class Data_Fields:
    """
    Classe permettant de gérer les champs d'entrée, de sortie ainsi que les indicateurs de sortie
    """
    def __init__(self, exit_indicators_type =["GPLIQ","GPTX"]):
        self.exit_indicators_type = exit_indicators_type
        self.load_columns()
        self.load_grouped_vars()

    def load_exit_parameters(self):
        self.load_exit_indicators()

    def load_columns(self):
        self.NC_LDP_ECHEANCE_VAL = "echeance_eur".upper()
        self.NC_LDP_CONTRAT = 'contract_reference'.upper()
        self.NC_LDP_MATUR_DATE = 'maturity_date'.upper()
        self.NC_LDP_RATE = "rate_value".upper()
        self.NC_LDP_FTP_RATE = "ftp_rate_value".upper()
        self.NC_LDP_NOMINAL = 'nominal_eur'.upper()
        self.NC_LDP_OUTSTANDING = 'outstanding_eur'.upper()
        self.NC_LDP_FIRST_AMORT_DATE = 'first_amort_date'.upper()
        self.NC_LDP_VALUE_DATE = "value_date".upper()
        self.NC_LDP_TRADE_DATE = "trade_date".upper()
        self.NC_LDP_TYPE_AMOR = "amortizing_type".upper()
        self.NC_LDP_FREQ_INT = "periodicity".upper()
        self.NC_LDP_FREQ_AMOR = "amortizing_periodicity".upper()
        self.NC_LDP_RELEASING_RULE = 'releasing_rule'.upper()
        self.NC_LDP_RELEASING_DATE = 'releasing_date'.upper()
        self.NC_LDP_ACCRUAL_BASIS = 'accrual_basis'.upper()
        self.NC_LDP_CONTRACT_TYPE = 'contract_type'.upper()
        self.NC_LDP_MATUR = 'matur'.upper()
        self.NC_LDP_MARCHE = 'family'.upper()
        self.NC_LDP_GESTION = 'gestion'.upper()
        self.NC_LDP_PALIER = "counterparty_code".upper()
        self.NC_LDP_ETAB = "etab".upper()
        self.NC_LDP_CAPITALIZATION_RATE = "capitalization_rate".upper()
        self.NC_LDP_CAPI_MODE = "releasing_irr_calc_mode".upper()
        self.NC_LDP_CAPITALIZATION_PERIOD = "compound_periodicity".upper()
        self.NC_LDP_CURRENCY = "currency".upper()
        self.NC_LDP_BROKEN_PERIOD = 'broken_period_type'.upper()
        self.NC_LDP_INTERESTS_ACCRUALS = 'accruals'.upper()
        self.NC_LDP_PERFORMING = 'performing_status'.upper()
        self.NC_LDP_CURVE_NAME = 'curve_name'.upper()
        self.NC_LDP_TENOR = 'tenor'.upper()
        self.NC_LDP_MKT_SPREAD = 'mkt_spread'.upper()
        self.NC_LDP_FIXING_NEXT_DATE = 'fixing_next_date'.upper()
        self.NC_LDP_CALC_DAY_CONVENTION = 'calc_day_convention'.upper()
        self.NC_LDP_FIXING_PERIODICITY = 'fixing_periodicity'.upper()
        self.NC_LDP_FIXING_RULE = 'fixing_rule'.upper()
        self.NC_LDP_RATE_CODE = "rate_code".upper()
        self.NC_LDP_BASSIN = "bassin".upper()
        self.NC_LDP_RATE_TYPE = "rate_category".upper()
        self.NC_LDP_CAP_STRIKE = "cap_Strike".upper()
        self.NC_LDP_FLOOR_STRIKE = "floor_Strike".upper()
        self.NC_LDP_MULT_SPREAD = "mult_spread".upper()
        self.NC_LDP_BUY_SELL = "Buy_sell".upper()
        self.NC_LDP_CURRENT_RATE = "Current_rate".upper()
        self.NC_LDP_NB_CONTRACTS = "nb_contracts".upper()
        self.NC_LDP_IS_CAP_FLOOR = "cap_floor".upper()
        self.NC_LDP_DATE_SORTIE_GAP = "irr_position_date".upper()
        self.NC_LDP_DATE_SORTIE_GAP_REAL = "irr_position_date_real".upper()
        self.NC_LDP_FIXING_NB_DAYS = "fixing_nb_days".upper()
        self.NC_LDP_FLOW_MODEL_NMD = "flow_model".upper()
        self.NC_LDP_FLOW_MODEL_NMD_GPTX = "flow_model_gptx".upper()
        self.NC_LDP_RM_GROUP = "rm_group".upper()
        self.NC_LDP_RM_GROUP_PRCT = "rm_group_prct".upper()
        self.NC_LDP_FIRST_COUPON_DATE = "first_coupon_date".upper()
        self.NC_LDP_TARGET_RATE = "target_rate".upper()
        self.NC_LDP_DIM6 = "dim6".upper()
        self.NC_LDP_FTP_FUNDING_SPREAD = "ftp_liq_float_value".upper()
        self.NC_LDP_FLOW_MODEL_NMD_TCI = "flow_model_tci".upper()
        self.NC_LDP_TCI_METHOD = "tci_method".upper()
        self.NC_LDP_TCI_FIXED_RATE_CODE = "tci_fixed_rate_code".upper()
        self.NC_LDP_TCI_FIXED_TENOR_CODE = "tci_fixed_tenor_code".upper()
        self.NC_LDP_TCI_VARIABLE_TENOR_CODE = "tci_var_tenor_code".upper()
        self.NC_LDP_TCI_VARIABLE_CURVE_CODE = "tci_var_curve_code".upper()
        self.NC_LDP_SAVINGS_MODEL = "savings_model".upper()
        self.NC_LDP_REDEMPTION_MODEL = "redemption_model".upper()

        """ FICHIER PAL HAB"""
        self.NC_PAL_CONTRAT = 'base_contract_ref'.upper()
        self.NC_PAL_BEGIN_DATE = 'begin_date'.upper()
        self.NC_PAL_RATE = 'rate_value'.upper()
        self.NC_PAL_VAL_EUR = 'new_value_eur'.upper()
        self.NC_PAL_FREQ_AMOR = 'amortizing_periodicity'.upper()
        self.NC_PAL_FREQ_INT = 'periodicity'.upper()
        self.NC_PAL_TYPE_AMOR = 'amortizing_type'.upper()
        self.NC_PAL_CAPITALIZATION_RATE = 'capitalization_rate'.upper()
        self.NC_PAL_CAPITALIZATION_PERIOD = 'compound_periodicity'.upper()
        self.NC_PAL_MKT_SPREAD = "mkt_spread".upper()
        self.NC_PAL_FIXING_NEXT_DATE = "fixing_next_date".upper()
        self.NC_PAL_FIXING_PERIODICITY = "fixing_periodicity".upper()
        self.NC_PAL_RATE_CODE = "rate_code".upper()
        self.NC_PAL_CAP_STRIKE = "cap".upper()
        self.NC_PAL_FLOOR_STRIKE = "floor".upper()
        self.NC_PAL_MULT_SPREAD = "mult_spread".upper()
        self.NC_PAL_RATE_TYPE = "rate_type".upper()
        self.NC_PAL_TENOR = "tenor".upper()
        self.NC_PAL_CURVE_NAME = "curve_name".upper()
        self.NC_PAL_ACCRUAL_BASIS = 'accrual_basis'.upper()
        self.NC_PAL_FIXING_NB_DAYS = "fixing_nb_days".upper()
        self.NC_PAL_FIXING_RULE = "fixing_rule".upper()
        self.NC_PAL_TARGET_RATE = "target_rate".upper()

        """ ADD-ON COLS"""

        """NOTE: RENOMMER LES COLONNES ICI"""
        self.NC_CURVE_ACCRUAL = "CURVE_ACCRUAL"
        self.NC_TENOR_NUM = "TENOR_NUM"
        self.NC_LDP_CLE = "LDP_CLE"
        self.DOUTEUX_PALIER = "DOUTEUX_PALIER"
        self.NC_MOIS_PALIER_AMOR = "MOIS_PALIER_AMOR"
        self.NC_MOIS_PALIER_DEBUT = "MOIS_PALIER_DEBUT"
        self.NC_DATE_PALIER_REAL = "DATE_PALIER_REAL"
        self.NC_MOIS_DATE_PALIER = "MOIS_DATE_PALIER"
        self.NC_ACCRUAL_BASIS_PALIER = "ACCRUAL_BASIS"
        self.NC_TENOR_PALIER = "TENOR_PALIER"
        self.NC_CURVE_NAME_PALIER = "CURVE_NAME_PALIER"
        self.NC_VAL_PALIER = "VALEUR_PALIER"
        self.NC_RATE_PALIER = "RATE_PALIER"
        self.NC_CAPI_PER_PALIER = "CAPI_PERIOD_PALIER"
        self.NC_CAPI_RATE_PALIER = "CAPI_RATE_PALIER"
        self.NC_NB_PALIER = "NB_PALIER"
        self.NC_CONTRAT_REF_PALIER = "CONTRACT_REF_PALIER"
        self.NC_MKT_SPREAD_PALIER = "MKT_SPREAD_PALIER".upper()
        self.NC_MULT_FACT_PALIER = "MULT_FACTOR_PALIER".upper()
        self.NC_FIXING_NEXT_DATE_PALIER = "fixing_next_date_palier".upper()
        self.NC_FIXING_PERIODICITY_NUM_PALIER = "fixing_periodicity_num_palier".upper()
        self.NC_FIXING_PERIODICITY_PALIER = "fixing_periodicity_palier".upper()
        self.NC_RATE_CODE_PALIER = "rate_code_palier".upper()
        self.NC_CAP_STRIKE_PALIER = "cap_strike_palier".upper()
        self.NC_FLOOR_STRIKE_PALIER = "floor_strike_palier".upper()
        self.NC_RATE_TYPE_PALIER = "rate_type_palier".upper()
        self.NC_CURRENCY_PALIER = "currency".upper()
        self.NC_FREQ_INT_NUM_PALIER = "interest_frequency_num".upper()
        self.NC_FREQ_INT_PALIER = "interest_frequency".upper()
        self.NC_DECALAGE_AMOR_PALIER = "DECALAGE_DEBUT_AMOR_CAPITAL_PALIER"
        self.NC_DECALAGE_DEB_PER = "DECALAGE_DEBUT_PER"
        self.FIXING_NB_DAYS_PAL = "FIXING_NB_DAYS_PALIER"
        self.FIXING_RULE_PAL = "FIXING_RULE_PALIER"
        self.TARGET_RATE_PALIER = "TARGET_RATE_PALIER"

        self.NC_DAY_MATUR_DATE = "DAY_MATUR_DATE"
        self.NC_PROFIL_AMOR = "PROFIL_AMORT"
        self.NC_FREQ_AMOR = "FREQ_AMOR"
        self.NC_SUSPEND_AMOR_OR_CAPITALIZE_INTERESTS = "SUSPEND_AMOR_OR_CAPITALIZE_INT"
        self.NC_CAPITALIZE = "CAPITALIZE"
        self.NC_FREQ_CAP = "FREQ_CAP"
        self.NC_DEB_PROJ = "DEB_PROJ"
        self.NC_MATUR_RESID = "MATUR_RESI"
        self.NC_MATUR_INIT = "MATUR_INIT"
        self.NC_KEY_AGREG = "KEY_AGREG"
        self.NC_DATE_DEBUT_AMOR = "DATE_DEBUT_AMOR"
        self.NC_DATE_FIN_AMOR = "DATE_FIN_AMOR"
        self.NC_DECALAGE_AMOR_CAP = "DECALAGE_DEBUT_AMOR_CAPITAL"
        self.NC_DECALAGE_VERS_INT = "DECALAGE_DEBUT_VERS_INT"
        self.NC_DECALAGE_INT_CAP = "DECALAGE_DEBUT_CAP_INT"
        self.NC_DECALAGE_DEB_PER_FIX = "DECALAGE_DEBUT_PER_FIX"
        self.NC_DATE_DEBUT = "DATE_DEBUT"
        self.NC_DATE_FIN = "DATE_FIN"
        self.NB_DAYS_AN = "NB_JOURS_ANNEE"
        self.NB_DAYS_M0 = "NB_DAYS_M0"
        self.IS_M0_M1 = "IS_M0_M1"
        self.NC_INDEX_RATE = "INDEX_RATE"
        self.NC_FIXING_PERIODICITY_NUM = "FIXING_FREQ"
        self.NC_FREQ_INT = "FREQ_INT_NUM"
        self.NC_LDP_VALUE_DATE_REAL = self.NC_LDP_VALUE_DATE + "_REAL"
        self.NC_LDP_FIRST_AMORT_DATE_REAL = self.NC_LDP_FIRST_AMORT_DATE + "_REAL"
        self.NC_LDP_MATUR_DATE_REAL = self.NC_LDP_MATUR_DATE + "_REAL"
        self.NC_LDP_TRADE_DATE_REAL = self.NC_LDP_TRADE_DATE + "_REAL"
        self.NC_LDP_FIXING_NEXT_DATE_REAL = self.NC_LDP_FIXING_NEXT_DATE + "_REAL"
        self.NC_IS_CAPI_BEFORE_AMOR = "IS_CAPI_BEFORE_AMOR"

        self.NC_PRODUCT_TYPE = "PRODUCT_TYPE"
        self.NC_NOM_MULTIPLIER = "NOMINAL_MULTIPLIER"
        self.NC_LDP_IS_FUTURE_PRODUCT = "IS_FUTURE_PRODUCT"
        self.NC_PRICING_CURVE = "PRICING_CURVE"
    def load_grouped_vars(self):
        self.proj_vars = [self.NC_LDP_CONTRAT, self.NC_LDP_TRADE_DATE, self.NC_LDP_VALUE_DATE,
                          self.NC_LDP_FIRST_AMORT_DATE, self.NC_LDP_MATUR_DATE, \
                          self.NC_LDP_RELEASING_DATE, self.NC_LDP_RELEASING_RULE, self.NC_LDP_RATE,
                          self.NC_LDP_TYPE_AMOR, \
                          self.NC_LDP_FREQ_AMOR, self.NC_LDP_FREQ_INT, self.NC_LDP_ECHEANCE_VAL, self.NC_LDP_NOMINAL,
                          self.NC_LDP_OUTSTANDING, self.NC_LDP_CLE] \
                         + [self.NC_LDP_VALUE_DATE_REAL, self.NC_LDP_MATUR_DATE_REAL, self.NC_LDP_FIXING_NEXT_DATE_REAL,
                            self.NC_LDP_FIRST_AMORT_DATE_REAL] \
                         + [self.NC_LDP_CAPITALIZATION_PERIOD, self.NC_LDP_CAPI_MODE,
                            self.NC_LDP_CAPITALIZATION_RATE, self.NC_LDP_ACCRUAL_BASIS, \
                            self.NC_LDP_CURRENCY, self.NC_LDP_BROKEN_PERIOD, self.NB_DAYS_AN, self.NC_LDP_ETAB,
                            self.NC_LDP_INTERESTS_ACCRUALS, self.NC_LDP_PERFORMING,
                            self.NC_LDP_CONTRACT_TYPE] \
                         + [self.NC_LDP_CURVE_NAME, self.NC_LDP_TENOR, self.NC_LDP_MKT_SPREAD,
                            self.NC_LDP_FIXING_NEXT_DATE,
                            self.NC_LDP_CALC_DAY_CONVENTION, self.NC_LDP_FIXING_PERIODICITY, self.NC_LDP_RATE_CODE,
                            self.NC_LDP_CAP_STRIKE, self.NC_LDP_FLOOR_STRIKE, self.NC_LDP_MULT_SPREAD,
                            self.NC_LDP_RATE_TYPE, self.NC_LDP_MARCHE,
                            self.NC_LDP_FTP_RATE, self.NC_LDP_MATUR, self.NC_LDP_BASSIN, self.NC_LDP_CURRENT_RATE,
                            self.NC_LDP_BUY_SELL,
                            self.NC_NOM_MULTIPLIER, self.NC_LDP_NB_CONTRACTS, self.NC_LDP_TRADE_DATE_REAL,
                            self.NC_LDP_IS_CAP_FLOOR, self.NC_LDP_DATE_SORTIE_GAP, self.NC_LDP_DATE_SORTIE_GAP_REAL,
                            self.NC_LDP_FIXING_NB_DAYS, self.NC_PRODUCT_TYPE, self.NC_LDP_IS_FUTURE_PRODUCT,
                            self.NC_PRICING_CURVE, self.NC_LDP_FLOW_MODEL_NMD, self.NC_LDP_RM_GROUP, self.NC_LDP_RM_GROUP_PRCT,
                            self.NC_LDP_FLOW_MODEL_NMD_GPTX, self.NC_LDP_FIXING_RULE, self.NC_LDP_FIRST_COUPON_DATE,
                            self.NC_IS_CAPI_BEFORE_AMOR, self.NC_LDP_PALIER, self.NC_LDP_TARGET_RATE,
                            self.NC_LDP_FTP_FUNDING_SPREAD, self.NC_LDP_FLOW_MODEL_NMD_TCI, self.NC_LDP_TCI_METHOD,
                            self.NC_LDP_TCI_FIXED_RATE_CODE, self.NC_LDP_TCI_FIXED_TENOR_CODE,
                            self.NC_LDP_TCI_VARIABLE_TENOR_CODE, self.NC_LDP_TCI_VARIABLE_CURVE_CODE,
                            self.NC_LDP_SAVINGS_MODEL, self.NC_LDP_REDEMPTION_MODEL]

        self.ldp_vars = [self.NC_LDP_CONTRAT, self.NC_LDP_TRADE_DATE, self.NC_LDP_VALUE_DATE,
                         self.NC_LDP_FIRST_AMORT_DATE, self.NC_LDP_MATUR_DATE, \
                         self.NC_LDP_RELEASING_DATE, self.NC_LDP_RELEASING_RULE, self.NC_LDP_RATE,
                         self.NC_LDP_TYPE_AMOR, \
                         self.NC_LDP_FREQ_AMOR, self.NC_LDP_FREQ_INT, self.NC_LDP_ECHEANCE_VAL, self.NC_LDP_NOMINAL,
                         self.NC_LDP_OUTSTANDING,
                         self.NC_LDP_CAPITALIZATION_PERIOD, self.NC_LDP_CAPITALIZATION_RATE, self.NC_LDP_ACCRUAL_BASIS,
                         self.NC_LDP_CURRENCY,
                         self.NC_LDP_BROKEN_PERIOD, self.NC_LDP_ETAB, self.NC_LDP_INTERESTS_ACCRUALS,
                         self.NC_LDP_CONTRACT_TYPE, self.NC_LDP_CURVE_NAME,
                         self.NC_LDP_TENOR, self.NC_LDP_MKT_SPREAD, self.NC_LDP_FIXING_NEXT_DATE,
                         self.NC_LDP_CALC_DAY_CONVENTION,
                         self.NC_LDP_FIXING_PERIODICITY, self.NC_LDP_RATE_CODE, self.NC_LDP_CAP_STRIKE,
                         self.NC_LDP_FLOOR_STRIKE, self.NC_LDP_MULT_SPREAD,
                         self.NC_LDP_RATE_TYPE, self.NC_LDP_MARCHE, self.NC_LDP_CAPI_MODE, self.NC_LDP_FTP_RATE,
                         self.NC_LDP_CURRENT_RATE,
                         self.NC_LDP_BUY_SELL, self.NC_LDP_NB_CONTRACTS, self.NC_LDP_PERFORMING, self.NC_LDP_MATUR,
                         self.NC_LDP_IS_CAP_FLOOR, self.NC_LDP_DATE_SORTIE_GAP, self.NC_LDP_FIXING_NB_DAYS,
                         self.NC_LDP_BASSIN, self.NC_LDP_GESTION, self.NC_LDP_PALIER, self.NC_LDP_IS_FUTURE_PRODUCT,
                         self.NC_PRICING_CURVE, self.NC_LDP_FLOW_MODEL_NMD, self.NC_LDP_RM_GROUP, self.NC_LDP_RM_GROUP_PRCT,
                         self.NC_LDP_FLOW_MODEL_NMD_GPTX, self.NC_LDP_FIXING_RULE, self.NC_LDP_FIRST_COUPON_DATE,
                         self.NC_LDP_TARGET_RATE, self.NC_LDP_DIM6, self.NC_LDP_FTP_FUNDING_SPREAD, self.NC_LDP_FLOW_MODEL_NMD_TCI, self.NC_LDP_TCI_METHOD,
                         self.NC_LDP_TCI_FIXED_RATE_CODE, self.NC_LDP_TCI_FIXED_TENOR_CODE,
                         self.NC_LDP_TCI_VARIABLE_TENOR_CODE, self.NC_LDP_TCI_VARIABLE_CURVE_CODE,
                         self.NC_LDP_SAVINGS_MODEL, self.NC_LDP_REDEMPTION_MODEL]

        self.default_ldp_vars = {self.NC_LDP_CONTRAT: '', self.NC_LDP_TRADE_DATE: ".", self.NC_LDP_VALUE_DATE: ".",
                                 self.NC_LDP_FIRST_AMORT_DATE: ".",
                                 self.NC_LDP_MATUR_DATE: ".", self.NC_LDP_RELEASING_DATE: ".",
                                 self.NC_LDP_RELEASING_RULE: np.nan,
                                 self.NC_LDP_RATE: 0, self.NC_LDP_TYPE_AMOR: "A", self.NC_LDP_FREQ_AMOR: "Monthly",
                                 self.NC_LDP_FREQ_INT: "M", self.NC_LDP_ECHEANCE_VAL: np.nan, self.NC_LDP_NOMINAL: 0,
                                 self.NC_LDP_OUTSTANDING: 0,
                                 self.NC_LDP_CAPITALIZATION_PERIOD: "", self.NC_LDP_CAPITALIZATION_RATE: 0,
                                 self.NC_LDP_ACCRUAL_BASIS: "30/360",
                                 self.NC_LDP_CURRENCY: "EUR", self.NC_LDP_BROKEN_PERIOD: "Start Short",
                                 self.NC_LDP_ETAB: "",
                                 self.NC_LDP_INTERESTS_ACCRUALS: 0, self.NC_LDP_CONTRACT_TYPE: "",
                                 self.NC_LDP_CURVE_NAME: "",
                                 self.NC_LDP_TENOR: "1M", self.NC_LDP_MKT_SPREAD: 0, self.NC_LDP_FIXING_NEXT_DATE: ".",
                                 self.NC_LDP_CALC_DAY_CONVENTION: 1,
                                 self.NC_LDP_FIXING_PERIODICITY: "1M", self.NC_LDP_RATE_CODE: "",
                                 self.NC_LDP_CAP_STRIKE: 100, self.NC_LDP_FLOOR_STRIKE: -100,
                                 self.NC_LDP_MULT_SPREAD: 1, self.NC_LDP_RATE_TYPE: "FIXED", self.NC_LDP_MARCHE: "",
                                 self.NC_LDP_FTP_RATE: np.nan,
                                 self.NC_LDP_CURRENT_RATE: 0, self.NC_LDP_CAPI_MODE: "P", self.NC_LDP_BUY_SELL: "",
                                 self.NC_LDP_NB_CONTRACTS: 1,
                                 self.NC_LDP_PERFORMING: "F", self.NC_LDP_MATUR: "", self.NC_LDP_IS_CAP_FLOOR: "",
                                 self.NC_LDP_DATE_SORTIE_GAP : ".", self.NC_LDP_FIXING_NB_DAYS : 0,
                                 self.NC_LDP_BASSIN:"",self.NC_LDP_GESTION: "", self.NC_LDP_PALIER:"",
                                 self.NC_LDP_IS_FUTURE_PRODUCT: False, self.NC_PRICING_CURVE : "",
                                 self.NC_LDP_FLOW_MODEL_NMD:"", self.NC_LDP_RM_GROUP : "",
                                 self.NC_LDP_RM_GROUP_PRCT : 0, self.NC_LDP_FLOW_MODEL_NMD_GPTX:"",
                                 self.NC_LDP_FIXING_RULE: "B", self.NC_LDP_FIRST_COUPON_DATE : ".",
                                 self.NC_LDP_TARGET_RATE : np.nan, self.NC_LDP_DIM6: "",
                                 self.NC_LDP_FTP_FUNDING_SPREAD:0, self.NC_LDP_FLOW_MODEL_NMD_TCI : "",
                                 self.NC_LDP_TCI_METHOD : "", self.NC_LDP_TCI_FIXED_RATE_CODE: "",
                                 self.NC_LDP_TCI_FIXED_TENOR_CODE : "", self.NC_LDP_TCI_VARIABLE_TENOR_CODE : "",
                                 self.NC_LDP_TCI_VARIABLE_CURVE_CODE : "", self.NC_LDP_SAVINGS_MODEL: "",
                                 self.NC_LDP_REDEMPTION_MODEL : ""}

        self.pal_vars = [self.NC_PAL_CONTRAT, self.NC_PAL_BEGIN_DATE, self.NC_PAL_RATE, self.NC_PAL_VAL_EUR,
                         self.NC_PAL_FREQ_AMOR, self.NC_PAL_FREQ_INT,
                         self.NC_PAL_TYPE_AMOR, self.NC_PAL_MKT_SPREAD,
                         self.NC_PAL_FIXING_NEXT_DATE, self.NC_PAL_FIXING_PERIODICITY, self.NC_PAL_RATE_CODE,
                         self.NC_PAL_CAP_STRIKE, self.NC_PAL_FLOOR_STRIKE,
                         self.NC_PAL_MULT_SPREAD, self.NC_PAL_RATE_TYPE, self.NC_PAL_TENOR, self.NC_PAL_CURVE_NAME,
                         self.NC_PAL_ACCRUAL_BASIS, self.NC_LDP_FTP_RATE,  self.NC_PAL_FIXING_NB_DAYS,
                         self.NC_PAL_FIXING_RULE, self.NC_PAL_TARGET_RATE]

        self.default_pal_vars = {self.NC_PAL_CONTRAT: "", self.NC_PAL_BEGIN_DATE: ".", self.NC_PAL_RATE: 0,
                                 self.NC_PAL_VAL_EUR: np.nan,
                                 self.NC_PAL_FREQ_AMOR: "Monthly", self.NC_PAL_FREQ_INT: "M",
                                 self.NC_PAL_TYPE_AMOR: "A",
                                 self.NC_PAL_MKT_SPREAD: 0, self.NC_PAL_FIXING_NEXT_DATE: ".",
                                 self.NC_PAL_FIXING_PERIODICITY: "1M",
                                 self.NC_PAL_RATE_CODE: "", self.NC_PAL_CAP_STRIKE: 100, self.NC_PAL_FLOOR_STRIKE: -100,
                                 self.NC_PAL_MULT_SPREAD: 1,
                                 self.NC_PAL_RATE_TYPE: "FIXED", self.NC_PAL_TENOR: "1M", self.NC_PAL_CURVE_NAME: "",
                                 self.NC_PAL_ACCRUAL_BASIS: "30/360",
                                 self.NC_LDP_FTP_RATE: np.nan,  self.NC_PAL_FIXING_NB_DAYS: 0, self.NC_PAL_FIXING_RULE: "B",
                                 self.NC_PAL_TARGET_RATE : np.nan}
    def load_exit_indicators(self):
        self.exit_indicators = []
        if "GPLIQ" in self.exit_indicators_type:
            self.exit_indicators = self.exit_indicators + ["lef", "lem", "mni"]

        if "PN_NMD" in self.exit_indicators_type:
            self.exit_indicators = (self.exit_indicators + ["lem_stable", "lem_volatile", "mni_stable", "mni_volatile"]
                                             + ["tem_stable", "tem_volatile", "mni_gptx_stable", "mni_gptx_volatile"] +
                                    ["mni_tci_stable", "mni_tci_volatile", "mni_tci_gptx_stable", "mni_tci_gptx_volatile",
                                     "lef_stable", "lef_volatile"])

        if "GPTX" in self.exit_indicators_type:
            self.exit_indicators = self.exit_indicators + ["tef", "tem", "mni_gptx"]

        if "SC_TX" in self.exit_indicators_type:
            self.exit_indicators = self.exit_indicators + ["sc_rates"]

        if "TCI" and "GPLIQ" in self.exit_indicators_type:
            self.exit_indicators = self.exit_indicators + ["mni_tci"]

        if "TCI" and "GPTX" in self.exit_indicators_type:
            self.exit_indicators = self.exit_indicators + ["mni_tci_gptx"]

        if "TCI" and "SC_TX" in self.exit_indicators_type:
            self.exit_indicators = self.exit_indicators + ["sc_rates_tci"]

        if "EFFET_RARN" in self.exit_indicators_type:
            self.exit_indicators = self.exit_indicators + ["effet_RA", "effet_RN"]

        if "GPLIQ" and "RENEG" in self.exit_indicators_type:
            self.exit_indicators = self.exit_indicators + ["lem_statique", "lem_renego", "mni_statique",
                                                           "mni_renego"]

        if "SC_TX" and "RENEG" in self.exit_indicators_type:
            self.exit_indicators = self.exit_indicators + ["sc_rates_statique", "sc_rates_reneg"]



