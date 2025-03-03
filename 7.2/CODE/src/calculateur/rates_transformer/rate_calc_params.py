from calculateur.rates_transformer.swap_rates_interpolator import Rate_Interpolator
import logging
import pandas as pd
from services.rate_services.params import tx_referential as tx_ref
from services.rate_services.params import maturities_referential as m_ref
from mappings import general_mappings as gma
from services.rate_services.rates_manager import RatesManager

logger = logging.getLogger(__name__)


class Calculator_RateParams_Manager():
    def __init__(self, is_formated = False, rate_file_path ="", liq_file_path = "", is_scenario = False,
                 scenario_name = "", interpolate_curves = False, zc_curves_path = "", tci_file_path = ""):
        self.is_formated = is_formated
        self.is_scenario = is_scenario
        self.rate_file_path = rate_file_path
        self.liq_file_path = liq_file_path
        self.scenario_name = scenario_name
        self.interpolate_curves = interpolate_curves
        self.zc_curves_path = zc_curves_path
        self.tci_file_path = tci_file_path
        self.cols_num_rates = ["M" + str(i) for i in range(0, tx_ref.NB_PROJ_TAUX + 1)]

    def format_tci_data(self, tci_data):
        tx_tci_values = tci_data.fillna("*").drop(["dar", "all_t"], axis=1)
        cle = ["reseau", "company_code", "devise", "contract_type", "family", "rate_category"]
        tx_tci_values['new_key'] = tx_tci_values[cle].apply(lambda row: '$'.join(row.values.astype(str)), axis=1)
        tx_tci_values = tx_tci_values.drop_duplicates(subset=cle).set_index('new_key').drop(columns=cle, axis=1).copy()
        return tx_tci_values

    def filter_by_scenario(self, rates_data):
        if self.is_scenario:
            rates_data = rates_data[rates_data["SCENARIO"] == self.scenario_name].copy()
        return rates_data

    def get_tci_data(self):
        tci_data = pd.read_csv(self.tci_file_path, sep=";", decimal=",")
        return tci_data

    def get_rates_data(self):
        rates_data = pd.read_csv(self.rate_file_path, sep=";", decimal=",")
        return rates_data

    def get_zc_data(self):
        zc_curves_data = pd.read_csv(self.zc_curves_path, sep=";", decimal=",")
        return zc_curves_data

    def load_rate_params(self, sc_rates_df = pd.DataFrame(), tci_df = pd.DataFrame(), zc_df =pd.DataFrame()):
        if not self.is_formated:
            sc_rates_df = RatesManager.get_sc_df(self.scenario_name,
                                                  self.rate_file_path, self.liq_file_path)
            sc_rates_df[self.cols_num_rates] = sc_rates_df[self.cols_num_rates] / 100
        else:
            if len(sc_rates_df) == 0:
                sc_rates_df = self.get_rates_data()

            if self.is_scenario:
                sc_rates_df =  self.filter_by_scenario(sc_rates_df)

        try:
            if len(tci_df) == 0:
                tci_df = self.get_tci_data()
            tci_df = self.format_tci_data(tci_df)
        except:
            logger.debug("NO TCI CURVES AVAILABLE")
            tci_df = []

        try:
            if len(zc_df) == 0:
                zc_df = self.get_zc_data()
        except:
            logger.debug("NO ZC CURVES AVAILABLE")
            zc_df = []

        tx_params = {"curves_df": {"data": sc_rates_df, "cols": self.cols_num_rates, "max_proj": tx_ref.NB_PROJ_TAUX,
                                   "curve_code": tx_ref.CN_CODE_COURBE, "tenor": tx_ref.CN_MATURITY,
                                   "maturity_to_days": m_ref.maturity_to_days_360, "curve_name_taux_pel": "TAUX_PEL",
                                   "tenor_taux_pel": "12M1D"},
                     "accrual_map": {'data': gma.mapping_taux["CURVES_BASIS_CONV"],
                                     "accrual_conv_col": "ACCRUAL_CONVERSION",
                                     "type_courbe_col": "TYPE DE COURBE", "accrual_method_col": "ACCRUAL_METHOD",
                                     "alias": "ALIAS",
                                     "standalone_const": "Standalone index", "curve_name": "CURVE_NAME"},
                     "ZC_DATA": {"data": zc_df},
                     "map_pricing_curves": {"data": gma.mapping_taux["COURBES_PRICING_PN"],
                                            "col_pricing_curve": "COURBE PRICING"},
                     "map_index_curve_tenor": {"data": gma.mapping_taux["RATE_CODE-CURVE"].set_index(["CCY_CODE", "RATE_CODE"]),
                                               "col_curve": "CURVE_NAME", "col_tenor": "TENOR"},
                     "tci_vals": {"data": tci_df}}

        rate_int_cls = Rate_Interpolator()
        if self.interpolate_curves:
            tx_params["dic_tx_swap"] = rate_int_cls.interpolate_curves(tx_params)
        else:
            tx_params["dic_tx_swap"] = []

        return tx_params
