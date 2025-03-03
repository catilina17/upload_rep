import numpy as np
import pandas as pd
import logging

nan = np.nan
logger = logging.getLogger(__name__)


class Data_Rate_Manager():
    """
    Formate les données
    """

    def __init__(self, cls_fields, cls_hz_params, cls_spread_index, cls_target_rates, with_dyn_data, tx_params):
        self.cls_fields = cls_fields
        self.cls_hz_params = cls_hz_params
        self.list_devises = ["EUR", "CHF", "GBP", "JPY", "USD"]
        self.dic_devises_index = {"EUR": "EURIBOR", "USD": "USDLIBOR",
                                  "GBP": "GBPLIBOR", "JPY": "JPYLIBOR", "CHF": "CHFLIBOR"}
        self.cls_spread_index = cls_spread_index
        self.cls_target_rates = cls_target_rates
        self.with_dyn_data = with_dyn_data
        self.ACCRUAL_CONVERSION = tx_params["accrual_map"]["accrual_conv_col"]
        self.ACCRUAL_METHOD = tx_params["accrual_map"]["accrual_method_col"]
        self.STANDALONE_INDEX = tx_params["accrual_map"]["type_courbe_col"]
        self.STANDALONE_INDEX_CONST = tx_params["accrual_map"]["standalone_const"]
        self.tx_params = tx_params
        self.dic_tx_swap = tx_params["dic_tx_swap"]
        self.missing_rate_input_curves = []
        self.missing_curves_in_accrual_map = []

    # ####@profile
    def prepare_curve_rates(self, cls_proj, cls_model_params, tx_params, cls_pricing_tf=None):
        data_ldp = cls_proj.data_ldp.reset_index(drop=True)
        data_rate_sc = np.zeros((cls_proj.n, cls_proj.t_max))
        data_ldp[self.cls_fields.NC_CURVE_ACCRUAL] = ""
        map_accrual_key_cols = {"CURVE_NAME": self.cls_fields.NC_LDP_CURVE_NAME, "TENOR": self.cls_fields.NC_TENOR_NUM}
        is_tv = (data_ldp[self.cls_fields.NC_LDP_RATE_TYPE] == "FLOATING").values
        _n = is_tv[is_tv].shape[0]
        if _n > 0:
            data_rate_sc[is_tv] \
                = self.match_data_with_sc_rates(data_ldp[is_tv].copy(), tx_params, cls_proj.t_max,
                                                [self.cls_fields.NC_LDP_CURVE_NAME, self.cls_fields.NC_LDP_TENOR],
                                                raise_error=False)

            data_ldp = self.get_curve_accruals_col(data_ldp, tx_params, map_accrual_key_cols, is_tv)

            if cls_proj.name_product == "all_ech_pn":
                if not cls_pricing_tf is None:
                    data_rate_sc[is_tv] = data_rate_sc[is_tv] + cls_pricing_tf.liq_rate_tv.reshape(_n, 1)
                floor = (data_ldp[[self.cls_fields.NC_LDP_CONTRACT_TYPE]]
                         .join(cls_model_params.cls_rates_floor.rates_floors,
                               on=[self.cls_fields.NC_LDP_CONTRACT_TYPE])).iloc[:, 1].values

                data_ldp.loc[is_tv, self.cls_fields.NC_LDP_FLOOR_STRIKE] \
                    = np.where(~np.isnan(floor[is_tv]), floor[is_tv],
                               data_ldp.loc[is_tv, self.cls_fields.NC_LDP_FLOOR_STRIKE].values)

            cls_proj.data_ldp = data_ldp.copy()

        self.data_rate_sc = data_rate_sc

    def match_data_with_sc_rates(self, data_ldp, tx_params, t, keys, **kwargs):
        data_rate_sc = self.map_rate_code_to_tx_curves(keys, tx_params, data_ldp, t, **kwargs)
        return data_rate_sc

    def map_rate_code_to_tx_curves(self, cols_rate_code, tx_params, data_ldp, t, raise_error=True, deb_col=1):
        tx_curves = tx_params["curves_df"]["data"].copy()
        curve_col = tx_params["curves_df"]["curve_code"]
        tenor_col = tx_params["curves_df"]["tenor"]
        max_cms = tx_params["curves_df"]["max_proj"]
        cols_cms = tx_params["curves_df"]["cols"]
        cols = [x for x in cols_cms if int(x[1:]) in range(deb_col, min(t + 1, max_cms + 1))]
        tx_curves[curve_col] = tx_curves[curve_col].str.strip().str.upper()
        tx_curves[tenor_col] = tx_curves[tenor_col].str.strip().str.upper()
        tx_curves = tx_curves.set_index([curve_col, tenor_col])

        data_ldp[cols_rate_code] = data_ldp[cols_rate_code].apply(lambda x:x.str.upper())
        data_rate_sc = data_ldp[cols_rate_code].reset_index(drop=True).copy().join(tx_curves, on=cols_rate_code)

        if data_rate_sc[cols_cms[0]].isnull().any() > 0:
            new_missing_curves = data_rate_sc[data_rate_sc[cols_cms[0]].isna()][cols_rate_code].fillna("").drop_duplicates().values.tolist()
            new_missing_curves = [x for x in new_missing_curves if x not in self.missing_rate_input_curves]
            if len(new_missing_curves) > 0:
                self.missing_rate_input_curves = self.missing_rate_input_curves + new_missing_curves
                logger.error("Certaines courbes,tenor sont manquantes dans le RATE INPUT")
                logger.error(data_rate_sc[data_rate_sc[cols_cms[0]].isna()][cols_rate_code].fillna("").drop_duplicates().values.tolist())
                if raise_error:
                    raise ValueError("Certaines courbes,tenor sont manquantes dans le RATE INPUT")
        if len(data_rate_sc) > len(data_ldp):
            list_dup = data_rate_sc[data_rate_sc.index.duplicated()][cols_rate_code].drop_duplicates().values.tolist()
            logger.error("Certaines courbes,tenor sont dupliquées dans le RATE INPUT : %s" % list_dup)
            if raise_error:
                raise ValueError("Certaines courbes,tenor sont dupliquées dans le RATE INPUT : %s" % list_dup)

        data_rate_sc = data_rate_sc[cols].copy()
        data_rate_sc = self.add_cols_sup(data_rate_sc, max_cms, t)
        data_rate_sc = np.array(data_rate_sc).astype(np.float64)

        return data_rate_sc

    def get_curve_accrual_basis(self, data_ldp, tx_params, map_accrual_key_cols, cols_accruals):
        accruals_map = tx_params["accrual_map"]["data"][cols_accruals].copy()
        curve_col = map_accrual_key_cols["CURVE_NAME"]
        tenor_col = map_accrual_key_cols["TENOR"]
        data = data_ldp[[curve_col, tenor_col]].copy()
        data[[curve_col, tenor_col]] \
            = data[[curve_col, tenor_col]].map(lambda x: str(x).upper()).map(lambda x: str(x).strip())
        data[tenor_col] = np.where(data[tenor_col].astype(float).astype(int).values >= 12, "1Y", "1D")
        data2 = data.copy()
        data2[tenor_col] = "1D"
        data_mapped = data.set_index([curve_col, tenor_col]).join(accruals_map, on=[curve_col, tenor_col])
        data_mapped2 = data2.set_index([curve_col, tenor_col]).join(accruals_map, on=[curve_col, tenor_col])
        data_mapped_final = data_mapped.reset_index(drop=True).copy()
        data_mapped_final.update(data_mapped2.reset_index(drop=True), overwrite=False)

        if data_mapped_final[self.ACCRUAL_METHOD].isnull().any():
            new_missing_curves = data[data_mapped_final[self.ACCRUAL_METHOD].isnull().values].reset_index()[
                [curve_col, tenor_col]].fillna("").agg("-".join, axis=1).unique().tolist()
            new_missing_curves = [x for x in new_missing_curves if x not in self.missing_curves_in_accrual_map]
            if len(new_missing_curves) > 0:
                self.missing_curves_in_accrual_map = self.missing_curves_in_accrual_map + new_missing_curves
                logger.error("Certaines courbes de taux sont manquantes dans le mapping des CURVE ACCRUALS")
                logger.error(
                    data[data_mapped_final[self.ACCRUAL_METHOD].isnull().values].reset_index()[
                        [curve_col, tenor_col]].fillna("").agg("-".join, axis=1).unique().tolist())
            #raise ValueError("Certaines courbes de taux sont manquantes dans le mapping des CURVE ACCRUALS")
        elif len(data_mapped) > len(data):
            logger.error("Certaines courbes de taux sont dupliquées dans le mapping des CURVE ACCRUALS")
            raise ValueError("Certaines courbes de taux sont dupliquées dans le mapping des CURVE ACCRUALS")

        return data_mapped_final.values

    def get_curve_accruals_col(self, data_ldp, tx_params, map_accrual_key_cols, is_tv):
        cols_accruals = [self.ACCRUAL_METHOD, self.ACCRUAL_CONVERSION, self.STANDALONE_INDEX]
        data_ldp = data_ldp.drop(cols_accruals, axis=1, errors="ignore")
        data_ldp = pd.concat([data_ldp, pd.DataFrame([], columns=cols_accruals)], axis=1)
        data_ldp.loc[is_tv, cols_accruals] = self.get_curve_accrual_basis(data_ldp[is_tv], tx_params,
                                                                          map_accrual_key_cols, cols_accruals)
        return data_ldp

    def add_cols_sup(self, tx_curves, max_cms, t):
        cols_sup = ["M" + str(i) for i in range(min(t, max_cms) + 1, t + 1)]
        tx_curves = pd.concat(
            [tx_curves, pd.DataFrame(np.full((tx_curves.shape[0], len(cols_sup)), np.nan), index=tx_curves.index,
                                     columns=cols_sup)], axis=1).ffill(axis=1)
        return tx_curves

    def replace_aliases_in_index_curves(self):
        accruals_map = self.tx_params["accrual_map"]
        accruals_map_data = accruals_map["data"].copy()
        col_alias = accruals_map["alias"]
        curve_name = accruals_map["curve_name"]
        accruals_map_data = accruals_map_data.reset_index()[[curve_name, col_alias]].drop_duplicates()
        accruals_map_data = accruals_map_data[accruals_map_data[col_alias].notnull()].copy()

        curves_df = self.tx_params["curves_df"]["data"].copy()
        CURVE_CODE = self.tx_params["curves_df"]["curve_code"]
        TENOR_CODE = self.tx_params["curves_df"]["tenor"]

        curves_df_sup = accruals_map_data.merge(curves_df, right_on=[CURVE_CODE], left_on=[col_alias], how="left")
        curves_df_sup[CURVE_CODE] = curves_df_sup[curve_name]

        curves_df_sup = curves_df_sup.drop([curve_name, col_alias], axis=1)

        new_tx_curves= pd.concat([self.tx_params["curves_df"]["data"], curves_df_sup])

        #new_tx_curves = new_tx_curves.drop_duplicates([CURVE_CODE, TENOR_CODE], keep='first')

        self.tx_params["curves_df"]["data"] = new_tx_curves.copy()





