import pandas as pd
import logging
import numpy as np
from utils import general_utils as gu

logger = logging.getLogger(__name__)


class Data_MARGES_INDEX_Manager():

    def __init__(self, cls_format, cls_hz_params, cls_fields, data_spread):
        self.cls_format = cls_format
        self.cls_fields = cls_fields
        self.data_spread = data_spread
        self.cls_hz_params = cls_hz_params
        self.load_columns()
        self.format_marge_file()

    def load_columns(self):
        self.NC_SPREAD_CONTRACT_TYPE = "contract_type".upper()
        self.NC_SPREAD_ETAB = "etab".upper()
        self.NC_SPREAD_PALIER = "counterparty_code".upper()
        self.NC_SPREAD_MARCHE = "family".upper()
        self.NC_SPREAD_RATE_CODE = "rate_code".upper()
        self.NC_SPREAD_INDEX_CODE = "index_code".upper()
        self.NC_SPREAD_CURRENCY = "currency".upper()
        self.NC_SPREAD_NUMS_COLS = ["M%s" % i if i <= 9 else "M%s" % i for i in range(1, 61)]
        self.CASDEN_CONTRACT = "A/P-CASDEN"
        self.TX_CASDEN = "TX_CASDEN"
        self.TX_CASDEN2 = "TX_CASDEN2"

    def format_marge_file(self):
        if len(self.data_spread) > 0:
            data_spread = self.cls_format.upper_columns_names(self.data_spread)
            if not self.NC_SPREAD_PALIER in self.data_spread.columns.tolist():
                self.NC_SPREAD_RATE_CODE = self.NC_SPREAD_INDEX_CODE
                keys = [self.NC_SPREAD_ETAB, self.NC_SPREAD_RATE_CODE, self.NC_SPREAD_CONTRACT_TYPE]
                self.type_spread = "normal"
            else:
                keys = [self.NC_SPREAD_ETAB, self.NC_SPREAD_RATE_CODE, self.NC_SPREAD_MARCHE,
                        self.NC_SPREAD_CONTRACT_TYPE, self.NC_SPREAD_PALIER, self.NC_SPREAD_CURRENCY]
                self.type_spread = "detailed"

                data_spread[self.NC_SPREAD_PALIER] = data_spread[self.NC_SPREAD_PALIER].fillna("")
                data_spread = gu.force_integer_to_string(data_spread, self.NC_SPREAD_PALIER)

            data_spread = self.change_casden_contract(data_spread)
            data_spread = self.replace_rate_code_dav(data_spread, self.NC_SPREAD_RATE_CODE)
            data_spread = data_spread.rename(columns={"M0" + str(i): "M" + str(i) for i in range(1, 10)})
            self.num_cols = [x for x in self.NC_SPREAD_NUMS_COLS if x in data_spread.columns]

            data_spread = data_spread[keys + self.num_cols].copy()
            data_spread[self.num_cols] = data_spread[self.num_cols].astype(np.float64).fillna(0)
            data_spread_rates_no_dup = data_spread.drop_duplicates(keys)
            if len(data_spread_rates_no_dup) > len(data_spread):
                logger.warning("Il y a des doublons dans les spread")
            data_spread_rates_no_dup = data_spread_rates_no_dup.set_index(keys)
            self.data_spread = data_spread_rates_no_dup

    def replace_rate_code_dav(self, data, col_rate_code):
        data[col_rate_code] = data[col_rate_code].replace("TX_SCDAVCORP", "EUREURIB3Mf").replace("TX_SCDAVPART",
                                                                                                 "CMS5Yf")
        return data

    ####@profile
    def get_spread_data(self, data, col_rate_code, n, t):
        if len(self.data_spread) > 0:
            data_spread = self.data_spread.copy()
            data_tmp = data.copy()
            cols_sup = ["M" + str(i) for i in range(len(self.num_cols) + 1, t + 1)]
            if len(cols_sup) > 0:
                data_spread = pd.concat(
                    [data_spread, pd.DataFrame(np.full((self.data_spread.shape[0], len(cols_sup)), np.nan), \
                                               index=self.data_spread.index, columns=cols_sup)], axis=1).ffill(axis=1)

            data_spread = data_spread.iloc[:, :t]

            if self.type_spread == "detailed":
                keys = [self.cls_fields.NC_LDP_ETAB, col_rate_code, self.cls_fields.NC_LDP_MARCHE,
                        self.cls_fields.NC_LDP_CONTRACT_TYPE, self.cls_fields.NC_LDP_PALIER,
                        self.cls_fields.NC_LDP_CURRENCY]
            else:
                keys = [self.cls_fields.NC_LDP_ETAB, col_rate_code, self.cls_fields.NC_LDP_CONTRACT_TYPE]

            data_tmp = self.format_palier(data_tmp)
            data_spread = data_tmp.join(data_spread, on=keys)[self.num_cols + cols_sup]
            if len(data_spread) != len(data_tmp):
                logger.error("Il y a un problème avec la jointure des Spreads. Des doublons?")
        else:
            return np.zeros((n, t))
        return np.nan_to_num(np.array(data_spread).astype(np.float64)) / 10000

    def format_palier(self, data_tmp):
        data_tmp[self.cls_fields.NC_LDP_PALIER] = data_tmp[self.cls_fields.NC_LDP_PALIER].fillna("").replace("nan", "")
        data_tmp = gu.force_integer_to_string(data_tmp, self.cls_fields.NC_LDP_PALIER)
        return data_tmp

    def change_casden_contract(self, data_spread):
        filtre_casden_cont = data_spread[self.NC_SPREAD_CONTRACT_TYPE] == self.CASDEN_CONTRACT
        filtre_casden_tx1 = data_spread[self.NC_SPREAD_RATE_CODE] == self.TX_CASDEN
        filtre_casden_tx2 = data_spread[self.NC_SPREAD_RATE_CODE] == self.TX_CASDEN2
        casden = {}
        for i in range(1, 3):
            casden[i] = data_spread[filtre_casden_cont & eval("filtre_casden_tx" + str(i))].copy().reset_index(
                drop=True)
            casden_a = casden[i].copy()
            casden_p = casden[i].copy()
            casden_a2 = casden[i].copy()
            casden_p2 = casden[i].copy()
            if casden[i].shape[0] > 0:
                casden_a[self.NC_SPREAD_CONTRACT_TYPE] = "A-CASDEN"
                casden_p[self.NC_SPREAD_CONTRACT_TYPE] = "P-CASDEN"
                casden_a2[self.NC_SPREAD_CONTRACT_TYPE] = "A-CASDEN2"
                casden_p2[self.NC_SPREAD_CONTRACT_TYPE] = "P-CASDEN2"
            casden[i] = pd.concat([casden_a, casden_p, casden_a2, casden_p2])

        data_spread = data_spread[~filtre_casden_cont].copy()

        return pd.concat([data_spread, casden[1], casden[2]])
