import pandas as pd
import logging
import numpy as np
from utils import general_utils as gu

logger = logging.getLogger(__name__)

class Data_TARGET_RATES_Manager():

    def __init__(self, cls_format, cls_hz_params, cls_fields, data_target_rates):
        self.cls_format = cls_format
        self.cls_fields = cls_fields
        self.data_target_rates = data_target_rates
        self.cls_hz_params = cls_hz_params
        self.load_columns()
        self.format_target_rates()

    def load_columns(self):
        self.NC_TARGET_RATES_CONTRACT_TYPE = "contract_type".upper()
        self.NC_TARGET_RATES_ETAB = "etab".upper()
        self.NC_TARGET_RATES_PALIER = "counterparty_code".upper()
        self.NC_TARGET_RATES_MARCHE = "family".upper()
        self.NC_TARGET_RATES_RATE_CODE = "rate_code".upper()
        self.NC_TARGET_RATES_INDEX_CODE = "index_code".upper()
        self.NC_TARGET_RATES_CURRENCY = "currency".upper()
        self.NC_TARGET_RATES_NUMS_COLS = ["M%s" %i if i<=9 else "M%s" %i for i in range(1,61)]

    def format_target_rates(self):
        if len(self.data_target_rates) > 0:
            data_target_rates = self.cls_format.upper_columns_names(self.data_target_rates)
            keys = [self.NC_TARGET_RATES_ETAB, self.NC_TARGET_RATES_RATE_CODE,
                    self.NC_TARGET_RATES_MARCHE, self.NC_TARGET_RATES_CONTRACT_TYPE, self.NC_TARGET_RATES_PALIER, self.NC_TARGET_RATES_CURRENCY]
            data_target_rates = data_target_rates.rename(columns = {"M0" + str(i):"M" + str(i) for i in range(1, 10)})
            self.num_cols = [x for x in self.NC_TARGET_RATES_NUMS_COLS if x in data_target_rates.columns]
            data_target_rates[self.NC_TARGET_RATES_PALIER] =  data_target_rates[self.NC_TARGET_RATES_PALIER].fillna("")
            data_target_rates = gu.force_integer_to_string(data_target_rates, self.NC_TARGET_RATES_PALIER)
            data_target_rates = data_target_rates[keys + self.num_cols].copy()
            data_target_rates[self.num_cols] = data_target_rates[self.num_cols].astype(np.float64)
            data_target_rates_no_dup = data_target_rates.drop_duplicates(keys)
            if len(data_target_rates_no_dup) > len(data_target_rates):
                logger.warning("Il y a des doublons dans les taux cibles")
            data_target_rates_no_dup = data_target_rates_no_dup.set_index(keys)
            self.data_target_rates = data_target_rates_no_dup

    ####@profile
    def get_target_rates_data(self, data, col_rate_code, n, t):
        if len(self.data_target_rates) > 0:
            data_target_rates = self.data_target_rates.copy()
            data_tmp = data.copy()
            cols_sup = ["M" + str(i) for i in range(len(self.num_cols) + 1, t + 1)]
            if len(cols_sup) > 0:
                data_target_rates = pd.concat(
                    [data_target_rates, pd.DataFrame(np.full((self.data_target_rates.shape[0], len(cols_sup)), np.nan), \
                                               index=self.data_target_rates.index, columns=cols_sup)], axis=1).ffill(axis=1)

            data_tmp = gu.force_integer_to_string(data_tmp, self.NC_TARGET_RATES_PALIER)
            data_target_rates = data_target_rates.iloc[:, :t]
            keys = [self.cls_fields.NC_LDP_ETAB, col_rate_code,
                               self.cls_fields.NC_LDP_MARCHE, self.cls_fields.NC_LDP_CONTRACT_TYPE,
                               self.cls_fields.NC_LDP_PALIER, self.cls_fields.NC_LDP_CURRENCY]

            data_tmp[self.cls_fields.NC_LDP_PALIER] =  data_tmp[self.cls_fields.NC_LDP_PALIER].fillna("").replace("nan","")
            data_tmp = gu.force_integer_to_string(data_tmp, self.cls_fields.NC_LDP_PALIER)

            data_target_rates = data_tmp.join(data_target_rates, on = keys)[self.num_cols + cols_sup]

            if len(data_target_rates) != len(data_tmp):
                logger.error("Il y a un problème avec la jointure des Spreads. Des doublons?")
                raise ValueError("")
        else:
            return np.full((n, t), np.nan)
        return np.array(data_target_rates).astype(np.float64) / 10000