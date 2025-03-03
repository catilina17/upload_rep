import numpy as np
import pandas as pd
from calculateur.data_transformers.data_in.nmd.class_nmd_templates import Data_NMD_TEMPLATES
import logging

logger = logging.getLogger(__name__)


class Data_NMD_SPREADs():
    def __init__(self, cls_fields, cls_fields_pa, cls_format, source_files_pn,
                 horizon, etab, max_pn, cls_tmp_nmd = None):
        self.cls_fields = cls_fields
        self.cls_fields_pa = cls_fields_pa
        self.cls_format = cls_format
        self.source_files_pn = source_files_pn
        self.horizon = horizon
        self.etab = etab
        self.data_template = cls_tmp_nmd.data_template_mapped
        self.max_pn = max_pn
        self.load_columns_names()

    def load_columns_names(self):
        self.ALLOCATION_KEY_RCO = Data_NMD_TEMPLATES.ALLOCATION_KEY
        self.TEMPLATE_WEIGHT_RCO = Data_NMD_TEMPLATES.TEMPLATE_WEIGHT_RCO
        self.ALLOCATION_KEY_PASS_ALM = [self.cls_fields_pa.NC_PA_ETAB, self.cls_fields_pa.NC_PA_DEVISE,
                                        self.cls_fields_pa.NC_PA_CONTRACT_TYPE,
                                        self.cls_fields_pa.NC_PA_MARCHE, self.cls_fields_pa.NC_PA_RATE_CODE,
                                        self.cls_fields_pa.NC_PA_PALIER]
        self.TEMPLATE_WEIGHT_PASS_ALM = "TEMPLATE_WEIGHT_PASS_ALM"

        self.NC_TX_CIBLE = "TxProdCible(bps)"
        self.NC_TX_SPREAD = "TxSpPN(bps)"

        self.num_cols = ["M%s" % i for i in range(1, min(self.max_pn + 1, self.horizon + 1))]

    def get_data_spreads(self):
        logging.debug('    Lecture du fichier PN NMD et import des SPREADs')
        if not "DATA" in self.source_files_pn["LDP"]:
            data_nmd_spread_and_target = pd.read_csv(self.source_files_pn["LDP"]["CHEMIN"],
                                                     delimiter=self.source_files_pn["LDP"]["DELIMITER"],
                                                     decimal=self.source_files_pn["LDP"]["DECIMAL"],
                                                     engine='python', encoding="ISO-8859-1")
        else:
            data_nmd_spread_and_target = self.source_files_pn["LDP"]["DATA"]

        if len(self.source_files_pn["LDP"]["DATA"]) > 0:
            data_nmd_spread_and_target = self.cls_format.upper_columns_names(data_nmd_spread_and_target)
            data_nmd_spread_and_target = self.filter_data(data_nmd_spread_and_target)

            self.data_tx_spread, self.data_tx_cible = self.generate_tx_spread_cible_table(data_nmd_spread_and_target,
                                                                                          self.data_template)
        else:
            self.data_tx_spread, self.data_tx_cible = [], []

    def generate_tx_spread_cible_table(self, data_nmd_pn, data_template):

        cles = [self.cls_fields.NC_LDP_ETAB, self.cls_fields.NC_LDP_CURRENCY, self.cls_fields.NC_LDP_CONTRACT_TYPE,
                self.cls_fields.NC_LDP_MARCHE, self.cls_fields.NC_LDP_RATE_CODE, self.cls_fields.NC_LDP_PALIER]

        data_transition_mapping = self.recuperation_transition_mapping_from_templates(data_template, cles)
        data_nmd_pn = data_nmd_pn.join(data_transition_mapping, on=self.ALLOCATION_KEY_PASS_ALM, rsuffix='_TEMPLATE')

        data_nmd_pn_tx_spread = data_nmd_pn[data_nmd_pn[self.cls_fields_pa.NC_PA_IND03] == self.NC_TX_SPREAD].copy()
        data_nmd_pn_tx_cible = data_nmd_pn[data_nmd_pn[self.cls_fields_pa.NC_PA_IND03] == self.NC_TX_CIBLE].copy()

        data_nmd_pn_tx_spread = data_nmd_pn_tx_spread[cles + self.num_cols].drop_duplicates()
        data_nmd_pn_tx_cible = data_nmd_pn_tx_cible[cles + self.num_cols].drop_duplicates()

        return data_nmd_pn_tx_spread, data_nmd_pn_tx_cible

    def recuperation_transition_mapping_from_templates(self, data_template, cles):
        keep_cols = list(set(cles + self.ALLOCATION_KEY_PASS_ALM))
        data_transition_mapping = data_template[keep_cols].drop_duplicates()
        data_transition_mapping = data_transition_mapping.set_index(self.ALLOCATION_KEY_PASS_ALM)
        return data_transition_mapping

    def filter_data(self, data):
        data = data[data[self.cls_fields_pa.NC_PA_IND03].isin([self.NC_TX_SPREAD, self.NC_TX_CIBLE])].copy()
        data_tx_spread = data[data[self.cls_fields_pa.NC_PA_IND03] == self.NC_TX_SPREAD].copy()
        data_tx_cible = data[data[self.cls_fields_pa.NC_PA_IND03] == self.NC_TX_CIBLE].copy()
        filter = ((~data_tx_spread[self.num_cols].isnull().all(1)).values
                  | (~data_tx_cible[self.num_cols].isnull().all(1)).values)
        data = data[filter.repeat(2, axis=0)]
        all_qual_cols = [x for x in data.columns if x not in ["M" + str(i) for i in range(0, 301)]]
        data = data[all_qual_cols + self.num_cols].copy()
        data[self.num_cols] = data[self.num_cols].astype(np.float64)
        return data
