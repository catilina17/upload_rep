import numpy as np
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import pandas as pd
import utils.general_utils as ut
import logging

logger = logging.getLogger(__name__)


def generate_index(pn_ech, repeat=1):
    INDEXO = np.array(["ECH" + str(i) for i in range(1, int(pn_ech.shape[0] / repeat) + 1)])
    INDEXO = np.repeat(INDEXO, repeat)
    pn_ech[pa.NC_PA_INDEX] = INDEXO
    return pn_ech


def add_missing_indics(pn_data):
    indic_ordered = [pa.NC_PA_DEM, pa.NC_PA_MG_CO, pa.NC_PA_TX_SP, pa.NC_PA_TX_CIBLE]
    keys_order = [pa.NC_PA_BILAN, pa.NC_PA_CLE, pa.NC_PA_INDEX] + pa.NC_PA_COL_SPEC_ECH
    empty_indics = [pa.NC_PA_MG_CO, pa.NC_PA_TX_SP, pa.NC_PA_TX_CIBLE]

    pn_data = ut.add_empty_indics(pn_data, empty_indics, pa.NC_PA_IND03, \
                                  pa.NC_PA_DEM, pa.NC_PA_COL_SORTIE_NUM_PN, order=True, indics_ordered=indic_ordered, \
                                  keys_order=keys_order)
    return pn_data


def add_col_bilan(data):
    filtres = [data[pa.NC_PA_CONTRACT_TYPE].str[0:2] == "A-", \
               data[pa.NC_PA_CONTRACT_TYPE].str[0:2] == "P-"]
    choices = ["B ACTIF", "B PASSIF"]
    data[pa.NC_PA_BILAN] = np.select(filtres, choices)
    return data


def add_missing_key_cols(data):
    missing_indics = [x for x \
                      in pa.NC_PA_CLE_OUTPUT \
                      if not x in data.columns.tolist()]

    data = pd.concat([data, pd.DataFrame([["-"] * len(missing_indics)], \
                                         index=data.index, columns=missing_indics)], axis=1)

    missing_indics = [x for x \
                      in pa.NC_PA_COL_SPEC_ECH \
                      if not x in data.columns.tolist()]

    data = pd.concat([data, pd.DataFrame([[""] * len(missing_indics)], \
                                         index=data.index, columns=missing_indics)], axis=1)

    data[pa.NC_PA_CLE] = \
        data[pa.NC_PA_CLE_OUTPUT].apply(lambda x: "_".join(x), axis=1)

    return data
