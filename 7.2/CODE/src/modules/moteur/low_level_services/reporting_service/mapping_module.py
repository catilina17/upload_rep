import utils.general_utils as gu
from mappings import mapping_functions as mp
import logging
from mappings.pass_alm_fields import PASS_ALM_Fields as pa

logger = logging.getLogger(__name__)

global map_pass_alm

def map_data_with_general_mappings(data, mapping_principaux, key_vars_dic):
    cles_data_ntx = {1: [pa.NC_PA_BILAN, key_vars_dic["CONTRACT_TYPE"]],
                     2: [pa.NC_PA_BILAN, key_vars_dic["CONTRACT_TYPE"], key_vars_dic["MATUR"] + "_TEMP"],
                     3: [key_vars_dic["GESTION"] + "_TEMP"],
                     4: [key_vars_dic["PALIER"] + "_TEMP"]}

    mappings = {1: "CONTRATS", 2: "MTY", 3: "GESTION", 4: "PALIER"}

    for i in range(1, len(mappings) + 1):
        data = mp.map_data(data, mapping_principaux[mappings[i]], keys_data=cles_data_ntx[i],
                        name_mapping="STOCK DATA vs.")

    data.drop([key_vars_dic["CONTRACT_TYPE"], key_vars_dic["MATUR"] + "_TEMP",
                key_vars_dic["GESTION"] + "_TEMP", key_vars_dic["PALIER"] + "_TEMP"],
              axis=1, inplace=True)

    return data


def format_paliers(data, palier_col):
    invalid_palier = (data[palier_col].isnull()) | (data[palier_col].astype(str) == "nan") \
                     | (data[palier_col] == " ") | (data[palier_col] == "(vide)")

    data[palier_col] = data[palier_col].mask(invalid_palier, "-")

    data = gu.force_integer_to_string(data, palier_col)

    return data
