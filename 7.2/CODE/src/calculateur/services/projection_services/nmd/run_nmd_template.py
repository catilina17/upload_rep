from calculateur.data_transformers.data_in.nmd.class_nmd_templates import Data_NMD_TEMPLATES
from calculateur.models.data_manager.data_format_manager.class_fields_manager import Data_Fields
from calculateur.models.data_manager.data_format_manager.class_data_formater import Data_Formater
from mappings.pass_alm_fields import PASS_ALM_Fields
from mappings import general_mappings as mp
import numpy as np
from utils import general_utils as gu
import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def run_nmd_template_getter(nmd_sources, etab, dar_usr, format_data=True, save=True,
                            rco_allocation_key = True):
    cls_fields = Data_Fields()
    cls_format = Data_Formater(cls_fields)
    cls_pa_fields = PASS_ALM_Fields()

    if not "DATA" in nmd_sources["PN"]["LDP"]:
        nmd_sources["PN"]["LDP"]["DATA"] = pd.read_csv( nmd_sources["PN"]["LDP"]["CHEMIN"],
                                                        delimiter= nmd_sources["PN"]["LDP"]["DELIMITER"],
                                                        decimal= nmd_sources["PN"]["LDP"]["DECIMAL"],
                                                        engine='python', encoding="ISO-8859-1")

    if format_data:
        data_nmd_fm = format_data_pn(nmd_sources["PN"]["LDP"]["DATA"].copy(), cls_pa_fields, cls_fields)
    else:
        data_nmd_fm =  nmd_sources["PN"]["LDP"]["DATA"].copy()

    cls_tmp_nmd = Data_NMD_TEMPLATES(nmd_sources["STOCK"], nmd_sources["MODELS"]["NMD"]["DATA"],
                                     etab, dar_usr, cls_fields, cls_format)

    data_template = cls_tmp_nmd.get_templates(data_nmd_fm, rco_allocation_key=rco_allocation_key)
    cls_tmp_nmd.data_template = data_template

    data_template_mapped = map_with_pass_alm_data(data_template, cls_fields, cls_pa_fields)

    if save :
        nom_fichier = etab + "_NMD_%s_TEMPLATES.csv" % dar_usr.strftime("%Y-%m-%d")
        output_path = os.path.join("\\".join(nmd_sources["STOCK"]["LDP"]["CHEMIN"].split("\\")[:-1]), nom_fichier)
        data_template_mapped.to_csv(output_path, sep=";", decimal=",", index=False)

    cls_tmp_nmd.data_template_mapped = data_template_mapped

    return cls_tmp_nmd


def format_data_pn(data_nmd_pn, cls_pa_fields, cls_fields):
    data_nmd_pn["RATE_CAT"] = np.where(data_nmd_pn[cls_pa_fields.NC_PA_RATE_CODE].str.contains("FIXE"), "FIXED",
                                       "FLOATING")

    join_keys = [cls_pa_fields.NC_PA_ETAB, cls_pa_fields.NC_PA_DEVISE, cls_pa_fields.NC_PA_CONTRACT_TYPE, cls_pa_fields.NC_PA_MARCHE, "RATE_CAT"]

    cols_cible = [cls_fields.NC_LDP_ETAB, cls_fields.NC_LDP_CURRENCY, cls_fields.NC_LDP_CONTRACT_TYPE,
                  cls_fields.NC_LDP_MARCHE, cls_fields.NC_LDP_RATE_TYPE]

    data_nmd_pn = data_nmd_pn.rename(columns={x: y for x, y in zip(join_keys, cols_cible)})
    data_nmd_pn = data_nmd_pn[cols_cible].copy().drop_duplicates()

    return data_nmd_pn

def map_with_pass_alm_data(data_template, cls_fields, cls_pa_fields):
    data_template = format_bilan(data_template, cls_fields)

    data_template[cls_pa_fields.NC_PA_MARCHE] = data_template[cls_fields.NC_LDP_MARCHE].values
    data_template[cls_pa_fields.NC_PA_RATE_CODE] = data_template[cls_fields.NC_LDP_RATE_CODE].values
    data_template[cls_pa_fields.NC_PA_DEVISE] = data_template[cls_fields.NC_LDP_CURRENCY].values

    data_template[cls_fields.NC_LDP_PALIER + "_MOD"] = data_template[cls_fields.NC_LDP_PALIER].fillna("-")

    data_template = gu.force_integer_to_string(data_template, cls_fields.NC_LDP_PALIER + "_MOD")

    cles_data = {1: [cls_pa_fields.NC_PA_BILAN, cls_fields.NC_LDP_CONTRACT_TYPE],
                 2: [cls_fields.NC_LDP_PALIER + "_MOD"]}

    mappings = {1: "CONTRATS", 2: "PALIER"}

    for i in range(1, 3):
        map = mp.map_pass_alm[mappings[i]]["TABLE"]
        cols_mapp = mp.map_pass_alm[mappings[i]]["OUT"]
        if i == 2:
            data_template = gu.map_data(data_template, map, keys_data=cles_data[i],
                                        name_mapping="PN DATA vs.", cols_mapp=cols_mapp, map_null_values=True,
                                        no_map_value="#Mapping")
        else:
            data_template = gu.map_data(data_template, map, keys_data=cles_data[i],
                                        name_mapping="PN DATA vs.", cols_mapp=cols_mapp)

    data_template = data_template.drop([cls_fields.NC_LDP_PALIER + "_MOD"], axis=1)

    return data_template

def format_bilan(data, cls_fields):
    filtres = [data[cls_fields.NC_LDP_CONTRACT_TYPE].str[:2] == "A-",
               data[cls_fields.NC_LDP_CONTRACT_TYPE].str[:2] == "P-", \
               (data[cls_fields.NC_LDP_CONTRACT_TYPE].str[-2:] == "-A") & (
                       data[cls_fields.NC_LDP_CONTRACT_TYPE].str[:2] == "HB"),
               (data[cls_fields.NC_LDP_CONTRACT_TYPE].str[-2:] == "-P") & (
                       data[cls_fields.NC_LDP_CONTRACT_TYPE].str[:2] == "HB")]
    choices = ["B ACTIF", "B PASSIF", "HB ACTIF", "HB PASSIF"]
    data["BILAN"] = np.select(filtres, choices)

    return data
