import utils.general_utils as gu
import modules.alim.parameters.general_parameters as gp
import modules.alim.parameters.user_parameters as up
import modules.alim.parameters.NTX_SEF_params as ntx_p
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import numpy as np
import pandas as pd
import mappings.general_mappings as gma
from modules.alim.rates_service.rates_module import RatesManager
import logging

logger = logging.getLogger(__name__)
global missing_mapping
missing_mapping = {}


def map_data(data_to_map, mapping, keys_data=[], keys_mapping=[], cols_mapp=[], override=False, name_mapping="", \
             error_mapping=True, no_map_value=None, allow_duplicates=False, option="PASS_ALM", join_how="left",
             col_err=[], select_inner=False, except_name_mapping=""):
    global missing_mapping

    len_data = data_to_map.shape[0]

    if no_map_value is None:
        no_map_value = gp.empty_mapping

    if option == "PASS_ALM":
        if cols_mapp == []:
            cols_mapp = mapping["OUT"]
        mapping_tab = mapping["TABLE"]
        name_mapping = name_mapping + " " + mapping["FULL_NAME"]
        est_facultatif =  mapping["est_facultatif"]
    else:
        mapping_tab = mapping.copy()
        est_facultatif =  False
        if keys_mapping != []:
            mapping_tab = gu.strip_and_upper(mapping_tab, keys_mapping)
            mapping_tab = mapping_tab.set_index(keys_mapping)
        if cols_mapp == []:
            cols_mapp = mapping.columns.tolist()

    original_keys = data_to_map[keys_data].copy().reset_index(drop=True)
    data_to_map["TMP_INDX"] = np.arange(0, data_to_map.shape[0])
    data_to_map = gu.strip_and_upper(data_to_map, keys_data)

    if override:
        old_index = data_to_map.index.copy()
        data_to_map = data_to_map.reset_index(drop=True)
        new_data = data_to_map[keys_data].copy()
        new_data = new_data.join(mapping_tab[cols_mapp], how="left", on=keys_data)
        if new_data.shape[0] != data_to_map.shape[0]:
            raise ValueError("Impossible to override, mappings " + name_mapping + " is not unique")
        data_to_map.update(new_data[cols_mapp].copy())
        data_to_map.index = old_index
    else:
        data_to_map = data_to_map.join(mapping_tab[cols_mapp], how=join_how, on=keys_data)

    data_to_map = data_to_map.join(original_keys, on=["TMP_INDX"], rsuffix="_OLD")
    data_to_map[keys_data] = data_to_map[[x + "_OLD" for x in keys_data]]
    data_to_map = data_to_map.drop(columns=[x + "_OLD" for x in keys_data] + ["TMP_INDX"], axis=1)

    filtero_none = (data_to_map[cols_mapp[0]].isnull()) | (data_to_map[cols_mapp[0]] == np.nan)
    if filtero_none.any():
        data_to_map.loc[filtero_none, cols_mapp] = no_map_value

    if error_mapping:
        if filtero_none.any():
            if col_err == []:
                data_err = data_to_map.loc[filtero_none, keys_data + cols_mapp].drop_duplicates()
            else:
                data_err = data_to_map.loc[filtero_none, keys_data + col_err + cols_mapp] \
                    .groupby(by=keys_data + cols_mapp, as_index=False).sum()
            if except_name_mapping != "":
                data_err = gma.filter_out_errors(data_err, except_name_mapping)

            if data_err.shape[0] > 0:
                if not name_mapping in missing_mapping:
                    missing_mapping[name_mapping] = (data_err, est_facultatif)
                else:
                    missing_mapping[name_mapping] = (pd.concat([missing_mapping[name_mapping][0], data_err]), \
                                                     est_facultatif)

    if not allow_duplicates and len_data != data_to_map.shape[0] and join_how == "left":
        logger.warning("   THERE ARE DUPLICATES WITH MAPPING: " + name_mapping)

    if select_inner:
        data_to_map = data_to_map[~filtero_none]

    return data_to_map


def map_data_with_bpce_mappings(data):
    data = data.drop(columns=[pa.NC_PA_PERIMETRE], axis=1)
    data = map_data(data, gma.mapping_bpce["PERIMETRE_BPCE"], keys_data=[pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_BOOK, pa.NC_PA_PALIER], \
                    name_mapping="STOCK DATA vs.")
    return data


def map_data_with_ntx_mappings(data):
    perim_NTX = ntx_p.perim_ntx
    if not perim_NTX in gma.mapping_NTX["MNI"]["TABLE"].columns:
        raise ValueError(
            "Le périmètre souhaité %s n'est pas dans le mappings %s" % (perim_NTX, gma.mapping_NTX["MNI"]["FULL_NAME"]))

    data[pa.NC_PA_TOP_MNI] = data[pa.NC_PA_TOP_MNI].str.replace("_", "")
    data = map_data(data, gma.mapping_NTX["MNI"], keys_data=[pa.NC_PA_TOP_MNI], \
                    name_mapping="STOCK DATA vs.", cols_mapp=[perim_NTX])
    data = gu.strip_and_upper(data, [perim_NTX])

    MNI_to_del = (data[pa.NC_PA_SCOPE] == gp.MNI) & (data[perim_NTX] != "OUI")
    LIQ_to_del = (data[pa.NC_PA_SCOPE] == gp.LIQ) & (data[ntx_p.NC_NTX_DATA_FLAG_LIQ_GAP] != "O")

    data = data[~((MNI_to_del) | (LIQ_to_del))].copy()

    MNI = (data[pa.NC_PA_SCOPE] == gp.MNI_AND_LIQ) & (
            data[perim_NTX] == "OUI")
    LIQ = (data[pa.NC_PA_SCOPE] == gp.MNI_AND_LIQ) & (data[ntx_p.NC_NTX_DATA_FLAG_LIQ_GAP] == "O")

    data.loc[MNI & LIQ, pa.NC_PA_SCOPE] = gp.MNI_AND_LIQ
    data.loc[MNI & (~LIQ), pa.NC_PA_SCOPE] = gp.MNI
    data.loc[(~MNI) & LIQ, pa.NC_PA_SCOPE] = gp.LIQ
    data = data[~((data[pa.NC_PA_SCOPE] == gp.MNI_AND_LIQ) & (~MNI) & (~LIQ))].copy()

    num_cols = [x for x in data.columns if ("TEF_M" in x) or ("LMN_M" in x) or ("TEM_M" in x) or ("LMN EVE_M" in x)]
    data.loc[data[pa.NC_PA_SCOPE] == gp.LIQ, num_cols] = 0

    # Mapping du traitement prudentiel, sous-métier
    data.drop(columns=[pa.NC_PA_SOUS_METIER, pa.NC_PA_ZONE_GEO, pa.NC_PA_METIER, pa.NC_PA_SOUS_ZONE_GEO], axis=1, inplace=True)
    data = map_data(data, gma.mapping_NTX["OTHER"], keys_data=[pa.NC_PA_BOOK], \
                    name_mapping=" STOCK DATA vs.")
    data[pa.NC_PA_SCOPE] = data[pa.NC_PA_SCOPE] + "_" + data[ntx_p.NC_MAP_NTX_TRAITEMENT_REG_ALM]

    data.drop([ntx_p.NC_MAP_NTX_TRAITEMENT_REG_ALM, perim_NTX, ntx_p.NC_NTX_DATA_FLAG_LIQ_GAP], axis=1, inplace=True)

    #Mapping du rate code
    sc_curves_df = RatesManager.get_sc_df(up.sc_ref_nmd)
    data_key = (data[pa.NC_PA_DEVISE] + data[pa.NC_PA_RATE_CODE]).str.upper()
    map_key = (sc_curves_df["DEVISE"] + sc_curves_df["CODE PASS_ALM"]).str.upper()
    unknown_rate_code = ~((data_key).isin((map_key).values.tolist()))
    unknown_rate_code = unknown_rate_code & (data[pa.NC_PA_RATE_CODE] != "FIXE")
    data_map = data[unknown_rate_code].copy()
    data_map = data_map.rename(columns={pa.NC_PA_RATE_CODE: pa.NC_PA_RATE_CODE + "_TMP", pa.NC_PA_DEVISE: pa.NC_PA_DEVISE + "_TMP"})
    data_map = map_data(data_map, gma.mapping_NTX["RATE CODE"], keys_data=[pa.NC_PA_DEVISE + "_TMP",
                    pa.NC_PA_RATE_CODE + "_TMP"], name_mapping=" STOCK DATA vs.")
    data.update(data_map[[pa.NC_PA_DEVISE, pa.NC_PA_RATE_CODE]].copy())

    data_key = (data[pa.NC_PA_DEVISE] + data[pa.NC_PA_RATE_CODE]).str.upper()
    map_key = (sc_curves_df["DEVISE"] + sc_curves_df["CODE PASS_ALM"]).str.upper()
    unknown_rate_code = ~((data_key).isin((map_key).values.tolist()))
    unknown_rate_code = unknown_rate_code & (data[pa.NC_PA_RATE_CODE] != "FIXE")
    data_unknown = data.loc[unknown_rate_code, [pa.NC_PA_DEVISE ,pa.NC_PA_RATE_CODE]].drop_duplicates().values.tolist()
    if len(data_unknown) > 0:
        logger.error("Certains DEVISE/RATE_CODE présents dans le mapping %s n'existent pas dans le RATE INPUT : %s" % (gma.mapping_NTX["RATE CODE"]["FULL_NAME"],
                                                                                                                       data_unknown))
    return data


def map_data_with_general_mappings(data):
    cles_data_ntx = {1: [pa.NC_PA_BILAN, gp.NC_CONTRACT_TEMP], 2: [gp.NC_MATUR_TEMP], \
                     3: [pa.NC_PA_BILAN, gp.NC_CONTRACT_TEMP, gp.NC_MATUR_TEMP], \
                     4: [pa.NC_PA_RATE_CODE], 5: [gp.NC_GESTION_TEMP], 6: [gp.NC_PALIER_TEMP]}

    mappings = {1: "CONTRATS", 2: "MTY DETAILED", 3: "MTY", 4: "INDEX_AGREG", \
                5: "GESTION", 6: "PALIER"}

    for i in range(1, len(mappings) + 1):
        if (not (up.current_etab in gp.MTY_DETAILED_ETAB and i == 3)) \
                and (not (up.current_etab not in gp.MTY_DETAILED_ETAB and i == 2)):
            data = map_data(data, gma.map_pass_alm[mappings[i]], keys_data=cles_data_ntx[i],
                            name_mapping="STOCK DATA vs.")

    data.drop([gp.NC_CONTRACT_TEMP, gp.NC_MATUR_TEMP,
               gp.NC_GESTION_TEMP, gp.NC_PALIER_TEMP], axis=1, inplace=True)

    return data


def mapping_consolidation_liquidite(data):
    keys_liq_BC = [pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_BOOK, pa.NC_PA_LCR_TIERS]
    keys_EM = [pa.NC_PA_BILAN, pa.NC_PA_Regroupement_1, pa.NC_PA_MATUR]

    """ MAPPING LIQUIDITE BILAN CASH """
    cles_a_combiner = keys_liq_BC
    mapping = gma.mapping_liquidite["LIQ_BC"]
    data = gu.gen_combined_key_col(data, mapping["TABLE"], cols_key=cles_a_combiner, symbol_any="-",
                                   name_col_key="CONTRAT_", set_index=False)
    data = map_data(data, mapping, keys_data=["CONTRAT_"], name_mapping="STOCK/PN DATA vs.")
    data = data.drop(["CONTRAT_"], axis=1)

    """ AUTRES mappings LIQUIDITE """
    keys_liq_IG = [pa.NC_PA_BILAN, pa.NC_PA_Bilan_Cash, pa.NC_PA_BASSIN, pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_PALIER]
    keys_liq_CT = [pa.NC_PA_BILAN, pa.NC_PA_Regroupement_1, pa.NC_PA_BASSIN, pa.NC_PA_PALIER]
    keys_liq_FI = [pa.NC_PA_BILAN, pa.NC_PA_Regroupement_1, pa.NC_PA_BASSIN, pa.NC_PA_Bilan_Cash, pa.NC_PA_CONTRACT_TYPE,
                   "IG/HG Social"]
    keys_liq_SC = [pa.NC_PA_Affectation_Social]
    data["IG/HG Social"] = np.where(data[pa.NC_PA_PALIER] == "-", "HG", "IG")

    keys_data = [keys_EM, keys_liq_IG, keys_liq_CT, keys_liq_FI, keys_liq_SC]
    mappings = ["LIQ_EM", "LIQ_IG", "LIQ_CT", "LIQ_FI", "LIQ_SC"]
    for i in range(0, len(mappings)):
        mapping = gma.mapping_liquidite[mappings[i]]
        key_data = keys_data[i]
        override = False if mappings[i] == "LIQ_SC" else True
        error_mapping = True if mappings[i] == "LIQ_SC" else False
        data = map_data(data, mapping, keys_data=key_data, override=override, error_mapping=error_mapping,
                        name_mapping="STOCK/PN DATA vs.")

    """ MAPPING NSFR """
    cles_a_combiner = [pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_LCR_TIERS]
    mapping = gma.mapping_liquidite["NSFR"]
    data = gu.gen_combined_key_col(data, mapping["TABLE"], cols_key=cles_a_combiner, symbol_any="*",
                                   name_col_key="CONTRAT_", set_index=False)
    data = map_data(data, mapping, keys_data=["CONTRAT_"], name_mapping="STOCK/PN DATA vs.")
    data = data.drop(["CONTRAT_", "IG/HG Social"], axis=1)

    return data


def map_intra_groupes(data):
    check = (data[pa.NC_PA_PALIER] != gp.empty_mapping) & (data[pa.NC_PA_CONTRACT_TYPE] != gp.empty_mapping)

    data = map_data(data, gma.mapping_IG["PALIER"], keys_data=[pa.NC_PA_PALIER], \
                    name_mapping="STOCK DATA vs. ")

    data = map_data(data, gma.mapping_IG["IG"], keys_data=[pa.NC_PA_CONTRACT_TYPE], \
                    name_mapping="STOCK DATA vs. ")

    if up.current_etab == "BPCE":
        check = check & (data["BASSIN IG"] == "BPCE") & (data["isBPCEIG"] == "")
    elif up.current_etab in gp.NTX_FORMAT:
        check = check & (data["BASSIN IG"] == "NTX") & (data["isNTXIG"] == "")
    else:
        check = check & (data["BASSIN IG"] == "BPCE") & (data["isRZOIG"] == "")

    if sum(check) > 0:
        logger.warning(
            "PROBLEME SUR LES mappings DES CONTREPARTIES INTRAGROUPES: Veuillez vérifier les mappings manquants")

    data.drop(["BASSIN IG", "isNTXIG", "isBPCEIG", "isRZOIG"], axis=1, inplace=True)

    return data
