import utils.excel_utils as ex
import modules.scenario.services.pn_services.pn_bpce_services.referential_bpce as rf_bpce
import pandas as pd
import numpy as np
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import modules.scenario.services.pn_services.pn_bpce_services.mapping_service as mapp
import modules.scenario.services.referential_file_service as rf
from modules.scenario.services.referential_file_service import get_pn_df
from modules.scenario.referentials.transco_pn import transco_pn_name_to_range_name
import logging

logger = logging.getLogger(__name__)

col_type_ajout = "TYPE AJOUT"

def ajout_pn_scenario(export_wb, pn_type, scenario_stress_pn, mapping_wb, bassin):
    pn_data = scenario_stress_pn.copy()
    pn_data = pn_data[pn_data[pa.NC_PA_BASSIN] == bassin].copy()
    diviseur_rows = 3 if "NMD" in pn_type else 4
    if len(pn_data) // diviseur_rows != len(pn_data) / diviseur_rows:
        msg_err = ("Le nombre de lignes de PNs ajoutées pour le type %s semble incohérent,"
                   " il doit être un multiple de %s") % (pn_type, diviseur_rows)
        logger.error(msg_err)
        raise ValueError(msg_err)

    old_pn_df = get_pn_df(export_wb, transco_pn_name_to_range_name[pn_type], bassin)
    rf.pn_df[pn_type] = old_pn_df.copy()

    if pn_data.shape[0] > 0:
        mapp.load_general_mappings(mapping_wb)
        pn_data = add_missing_key_cols(pn_data, bassin)
        pn_data = map_pn(pn_data)

        if "NMD" in pn_type:
            is_remplacement = (pn_data[col_type_ajout] == "REMPLACEMENT").values &  pn_data[pa.NC_PA_CLE].isin(old_pn_df.reset_index()[pa.NC_PA_CLE].unique().tolist())
            is_ajout = ~is_remplacement
        else:
            is_remplacement = ((pn_data[col_type_ajout] == "REMPLACEMENT").values
                               &  pn_data.set_index([pa.NC_PA_CLE] + pa.NC_PA_COL_SPEC_ECH).index.isin(old_pn_df.reset_index().set_index([pa.NC_PA_CLE] + pa.NC_PA_COL_SPEC_ECH).index.values.tolist()))
            is_ajout = ~is_remplacement

        pn_data = final_format(pn_data, pn_type)

        pn_data_ajout = pn_data[is_ajout].copy()
        if "NMD" in pn_type:
            pn_data_remplacement = pn_data[is_remplacement].copy().set_index([pa.NC_PA_CLE, pa.NC_PA_IND03])
            old_pn_df = old_pn_df.reset_index().set_index([pa.NC_PA_CLE, pa.NC_PA_IND03])
        else:
            pn_data_remplacement = pn_data[is_remplacement].copy().set_index([pa.NC_PA_CLE] + pa.NC_PA_COL_SPEC_ECH + [pa.NC_PA_IND03])
            old_pn_df = old_pn_df.reset_index().set_index([pa.NC_PA_CLE] + pa.NC_PA_COL_SPEC_ECH + [pa.NC_PA_IND03])

        if "NMD" in pn_type:
            pn_data_ajout = verify_uniqueness(pn_data_ajout, old_pn_df, bassin)

        n = old_pn_df.shape[0]
        old_pn_df[pa.NC_PA_COL_SORTIE_NUM_PN] = np.where(old_pn_df.index.isin(pn_data_remplacement.index.values.tolist()).reshape(n, 1),
                                                         np.nan, old_pn_df[pa.NC_PA_COL_SORTIE_NUM_PN])

        old_pn_df.update(pn_data_remplacement[pa.NC_PA_COL_SORTIE_NUM_PN].astype(str).replace("", "nan").astype(np.float64))

        new_pn_data = pd.concat([old_pn_df.reset_index(), pn_data_ajout], ignore_index=False)

        new_pn_data = final_format(new_pn_data, pn_type)

        ex.clear_range_content(export_wb, transco_pn_name_to_range_name[pn_type], offset = 2)
        ex.write_df_to_range_name2(new_pn_data, transco_pn_name_to_range_name[pn_type],
                                   export_wb, header=False, offset=2, end_xldown=False)


def verify_uniqueness(pn_data, old_pn_df, bassin):
    if pn_data[pa.NC_PA_CLE].isin(old_pn_df.reset_index()[pa.NC_PA_CLE].values.tolist()).any():
        listo = pn_data[pn_data[pa.NC_PA_CLE].isin(old_pn_df.reset_index()[pa.NC_PA_CLE].values.tolist())][
            pa.NC_PA_CLE].unique().tolist()
        msg_err = ("      Les clés suivantes existent déjà dans les NMDs de %s."
                   "      Elles ne seront pas ajoutées. Pensez à remplacer plutôt les PNs existantes") % (bassin)
        logger.warning(msg_err)
        logger.warning("      %s" % listo)
        pn_data = pn_data[~pn_data[pa.NC_PA_CLE].isin(listo)].copy()

    pn_data["key"] = pn_data[pa.NC_PA_CLE] + pn_data[pa.NC_PA_IND03]
    if len(pn_data["key"]) != len(pn_data["key"].unique().tolist()):
        listo = pn_data[pn_data[["key"]].duplicated()]["key"].unique().tolist()
        msg_err = "      Vous avez des doublons dans les NMDs de %s que vous avez ajoutées. Seul le premier contrat sera conservé" % (
            bassin)
        logger.warning(msg_err)
        logger.warning("%s" % listo)
        pn_data = pn_data.drop_duplicates(subset=["key"]).copy().drop(["key"], axis=1)

    return pn_data


def add_missing_key_cols(data, bassin):
    data[pa.NC_PA_SCOPE] = rf_bpce.SCOPE_MNI_LIQUIDITE
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


def map_pn(pn_data):
    filtres = [pn_data[pa.NC_PA_CONTRACT_TYPE].str[0:2] == "A-",
               pn_data[pa.NC_PA_CONTRACT_TYPE].str[0:2] == "P-",
               (pn_data[pa.NC_PA_CONTRACT_TYPE].str[-2:] == "-A") | (pn_data[pa.NC_PA_CONTRACT_TYPE].str[-5:] == "-A-KN") | (pn_data[pa.NC_PA_CONTRACT_TYPE].str[-5:] == "-A-OC")
               | (pn_data[pa.NC_PA_CONTRACT_TYPE].str[:3] == "AHB"),
               (pn_data[pa.NC_PA_CONTRACT_TYPE].str[-2:] == "-P") | (pn_data[pa.NC_PA_CONTRACT_TYPE].str[-5:] == "-P-KN") | (pn_data[pa.NC_PA_CONTRACT_TYPE].str[-5:] == "-P-OC")
               | (pn_data[pa.NC_PA_CONTRACT_TYPE].str[:3] == "PHB")]

    choices = ["B ACTIF", "B PASSIF", "HB ACTIF", "HB PASSIF"]
    pn_data[pa.NC_PA_BILAN] = np.select(filtres, choices)

    pn_data = mapp.mapping_consolidation_liquidite(pn_data)

    pn_data = mapp.map_data(pn_data, mapp.mapping["mapping_global"]["CONTRATS"], \
                            keys_data=[pa.NC_PA_CONTRACT_TYPE], name_mapping="PN DATA vs.")

    pn_data = mapp.map_data(pn_data, mapp.mapping["mapping_global"]["INDEX_AGREG"], \
                            keys_data=[pa.NC_PA_RATE_CODE], name_mapping="PN DATA vs.")

    return pn_data


def final_format(pn_data, type_pn):
    if "ECH" in type_pn:
        col_sortie = pa.NC_PA_COL_SORTIE_QUAL_ECH + pa.NC_PA_COL_SORTIE_NUM_PN
    else:
        col_sortie = pa.NC_PA_COL_SORTIE_QUAL + pa.NC_PA_COL_SORTIE_NUM_PN

    missing_indics = [x for x \
                      in col_sortie \
                      if not x in pn_data.columns.tolist()]

    pn_ech = pd.concat([pn_data, pd.DataFrame([[""] * len(missing_indics)], \
                                              index=pn_data.index, columns=missing_indics)], axis=1)

    pn_ech = pn_ech[col_sortie].copy()

    pn_ech[pa.NC_PA_COL_SORTIE_NUM_PN] = pn_ech[pa.NC_PA_COL_SORTIE_NUM_PN].fillna("")

    return pn_ech