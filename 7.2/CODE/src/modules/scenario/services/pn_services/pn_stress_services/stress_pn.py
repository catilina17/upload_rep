import logging
import numpy as np
import pandas as pd
from modules.scenario.parameters.general_parameters import *
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
from services.rate_services.params import tx_referential as tx_ref
from modules.scenario.utils import paths_resolver as pr

RN_VAL_STRESS = 'VAL. STRESS'

RN_TYPE_DE_STRESS = 'TYPE DE STRESS'

logger = logging.getLogger(__name__)


def process_scenarios_stress(pn_output_path, pn_type, scenario_stress):
    if "NMD" in pn_type:
        if "%" in pn_type:
            file_substring = "NMD_BC"
            no_files_substring = [""]
        else:
            file_substring = "NMD"
            no_files_substring = ["NMD_BC", "NMD_CALAGE"]
    else:
        if "%" in pn_type:
            file_substring = "PN_ECH_BC"
            no_files_substring = [""]
        else:
            file_substring = "PN_ECH"
            no_files_substring = ["PN_ECH_BC"]

    pn_file_path = pr._get_file_path(pn_output_path, file_substring=file_substring, no_files_substring=no_files_substring)

    pn_df = pd.read_csv(pn_file_path, sep=";", decimal=",")
    if pn_df.empty:
        logger.info('    Pas de contrats de type {} dans le scénario'.format(pn_type))
    pn_df = pn_df.set_index([x for x in pn_df.columns if x not in pa.NC_PA_COL_SORTIE_NUM_PN and x!= "IND03"])
    pn_indicators = scenario_stress[CN_PN_INDICATOR].unique()
    for pn_indicator in pn_indicators:
        pn_df = stress_indicator(pn_df, scenario_stress[scenario_stress[CN_PN_INDICATOR] == pn_indicator],
                                                     pn_type, pn_indicator)

    pn_df = pn_df.reset_index()
    pn_df.to_csv(pn_file_path, sep=";", decimal=",", index=False)


def stress_indicator(pn_df, scenario_stress, pn_type, pn_indicator):
    num_cols = [x for x in pn_df.columns if x in pa.NC_PA_COL_SORTIE_NUM_PN]
    pn_df[num_cols] = pn_df[num_cols].astype(np.float64)
    pn_df_copy = pn_df.copy()
    j = 0
    filter_cond_cum = np.array([False] * pn_df.shape[0])
    for i, row in scenario_stress[::-1].iterrows():
        stressed_pn_df = pn_df.copy()
        try:
            filter_condition, debut, fin = get_filter_condition(pn_df, row[pa.NC_PA_ETAB],
                                                                row[pa.NC_PA_DEVISE],
                                                                row[pa.NC_PA_RATE_CODE], row[pa.NC_PA_CONTRACT_TYPE],
                                                                row[pa.NC_PA_POSTE], row[pa.NC_PA_PERIMETRE],
                                                                row[pa.NC_PA_METIER], row[pa.NC_PA_SOUS_METIER],
                                                                row[pa.NC_PA_ZONE_GEO], row[pa.NC_PA_SOUS_ZONE_GEO],
                                                                row[tx_ref.CN_DATE_DEBUT], row[tx_ref.CN_DATE_FIN_FR]
                                                                )
            filter_cond_cum = filter_condition | filter_cond_cum
            if True in filter_condition:
                stressed_pn_df = apply_stress_line(stressed_pn_df, pn_indicator, filter_condition, debut, fin,
                                                   row[RN_TYPE_DE_STRESS],
                                                   row[RN_VAL_STRESS], pn_type)
        except Exception as e:
            logger.error(
                'Erreur sur la ligne de stress PN:  {}     {}    {}    {}     {}     {}    {}'.format(pn_indicator,
                                                                                                      *list(row)))
            logger.error(e, exc_info=True)
            logger.info('Le programme poursuit l\'application des autres lignes de stress PN')
        if j == 0:
            pn_df_copy = stressed_pn_df.copy()
            j = +1
        else:
            pn_df_copy[num_cols] = pn_df_copy[num_cols].mask(((pn_df_copy[num_cols] == pn_df[num_cols])
                                                              | (np.isnan(pn_df_copy[num_cols]) & np.isnan(pn_df[num_cols]))),
                                                             stressed_pn_df[num_cols])

    if "%" in pn_type:
        pn_df_copy = pn_df_copy[filter_cond_cum].copy()

    return pn_df_copy


def apply_stress_line(stressed_pn_df, pn_indicator, filter_condition, debut, fin, type_de_stress, delta, pn_type):
    indic = pa.NC_PA_DEM if (pn_type in ["PN ECH"] and pn_indicator == "Encours") else (pa.NC_PA_DEM_CIBLE if pn_indicator == "Encours" else pn_indicator)
    filter_ind = (stressed_pn_df["IND03"] == indic).values
    if type_de_stress == STRESS_TYPE_REMPLACEMENT:
        stressed_pn_df.loc[filter_condition & filter_ind, debut: fin] = delta
    if type_de_stress == STRESS_TYPE_CUMULATIF:
        if not '%' in pn_type:
            stressed_pn_df.loc[filter_condition & filter_ind, debut: fin] = \
                stressed_pn_df.loc[filter_ind & filter_condition, debut: fin].values * (1 + delta / 100)
        else:
            stressed_pn_df.loc[filter_ind & filter_condition, debut: fin] = delta / 100
    if type_de_stress == STRESS_TYPE_ADDITIF:
        stressed_pn_df.loc[filter_ind & filter_condition, debut: fin] = stressed_pn_df.loc[
                                                                        filter_ind & filter_condition,
                                                                        debut: fin].values + delta
    return stressed_pn_df


def get_filter_condition(pn_df, etab, devise, index_calc, contrat, poste, perimetre, metier, \
                         ss_metier, zone_geo, ss_zone_geo, debut, fin):
    filter_condition = np.array([True for x in range(len(pn_df.index))])
    if etab != SELECT_ALL_KEYWORD:
        filter_condition = filter_condition & (pn_df.index.get_level_values(pa.NC_PA_ETAB) == etab)
    if index_calc != SELECT_ALL_KEYWORD:
        filter_condition = filter_condition & (pn_df.index.get_level_values(pa.NC_PA_RATE_CODE) == index_calc)
    if devise != SELECT_ALL_KEYWORD:
        filter_condition = filter_condition & (pn_df.index.get_level_values(pa.NC_PA_DEVISE) == devise)
    if contrat != SELECT_ALL_KEYWORD:
        filter_condition = filter_condition & (pn_df.index.get_level_values(pa.NC_PA_CONTRACT_TYPE) == contrat)
    if poste != SELECT_ALL_KEYWORD:
        filter_condition = filter_condition & (pn_df.index.get_level_values(pa.NC_PA_POSTE) == poste)
    if perimetre != SELECT_ALL_KEYWORD:
        filter_condition = filter_condition & (pn_df.index.get_level_values(pa.NC_PA_PERIMETRE) == perimetre)
    if metier != SELECT_ALL_KEYWORD:
        filter_condition = filter_condition & (pn_df.index.get_level_values(pa.NC_PA_METIER) == metier)
    if ss_metier != SELECT_ALL_KEYWORD:
        filter_condition = filter_condition & (pn_df.index.get_level_values(pa.NC_PA_SOUS_METIER) == ss_metier)
    if zone_geo != SELECT_ALL_KEYWORD:
        filter_condition = filter_condition & (pn_df.index.get_level_values(pa.NC_PA_ZONE_GEO) == zone_geo)
    if ss_zone_geo != SELECT_ALL_KEYWORD:
        filter_condition = filter_condition & (pn_df.index.get_level_values(pa.NC_PA_SOUS_ZONE_GEO) == ss_zone_geo)

    return filter_condition, "M" + str(int(debut)), "M" + str(int(fin))
