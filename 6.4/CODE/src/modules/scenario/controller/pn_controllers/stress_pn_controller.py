import logging
import numpy as np
from modules.scenario.referentials.general_parameters import *
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
from modules.scenario.referentials.transco_pn import transco_pn_name_to_range_name
import modules.scenario.services.referential_file_service as rf
from modules.scenario.services.referential_file_service import get_pn_df
from utils.general_utils import to_nan_np
import pandas as pd
from modules.scenario.rate_services import tx_referential as tx_ref

RN_VAL_STRESS = 'VAL. STRESS'

RN_TYPE_DE_STRESS = 'TYPE DE STRESS'

logger = logging.getLogger(__name__)


def replace_tx_spn_margeco_na_zero(pn_df):
    filtre = pn_df["IND03"].isin([pa.NC_PA_MG_CO, pa.NC_PA_TX_SP])
    cols = ["M" + str(i) for i in range(0, tx_ref.NB_PROJ_TAUX + 1)]
    pn_df = to_nan_np(pn_df, cols, filtre=filtre)
    return pn_df


def stress_scenario_pn(input_wb, pn_type, scenario_stress, etab, option=""):
    if False:#pn_type in rf.pn_df:
        pn_df = rf.pn_df[pn_type].copy()
    else:
        pn_df = get_pn_df(input_wb, transco_pn_name_to_range_name[pn_type], etab)
        rf.pn_df[pn_type] = pn_df.copy()
    non_applied_stress = pd.DataFrame(columns=scenario_stress.columns)
    if pn_df.empty:
        logger.info('    Pas de contrats de type {} dans le scénario'.format(pn_type))
        return pn_df, non_applied_stress

    pn_df = replace_tx_spn_margeco_na_zero(pn_df)
    pn_indicators = scenario_stress[CN_PN_INDICATOR].unique()
    for pn_indicator in pn_indicators:
        pn_df, non_applied_stress = stress_indicator(pn_df,
                                                     scenario_stress[scenario_stress[CN_PN_INDICATOR] == pn_indicator],
                                                     pn_type, pn_indicator,
                                                     non_applied_stress,
                                                     option=option)
    return pn_df, non_applied_stress


def stress_indicator(pn_df, scenario_stress, pn_type, pn_indicator, non_applied_stress, option=""):
    num_cols = [x for x in pn_df.columns if x!="IND03"]
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
            else:
                non_applied_stress = pd.concat([non_applied_stress, row])
        except Exception as e:
            logger.error(
                'Erreur sur la ligne de stress PN:  {}     {}    {}    {}     {}     {}    {}'.format(pn_indicator,
                                                                                                      *list(row)))
            logger.error(e, exc_info=True)
            logger.info('Le programme poursuit l\'application des autres lignes de stress PN')
            non_applied_stress = pd.concat([non_applied_stress, row])
        if j == 0:
            pn_df_copy = stressed_pn_df.copy()
            j = +1
        else:
            pn_df_copy[num_cols] = pn_df_copy[num_cols].mask(((pn_df_copy[num_cols] == pn_df[num_cols])
                                                              | (np.isnan(pn_df_copy[num_cols]) & np.isnan(pn_df[num_cols]))),
                                                             stressed_pn_df[num_cols])

    if "%" in pn_type:
        pn_df_copy = pn_df_copy[filter_cond_cum].copy()

    return pn_df_copy, non_applied_stress


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
