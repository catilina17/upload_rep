# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:02:43 2020

@author: Hossayne
"""
import modules.moteur.mappings.main_mappings as mp
import modules.moteur.parameters.user_parameters as up
import modules.moteur.parameters.general_parameters as gp
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

WARNING_BASSIN = "        Les données de PN %s semblent appartenir à un bassin différent de celui spécifié dans les paramètres"

def add_ig_colum(data, palier_col, contrat_col):
    """ Fonction permettant d'ajouter la colonne isIG"""
    if not up.bassin_usr in data[gp.nc_output_bassin_cle].values.tolist() and data.shape[0] != 0:
        logger.warning(WARNING_BASSIN)

    data["BASSIN_IG"] = mp.bassin_ig_map.loc[up.bassin_usr, gp.nc_igm_bassinig]

    data["CONTREPARTIE"] = data.join(mp.bassin_ig_map[[gp.nc_igm_bassinig]], on=palier_col)[gp.nc_igm_bassinig].copy()

    data["CLE_IG"] = data[contrat_col] + "_" + data["BASSIN_IG"] + "_" + data["CONTREPARTIE"]

    filtre_ig = data["CLE_IG"].isin(mp.ig_mirror_contracts_map.index.values.tolist())

    filtre_ig = (filtre_ig) & ((data[palier_col] != up.bassin_usr) | (data[contrat_col].astype(str).str[:1] != "P")) & \
                (~  ((data[palier_col] != up.bassin_usr) & (data["BASSIN_IG"] == "RZO") & \
                     ((data[palier_col] == "CFF") | (
                             (data[gp.nc_output_bassin_cle] == "BP") & (data[palier_col] == "PAL")) | (
                          ~data[gp.nc_output_bassin_cle].isin(["PAL", "CFF", "BP"])))))

    data[gp.nc_output_isIG] = np.where(filtre_ig, "IG", "-")

    return data


def add_mirror_contract(data):
    """ Fonction permettant d'ajouter les contrats intra-group miroir"""
    data_igm = data[data[gp.nc_output_isIG] == "IG"].copy()

    if data_igm.shape[0] > 0:
        data_igm[gp.nc_output_isIG] = "IGM"

        num_cols = ["M0" + str(k) if k < 10 else "M" + str(k) for k in range(0, up.nb_mois_proj_usr + 1)]
        data_igm[num_cols] = -data_igm[num_cols]

        data = pd.concat([data, data_igm])

    return data


def add_counterparty_contract(data):
    """ Fonction permettant d'ajouter les contrats contraprtie"""

    data["CLE2"] = data[gp.nc_output_key].copy()

    data["CLE3"] = data[gp.nc_output_isIG].copy()

    data["CLE4"] = data[gp.nc_output_bilan].copy()

    data_c = data[data[gp.nc_output_isIG] == "IG"].copy()

    if data_c.shape[0] > 0:

        data_c[gp.nc_output_isIG] = "IGM"

        temp_palier = data_c[[gp.nc_output_palier_cle]].copy()

        temp_cle = data_c[gp.nc_output_key].copy()

        temp_bilan = data_c[gp.nc_output_bilan].copy()

        data_c[gp.nc_output_palier_cle] = data_c[gp.nc_output_bassin_cle]

        data_c[gp.nc_output_bassin_cle] = temp_palier

        data_c["BASSIN_IG"] = mp.bassin_ig_map.loc[up.bassin_usr, gp.nc_igm_bassinig]

        data_c["CONTREPARTIE"] = temp_palier.join(mp.bassin_ig_map[[gp.nc_igm_bassinig]], on=gp.nc_output_palier_cle)[
            gp.nc_igm_bassinig].copy()

        data_c["CLE_IG"] = data_c[gp.nc_output_contrat_cle] + "_" + data_c["BASSIN_IG"] + "_" + data_c["CONTREPARTIE"]

        data_c[gp.nc_output_contrat_cle] = \
            data_c.join(mp.ig_mirror_contracts_map, how="left", on="CLE_IG", rsuffix="_IG")[gp.nc_ig_contrat_new]

        for col in [gp.nc_output_bilan, gp.nc_output_poste, gp.nc_output_dim2, gp.nc_output_dim3, gp.nc_output_dim4,
                    gp.nc_output_dim5]:
            data_c[col] = data_c.join(mp.contrats_map, how="left", on=gp.nc_output_contrat_cle, rsuffix="_MAP")[
                col + "_MAP"]

        data_c['new_key'] = data_c[gp.cle_stock].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

        data_c[gp.nc_output_key] = data_c["new_key"].values

        data_c["CLE2"] = temp_cle

        data_c["CLE3"] = "IGMZ"

        data_c["CLE4"] = temp_bilan

        data_c = data_c.reset_index(drop=True)

        data_c = data_c.drop(["BASSIN_IG", "CONTREPARTIE", "CLE_IG"], axis=1)

        data_c = data_c.set_index('new_key')

        data_c = mp.map_liq(data_c)

        filtre_surcentralisation = np.where(np.array(temp_bilan) != np.array(data_c[gp.nc_output_bilan]), True, False)
        num_cols = ["M0" + str(k) if k < 10 else "M" + str(k) for k in range(0, up.nb_mois_proj_usr + 1)]
        data_c.loc[filtre_surcentralisation, num_cols] = -data_c.loc[filtre_surcentralisation, num_cols]

    data = pd.concat([data, data_c])

    return data
