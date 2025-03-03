import os
import numpy as np
import pandas as pd
from mappings import general_mappings as mapp
from mappings import mapping_functions as mp
import modules.scenario.services.pn_services.pn_bpce_services.referential_bpce as rf_bpce
import modules.scenario.services.pn_services.pn_bpce_services.pn_bpce_common as com
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import logging

logger = logging.getLogger(__name__)


def generate_PN_REFI_SC(NG_SC_PN_AUTRES_BASSINS, etab, scenario_name):
    data_ech_bassins = pd.DataFrame(columns=["ADZDZDZ"])

    mapping_refi = mapp.mapping_bpce_pn["REFI RZO"]

    scen_path = NG_SC_PN_AUTRES_BASSINS
    if not scen_path is None and scen_path != "":
        logger.info("       GENERATION DES OPE DE REFI A PARTIR DES PN ECH DES AUTRES BASSINS")
        sc_dir = os.walk(scen_path)
        for subdir, dir, files in sc_dir:
            for file_name in files:
                if ".csv" in file_name and "PN_ECH" in file_name and not "PN_ECH_BC" in file_name:
                    if scenario_name == subdir.split("\\")[-2]:
                        logger.info("          Lecture de : " + file_name + " pour " + subdir.split("\\")[-1])
                        data_temp = pd.read_csv(os.path.join(subdir, file_name), sep=";", decimal=",")
                        data_temp[pa.NC_PA_PALIER] = data_temp[pa.NC_PA_BASSIN]

                        if len(data_ech_bassins) != 0:
                            data_ech_bassins = pd.concat([data_ech_bassins, data_temp])
                        else:
                            data_ech_bassins = data_temp

        if len(data_ech_bassins) > 0:
            data_ech_bassins = data_ech_bassins.reset_index(drop=True)
            data_ech_bassins = data_ech_bassins[data_ech_bassins[pa.NC_PA_IND03].isin([pa.NC_PA_DEM, pa.NC_PA_MG_CO])].copy()
            data_ech_bassins = data_ech_bassins[pa.NC_PA_COL_SORTIE_QUAL_ECH + pa.NC_PA_COL_SORTIE_NUM_PN].copy()
            cols_to_mapp = [x for x in mapping_refi["OUT"] if not x in pa.NC_PA_PERIMETRE]
            data_ech_bassins = mp.map_data(data_ech_bassins, mapping_refi, cols_mapp=cols_to_mapp,
                                             keys_data=[pa.NC_PA_CONTRACT_TYPE], join_how="inner")
            data_ech_bassins = data_ech_bassins.drop(columns=[pa.NC_PA_CONTRACT_TYPE], axis=1).rename(
                columns={rf_bpce.NC_CONTRAT_BPCE: pa.NC_PA_CONTRACT_TYPE})

            data_ech_bassins[pa.NC_PA_BASSIN] = etab

            num_cols = pa.NC_PA_COL_SORTIE_NUM_PN

            cols_to_keep = pa.NC_PA_CLE_OUTPUT + pa.NC_PA_COL_SPEC_ECH + [
                pa.NC_PA_IND03]  # ON ELIMINE COLONNES DE LIQUIDITE

            data_ech_bassins[pa.NC_PA_COL_SORTIE_NUM_PN] = data_ech_bassins[pa.NC_PA_COL_SORTIE_NUM_PN].fillna(0)

            filtre_mg_co = data_ech_bassins[pa.NC_PA_IND03] == pa.NC_PA_MG_CO
            filtre_dem = data_ech_bassins[pa.NC_PA_IND03] == pa.NC_PA_DEM

            data_ech_bassins.loc[filtre_mg_co, num_cols] \
                = data_ech_bassins.loc[filtre_mg_co, num_cols].values * data_ech_bassins.loc[
                filtre_dem, num_cols].values

            data_ech_bassins = data_ech_bassins.groupby(by=cols_to_keep, axis=0,
                                                        as_index=False, dropna=False).sum(numeric_only=True).reset_index(drop=True)

            filtre_mg_co = data_ech_bassins[pa.NC_PA_IND03] == pa.NC_PA_MG_CO
            filtre_dem = data_ech_bassins[pa.NC_PA_IND03] == pa.NC_PA_DEM

            data_ech_bassins.loc[filtre_mg_co, num_cols] = np.nan_to_num((data_ech_bassins.loc[filtre_mg_co, num_cols].values / \
                                                            data_ech_bassins.loc[filtre_dem, num_cols].values),posinf=0, neginf=0)

            data_ech_bassins = com.add_col_bilan(data_ech_bassins)
            data_ech_bassins = com.add_missing_key_cols(data_ech_bassins)

            data_ech_bassins = com.generate_index(data_ech_bassins, repeat=2)

            data_ech_bassins = data_ech_bassins.reset_index(drop=True)

            data_ech_bassins = com.add_missing_indics(data_ech_bassins)

            data_ech_bassins = data_ech_bassins.reset_index(drop=True)

        else:
            logger.info("         Pas de fichier PN_ECH trouvé")

        return data_ech_bassins