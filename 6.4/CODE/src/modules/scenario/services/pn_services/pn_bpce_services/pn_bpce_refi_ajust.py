import os
import pandas as pd
import modules.scenario.services.pn_services.pn_bpce_services.mapping_service as mapp
import modules.scenario.services.pn_services.pn_bpce_services.referential_bpce as rf_bpce
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import modules.scenario.services.pn_services.pn_bpce_services.pn_bpce_common as com
import logging

logger = logging.getLogger(__name__)


def generate_PN_ajustements(NG_COMPILS_AUTRES_BASSINS, etab, scenario_name):
    data_ajustements = pd.DataFrame(columns=["ADZDZDZ"])

    mapping_refi = mapp.mapping["mapping_BPCE"]["REFI RZO"]
    mapping_refi_bpce = mapp.mapping["mapping_BPCE"]["REFI BPCE"]
    mapping_conso_ech = mapp.mapping["mapping_PN"]["mapping_CONSO_ECH"]

    scen_path = NG_COMPILS_AUTRES_BASSINS
    if scen_path != "" and scen_path is not None:
        logger.info("       GENERATION DES OPE DE REFI A PARTIR DES AJUSTEMENTS DES AUTRES BASSINS")
        sc_dir = os.walk(scen_path)
        cols_to_keep = pa.NC_PA_CLE_OUTPUT + rf_bpce.COL_NUM_COMPIL
        for subdir, dir, files in sc_dir:
            for file_name in files:
                if ".csv" in file_name and rf_bpce.NAME_COMPIL_PN in file_name and not file_name[0:2] == "~$":
                    if scenario_name in subdir.split("\\")[-1]:
                        logger.info(
                            "          Lecture de : " + file_name + " pour " + "\\".join(subdir.split("\\")[-2:]))

                        data_temp = pd.read_csv(os.path.join(subdir, file_name), sep=";", decimal=",", engine='c',
                                                low_memory=False, encoding='latin-1')
                        data_temp = data_temp[
                            (data_temp[pa.NC_PA_CONTRACT_TYPE].isin(rf_bpce.CONTRATS_AJUST)) & (
                                    data_temp[rf_bpce.NC_IND3_COMPIL] == "EM")]

                        data_temp[pa.NC_PA_PALIER] = data_temp[pa.NC_PA_BASSIN]

                        try:
                            data_temp = data_temp[cols_to_keep].copy()
                        except:
                            msg = "          The following columns are missing in " + subdir.split("\\")[
                                -1] + "\\" + file_name + ": " + str(
                                [x for x in cols_to_keep if x not in data_temp])
                            logger.error(msg)
                            raise ValueError(msg)

                        if len(data_ajustements) != 0:
                            data_ajustements = pd.concat([data_ajustements, data_temp])
                        else:
                            data_ajustements = data_temp

        if len(data_ajustements) > 0:
            data_ajustements = data_ajustements.groupby(by=[pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_PALIER], axis=0,
                                                        as_index=False).sum(numeric_only=True)

            cols_to_mapp = [x for x in mapping_refi["OUT"] if not x in pa.NC_PA_PERIMETRE]
            data_ajustements = mapp.map_data(data_ajustements, mapping_refi, cols_mapp=cols_to_mapp,
                                             keys_data=[pa.NC_PA_CONTRACT_TYPE], join_how="inner")
            data_ajustements = data_ajustements.drop(columns=[pa.NC_PA_CONTRACT_TYPE], axis=1).rename(
                columns={rf_bpce.NC_CONTRAT_BPCE: pa.NC_PA_CONTRACT_TYPE})

            data_ajustements = mapp.map_data(data_ajustements, mapping_refi_bpce, keys_data=[pa.NC_PA_CONTRACT_TYPE],
                                             join_how="inner")

            data_ajustements = mapp.map_data(data_ajustements, mapping_conso_ech, keys_data=[pa.NC_PA_CONTRACT_TYPE],
                                             join_how="inner")
            data_ajustements[pa.NC_PA_JR_PN] = 1
            data_ajustements[pa.NC_PA_MATURITY_DURATION] = 1
            data_ajustements[pa.NC_PA_BASSIN] = etab
            data_ajustements[pa.NC_PA_LCR_TIERS] = rf_bpce.STD_LCR_TIERS
            data_ajustements[pa.NC_PA_IND03] = pa.NC_PA_DEM
            data_ajustements = data_ajustements.rename(columns={"M0" + str(i): "M" + str(i) for i in range(0, 10)})
            data_ajustements[pa.NC_PA_INDEX] = ["ECH" + str(i) for i in
                                                  range(1, data_ajustements.shape[0] + 1)]
            data_ajustements = com.add_col_bilan(data_ajustements)
            data_ajustements = com.add_missing_key_cols(data_ajustements)
            data_ajustements = data_ajustements.reset_index(drop=True)
            data_ajustements = com.add_missing_indics(data_ajustements)
            data_ajustements = data_ajustements.reset_index(drop=True)
        return data_ajustements