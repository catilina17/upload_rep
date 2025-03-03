import numpy as np
import pandas as pd
import os
from modules.scenario.services.pn_services.pn_bpce_services.pn_bpce_call import generate_PN_call
from modules.scenario.services.pn_services.pn_bpce_services.pn_bpce_refi_ajust import generate_PN_ajustements
from modules.scenario.services.pn_services.pn_bpce_services.pn_bpce_refi_ct import generate_PN_REFI_CT
from modules.scenario.services.pn_services.pn_bpce_services.pn_bpce_refi_mlt import generate_PN_REFI_MLT
from modules.scenario.services.pn_services.pn_bpce_services.pn_bpce_refi_sc import generate_PN_REFI_SC
import modules.scenario.services.pn_services.pn_bpce_services.pn_bpce_common as com
from mappings import mapping_functions as mapp
from modules.scenario.utils import paths_resolver as pr
from mappings import general_mappings as mp
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import modules.scenario.services.pn_services.pn_bpce_services.referential_bpce as rf_bpce
import logging

logger = logging.getLogger(__name__)

class PN_BPCE_Generator():
    def __init__(self, cls_usr, pn_output_path, etab, scenario_name):
        self.up = cls_usr
        self.pn_output_path = pn_output_path
        self.etab = etab
        self.scenario_name = scenario_name
    
    def get_PN_BPCE_service(self, scenario_rows):
        if self.etab != "BPCE":
            return
        sc_bpce = self.up.pn_bpce_sc_list[self.up.pn_bpce_sc_list["NOM SCENARIO"] == scenario_rows["PN BPCE"].iloc[0]]
        if sc_bpce.shape[0] > 0:
            logger.info("    Génération des PN ECH pour BPCE")
            self.generate_PN_BPCE(sc_bpce.iloc[0, 1], sc_bpce.iloc[0, 2])
        else:
            logger.info("    Pas de PN ECH pour BPCE")
    
    
    def generate_PN_BPCE(self, NG_SC_PN_AUTRES_BASSINS, NG_COMPILS_AUTRES_BASSINS):
    
        path_call = os.path.join(self.up.source_dir, mp.nomenclature_stock_ag[mp.nomenclature_stock_ag["TYPE FICHIER"] == "CALL"]["CHEMIN"].iloc[0])
        path_refi_ct = os.path.join(self.up.source_dir, mp.nomenclature_stock_ag[mp.nomenclature_stock_ag["TYPE FICHIER"] == "REFI-CT"]["CHEMIN"].iloc[0])
        path_refi_mlt = os.path.join(self.up.source_dir, mp.nomenclature_stock_ag[mp.nomenclature_stock_ag["TYPE FICHIER"] == "REFI-MLT"]["CHEMIN"].iloc[0])
    
        pn_ech_ajust = generate_PN_ajustements(NG_COMPILS_AUTRES_BASSINS, self.etab, self.scenario_name)
        pn_ech_refi_bassins = generate_PN_REFI_SC(NG_SC_PN_AUTRES_BASSINS, self.etab, self.scenario_name)
        pn_ech_refi_ct = generate_PN_REFI_CT(self.etab, path_refi_ct)
        pn_ech_refi_mlt = generate_PN_REFI_MLT(self.up.dar, self.etab, path_refi_mlt)
        pn_calls = generate_PN_call(self.up.dar, self.etab, path_call)
    
        pn_ech = pd.concat([pn_ech_refi_mlt, pn_ech_refi_ct, pn_calls, pn_ech_ajust, pn_ech_refi_bassins])
    
        if pn_ech.shape[0] > 0:
            pn_ech = self.map_pn_bpce(pn_ech)
            pn_ech = self.final_format(pn_ech)
            pn_ech_file_path = pr._get_file_path(self.pn_output_path, file_substring='PN_ECH', no_files_substring=["PN_ECH_BC"])
            pn_ech.to_csv(pn_ech_file_path, sep =";", decimal=",", index= False)
    
        else:
            logger.warning(
                "     Un scénario BPCE a été ajouté mais aucun fichier n'a été trouvé ou les données procurées sont vides")
    
    
    def final_format(self, pn_ech):
        col_sortie = pa.NC_PA_COL_SORTIE_QUAL_ECH + pa.NC_PA_COL_SORTIE_NUM_PN
    
        pn_ech[pa.NC_PA_LCR_TIERS_SHARE] = 100
    
        missing_indics = [x for x \
                          in col_sortie \
                          if not x in pn_ech.columns.tolist()]
    
        pn_ech = pd.concat([pn_ech, pd.DataFrame([[""] * len(missing_indics)], \
                                                 index=pn_ech.index, columns=missing_indics)], axis=1)
    
        pn_ech = com.generate_index(pn_ech, repeat=4)
    
        pn_ech = pn_ech[col_sortie].copy()
    
        pn_ech[pa.NC_PA_COL_SORTIE_NUM_PN] = pn_ech[pa.NC_PA_COL_SORTIE_NUM_PN].fillna("")
    
        return pn_ech
    
    def map_pn_bpce(self, pn_ech):
        filtres = [pn_ech[pa.NC_PA_CONTRACT_TYPE].str[0:2] == "A-", pn_ech[pa.NC_PA_CONTRACT_TYPE].str[0:2] == "P-"]
        choices = ["B ACTIF", "B PASSIF"]
        pn_ech[pa.NC_PA_BILAN] = np.select(filtres, choices)
    
        pn_ech[pa.NC_PA_SCOPE] = rf_bpce.SCOPE_MNI_LIQUIDITE
    
        CLES = pa.NC_PA_CLE_OUTPUT
        pn_ech[CLES] = pn_ech[CLES].fillna("-").replace("", "-")
    
        pn_ech = mapp.mapping_consolidation_liquidite(pn_ech)
    
        pn_ech = mapp.map_data(pn_ech, mp.map_pass_alm["CONTRATS2"], \
                               keys_data=[pa.NC_PA_CONTRACT_TYPE], name_mapping="PN ECH DATA vs.")
    
        return pn_ech
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
