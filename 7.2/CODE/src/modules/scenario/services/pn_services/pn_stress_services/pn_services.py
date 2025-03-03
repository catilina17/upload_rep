import logging
import os
from modules.scenario.parameters.general_parameters import *
import utils.general_utils as ut
from modules.scenario.utils import paths_resolver as pr
from modules.scenario.services.pn_services.pn_stress_services.stress_pn import process_scenarios_stress
from modules.scenario.services.pn_services.pn_stress_services.ajout_pn import ajout_pn_scenario

logger = logging.getLogger(__name__)


class PN_Manager():
    def __init__(self, cls_usr, etab, pn_output_path, scenario_rows, output_dir, scenario_name):
        self.up = cls_usr
        self.etab = etab
        self.pn_output_path = pn_output_path
        self.scenario_rows = scenario_rows
        self.output_dir = output_dir
        self.scenario_name = scenario_name

    def process_scenario_pn_stress(self):
        all_scenarios = self.scenario_rows.merge(self.up.pn_stress_list, left_on="SC PN", right_on="NOM STRESS PN")
        type_pn_unique = all_scenarios[CN_TYPE_PN].unique().tolist()
        list_type_pn = [item for sublist in [x.split("&") for x in self.type_pn.split(",")] for item in sublist]
        pn_prcent_non_stressed = [x for x in ['PN ECH%', 'NMD%'] if x.replace("PN ", "") not in list_type_pn]
        type_pn_unique2 = [x for x in type_pn_unique if x not in pn_prcent_non_stressed]
        if len(type_pn_unique) != len(type_pn_unique2):
            logger.info("    Les PNs suivantes n'ont pas été stressées par ces PNs n'ont pas été activées : " + str(
                [x for x in type_pn_unique if x not in type_pn_unique2]))

        if type_pn_unique2 != []:
            for type_pn in type_pn_unique2:
                scenarios = all_scenarios[all_scenarios[CN_TYPE_PN] == type_pn]
                if len(scenarios) > 0:
                    logger.info('    Calcul des PN stressées pour le type: ' + type_pn)
                    process_scenarios_stress(self.pn_output_path, type_pn, scenarios)
        else:
            logger.info("    Scénario sans stress de PN")

    def process_scenario_pn_ajout(self):
        all_scenarios = self.scenario_rows.merge(self.up.pn_ajout_list, left_on="SC PN", right_on="NOM SC AJOUT PN")
        type_pn_unique = all_scenarios[CN_TYPE_PN].unique().tolist()
        type_pn_unique2 = [x for x in type_pn_unique if x.replace("PN ", "") in self.type_pn]
        if len(type_pn_unique) != len(type_pn_unique2):
            logger.info("    Les PNs suivantes n'ont pas été ajoutées par ces PNs n'ont pas été activées : " + str(
                [x for x in type_pn_unique if x not in type_pn_unique2]))

        if type_pn_unique2 != []:
            for type_pn in type_pn_unique2:
                scenarios = all_scenarios[all_scenarios[CN_TYPE_PN] == type_pn]
                if len(scenarios) > 0:
                    logger.info('    Ajout des PNs pour le type: ' + type_pn)
                    ajout_pn_scenario(self.pn_output_path, type_pn, scenarios, self.etab)
        else:
            logger.info("    Scénario sans ajout de PN")

    def deactivate_non_requested_pns(self):
        all_pns = ["PN ECH", "PN ECH%", "NMD", "NMD%"]
        name_files = ["PN_ECH", "PN_ECH_BC", "NMD", "NMD_BC"]
        excl_files = [["PN_ECH_BC"], [], ["NMD_BC", "NMD_CALAGE"], []]
        activated_pns = ut.flatten([pn.split("&") for pn in self.type_pn.split(",")])
        for pn, name_file, excl_files in zip(all_pns, name_files, excl_files):
            if pn.replace("PN", "").replace(" ", "") not in activated_pns:
                pn_file_path = pr._get_file_path(os.path.join(self.output_dir, self.scenario_name), file_substring=name_file,
                                                 sub_folder="SC_VOLUME", no_files_substring=excl_files)
                os.remove(pn_file_path)

    def get_type_pn_a_activer(self):
        etab_params = self.up.pn_a_activer_df[self.up.pn_a_activer_df['ENTITE'] == self.etab]
        if etab_params.empty:
            etab_params = self.up.pn_a_activer_df[self.up.pn_a_activer_df['ENTITE'] == 'DEFAULT']
        self.type_pn = etab_params['PN A ACTIVER'].iloc[0]
