import os
from datetime import datetime
from modules.scenario.parameters import general_parameters as gp
import logging
import numpy as np
from pathlib import Path
logger = logging.getLogger(__name__)


class UserParameters():
    def __init__(self, sp_cls):
        self.simul_cls = sp_cls
        self.om = sp_cls.output_cls
        self.sm = sp_cls.sources_cls
        self.dar = sp_cls.dar
        self.etabs = sp_cls.entities_list
        self.sources_folder = self.sm.sources_folder
        self.output_folder = self.om.output_folder
        self.config_cls = sp_cls.config_cls
        self.names = self.config_cls.names_ex
        self.RN_USER_RATE_SC = 'USER RATE SCENARIO'
        self.RN_USER_RATE_SHOCK = 'CHOC TAUX'
        self.SCENARIO_NAME = 'NOM SCENARIO'
        self.SCENARIO_NAME_ORIG = 'NOM SCENARIO ORIG'

    def get_ihm_parameters(self):
        self.get_alim_dir()
        self.get_output_dir()
        self.get_scenarios_parameters()
        self.get_all_etabs()
        self.get_rates_files_path()
        self.get_model_file_paths()
        self.get_pn_a_activer()
        self.get_reference_scenarios_list()
        self.get_holidays_dates()
        self.get_main_sc_eve()

    def get_main_sc_eve(self):
        self.main_sc_eve = self.config_cls.get_value_from_named_ranged(self.names.RN_MAIN_SC_EVE)

    def get_holidays_dates(self):
        holidays_df = self.config_cls.get_dataframe_from_named_ranged(self.names.RN_EUR_HOLIDAYS, False)
        holidays_df.drop(0, inplace=True)
        holidays_df[0] = holidays_df[0].astype(str)
        holidays_df[0] = holidays_df[0].str.split(n=1, expand=True)[0]
        self.holidays_list = np.array([datetime.strptime(x, '%Y-%m-%d') for x in holidays_df[0]], dtype='datetime64[D]')

    def get_reference_scenarios_list(self):
        self.st_refs = self.config_cls.get_dataframe_from_named_ranged(self.names.RN_STOCK_REF_TX_SC)

    def get_pn_a_activer(self):
        self.pn_a_activer_df = self.config_cls.get_dataframe_from_named_ranged(self.names.RN_PN_TO_ACTIVATE)

    def get_model_file_paths(self):
        modele = self.config_cls.get_value_from_named_ranged(self.names.RN_MODELE_DAV)
        self.modele_dav_path = os.path.join(self.sm.sources_folder, "MODELES", modele)

    def get_rates_files_path(self):
        self.tx_curves_path = self.get_input_tx_file_path(self.names.RN_TX_PATH)
        self.liq_curves_path = self.get_input_tx_file_path(self.names.RN_LIQ_PATH)
        self.tci_curves_path = self.get_input_tx_file_path(self.names.RN_TCI_PATH)
        self.get_input_zc_file_path(self.names.RN_ZC_FILE_PATH)

    def get_input_zc_file_path(self, range_name):
        zc_file_path = str(self.config_cls.get_value_from_named_ranged(range_name))
        if zc_file_path == "":
            return ""
        self.zc_file_path = os.path.join(self.sm.sources_folder, self.sm.rate_input_folder_name, zc_file_path)

    def get_input_tx_file_path(self, range_name):
        tx_file_path = self.config_cls.get_value_from_named_ranged(range_name)
        tx_file_path = os.path.join(self.sm.sources_folder, self.sm.rate_input_folder_name, tx_file_path)
        return tx_file_path

    def get_output_dir(self):
        output_path = self.output_folder
        output_path = os.path.join(output_path, '{}' + '_DAR-{:%Y%m%d}'.format(self.dar)
                                   + '_EXEC-' + '{:%Y%m%d.%H%M.%S}'.format(datetime.now()))
        self.output_dir = output_path.format('SC')
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def get_alim_dir(self):
        self.alim_dir_path = self.sm.alim_folder

    def get_scenarios_parameters(self):
        self.pn_bpce_sc_list = self.get_scenario_pn_bpce_list()
        self.pn_stress_list = self.get_scenario_stress_pn_list()
        self.pn_ajout_list = self.get_scenario_ajout_pn_list()
        self.tx_chocs_list = self.get_scenario_tx_shocks_list()
        self.scenario_list = self.get_scenarios_list()
        self.stress_dav_list = self.get_stress_dav_list()
        self.models_list = self.get_models_list()
        self.scenarii_calc_all, self.scenarii_dav_all = self.get_calculateur_and_dav_scenarios()

    def get_calculateur_and_dav_scenarios(self):
        scenarii_calc_all = self.config_cls.get_dataframe_from_named_ranged(self.names.RN_SCENARIO_RC_ST)
        scenarii_dav_all = self.config_cls.get_dataframe_from_named_ranged(self.names.RN_SCENARIO_SRC_DAV)
        return scenarii_calc_all, scenarii_dav_all

    def get_models_list(self):
        stress_dav_list = self.config_cls.get_dataframe_from_named_ranged('_SC_MOD')
        return stress_dav_list

    def get_stress_dav_list(self):
        models_list = self.config_cls.get_dataframe_from_named_ranged(self.names.RN_SCENARIO_SRC_DAV)
        return models_list

    def get_scenarios_list(self):
        scenario_list = self.config_cls.get_dataframe_from_named_ranged(self.names.RN_SCENARIO_LIST_CELL)
        scenario_list.drop_duplicates(inplace=True)
        scenario_list[gp.RN_SC_TAUX_USER] = scenario_list[gp.RN_SC_TAUX_USER].str.split(',')
        scenario_list = scenario_list.explode(gp.RN_SC_TAUX_USER)
        scenario_list[self.RN_USER_RATE_SC] = scenario_list[gp.RN_SC_TAUX_USER] + '_' + scenario_list[self.RN_USER_RATE_SHOCK]
        scenario_list[self.SCENARIO_NAME_ORIG] = scenario_list[self.SCENARIO_NAME]
        scenario_list[self.SCENARIO_NAME] = scenario_list[self.SCENARIO_NAME] + '_' + scenario_list[self.RN_USER_RATE_SC]
        scenario_list.reset_index(inplace=True, drop=True)
        return scenario_list

    def get_scenario_tx_shocks_list(self):
        scenario_list = self.config_cls.get_dataframe_from_named_ranged(self.names.RN_SHOCKS_FST_CELL)
        return scenario_list

    def get_scenario_ajout_pn_list(self):
        stress_pn_list = self.config_cls.get_dataframe_from_named_ranged(self.names.RN_ADD_PN_FST_CELL)
        return stress_pn_list

    def get_scenario_stress_pn_list(self):
        stress_pn_list = self.config_cls.get_dataframe_from_named_ranged(self.names.RN_STRESS_PN_FST_CELL)
        return stress_pn_list

    def get_scenario_pn_bpce_list(self):
        listo = self.config_cls.get_dataframe_from_named_ranged(self.names.RN_SC_PN_BPCE_FST_CELL)
        return listo

    def get_all_etabs(self):
        list_all_etabs = self.config_cls.get_dataframe_from_named_ranged(self.names.NAME_RANGE_ALL_ETABS).iloc[:, 0].tolist()
        list_all_etabs = [x.strip(' ') for x in list_all_etabs]
        list_etab = list(set(
            [item for sublist in [x.split(",") for x in self.scenario_list['ETABLISSEMENTS'].values.tolist()] for item
             in
             sublist]))
        list_etab = [x.strip(' ') for x in list_etab]
        codification = self.config_cls.get_dataframe_from_named_ranged('_codification_etab')
        j = 0
        a_virer = [None]
        for x in codification.columns:
            if x in list_etab:
                if x not in list_all_etabs:
                    list_etab = list_etab + codification.iloc[:, j].values.tolist()
                    a_virer = a_virer + [x]
                else:
                    logger.warning(
                        x + " est à la fois une liste d'établissements et un nom d'établissement." \
                        + " Veuillez changer le nom de la liste")
            j = j + 1
        list_etab = [x for x in list_etab if x not in a_virer]
        a_virer = []
        for x in list_etab:
            if x not in list_all_etabs:
                a_virer = a_virer + [x]
                logger.warning(
                    "L'établissement " + x + " n'est pas un établissement pris en charge ou une liste d'établissements")
        list_etab = [x for x in list_etab if x not in a_virer]
        list_etab = list(set(list_etab))
        self.all_etabs = self.check_if_etab_data_are_in_alim_output_dir(list_etab, warning=True)
        logger.info("La liste des établissements traitée sera: " + str(self.all_etabs))

        self.codification_etab = self.config_cls.get_dataframe_from_named_ranged('_codification_etab')
    def check_if_etab_data_are_in_alim_output_dir(self, etabs_liste, warning=False):
        valide_etab_liste = [x for x in etabs_liste if os.path.exists(os.path.join(self.alim_dir_path, x))]
        not_found_etabs = [x for x in etabs_liste if x not in valide_etab_liste]
        if warning:
            if len(not_found_etabs) >= 1:
                logger.warning(
                    "Les établissements suivants ne sont pas disponibles dans le dossier de l'ALIM : {}".format(
                        ','.join(not_found_etabs)))
        return valide_etab_liste