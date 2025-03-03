import os
import logging
import json
import pandas as pd
import utils.general_utils as ut
from modules.scenario.utils.paths_resolver import copy_file_if_not_exist, copy_folder_to_folder
from pathlib import Path
from modules.scenario.utils import paths_resolver as pr
from params import version_params as vp

logger = logging.getLogger(__name__)


class Stock_and_PN_Files_Exporter():
    def __init__(self, cls_usr, etab, output_dir, scenario_name, scenario_rows):
        self.up = cls_usr
        self.scenario_name = scenario_name
        self.output_dir = output_dir
        self.scenario_calc = scenario_rows['SC CALCULATEUR'].iloc[0]
        self.scenario_dav = scenario_rows['SURCOUCHE DAV'].iloc[0]
        self.scenario_models = scenario_rows['SC MODELE'].iloc[0]
        self.etab = etab

    def copy_alim_files(self):
        _stock_file_path = pr.get_stock_file(self.up.alim_dir_path, self.etab)
        copy_file_if_not_exist(_stock_file_path,
                               os.path.join(self.output_dir.format('SC'), os.path.basename(_stock_file_path)))

        _stock_nmd_template_file_path = pr.get_stock_nmd_template_file(self.up.alim_dir_path, self.etab)
        copy_file_if_not_exist(_stock_nmd_template_file_path,
                               os.path.join(self.output_dir.format('SC'),
                                            os.path.basename(_stock_nmd_template_file_path)))

        sc_volume_path = pr.get_sc_volume_folder(self.up.alim_dir_path, self.etab)
        pn_file_path = os.path.join(self.output_dir, self.scenario_name, "SC_VOLUME")
        copy_folder_to_folder(sc_volume_path, pn_file_path)

        return pn_file_path

    def export_scenario_parameters(self):
        Path(os.path.dirname(os.path.join(self.output_dir, self.scenario_name, "MODELES"))).mkdir(parents=True,
                                                                                                  exist_ok=True)
        if self.scenario_dav.upper() != "SANS SC SURCOUCHE":
            self.check_and_copy_modele_dav("DAV")
        self.check_and_copy_modele("ECH")
        self.check_and_copy_modele("NMD")
        self.create_json_file_with_scenarios_params()

    def create_json_file_with_scenarios_params(self):
        dic_params = {}
        scenarii_calc = self.up.scenarii_calc_all[
            self.up.scenarii_calc_all["NOM SCENARIO"] == self.scenario_calc].copy()
        if len(scenarii_calc) == 0:
            scenarii_calc = pd.DataFrame([["", ""]], columns=["NOM SCENARIO", "TYPE PRODUIT"])

        scenarii_dav = self.up.scenarii_dav_all[self.up.scenarii_dav_all["NOM SCENARIO"] == self.scenario_dav].copy()
        if len(scenarii_dav) > 0:
            scenarii_dav = scenarii_dav.drop("NOM SCENARIO", axis=1).rename(columns={"NOM MODELE": "NOM SCENARIO"})
            scenarii_dav['TYPE PRODUIT'] = "DAV SURCOUCHE"
            scenarii_calc = pd.concat([scenarii_calc, scenarii_dav])

        dic_params["DATA_SC_CALC"] = scenarii_calc.to_dict('records')
        dic_params["MAIN_SCENARIO_EVE"] = self.up.main_sc_eve
        with open(os.path.join(self.output_dir, self.scenario_name, 'scenario_params.json'), 'w') as jsonFile:
            json.dump(dic_params, jsonFile)

    def check_and_copy_modele_dav(self, modele_type, warning_existence=True, copy_model=True):
        scenarii = self.check_model_existence(self.scenario_dav, self.up.stress_dav_list)
        if scenarii is None:
            if warning_existence:
                logger.warning("    Vous n'avez pas précisé de scénario de %s pour le scénario %s."
                               " Les %s ne pourront pas tourner dans le moteur" % (modele_type, self.scenario_name,
                                                                                   modele_type))
            return False

        ut.check_version_templates(self.up.modele_dav_path, path=self.up.modele_dav_path,
                                   version=eval("vp.version_modele_%s" % modele_type.lower()), open=True)
        modele_name = "DAV"
        if copy_model:
            file_name = ("SC_MOD_%s_" % modele_name + os.path.splitext(os.path.basename(self.up.modele_dav_path))[0]
                         + '_%s.xlsx' % "_".join(self.scenario_name.split("_")[:-2]))
            copy_file_if_not_exist(self.up.modele_dav_path,
                                   os.path.join(self.output_dir.format('SC'), self.scenario_name, "MODELES", file_name))

    def check_and_copy_modele(self, modele_type, warning_existence=True, copy_model=True):
        scenarii = self.check_model_existence(self.scenario_models, self.up.models_list)
        if scenarii is None:
            if warning_existence:
                logger.warning(
                    "    Vous n'avez pas précisé de scénario de %s pour le scénario %s. Les %s ne pourront pas tourner dans le moteur" % (
                        modele_type, self.scenario_name, modele_type))
            return False
        scenario_name = scenarii[scenarii["TYPE MODELE"] == modele_type]
        if len(scenario_name) == 0:
            if warning_existence:
                logger.warning(
                    "    Vous n'avez pas précisé de scénario de %s pour le scénario %s. Les %s ne pourront pas tourner dans le moteur" % (
                        modele_type, self.scenario_name, modele_type))

        modele = scenario_name["MODELE"].iloc[0]
        modele_path = os.path.join(self.up.sources_folder, "MODELES", modele)
        ut.check_version_templates(modele_path, path=modele_path,
                                   version=eval("vp.version_modele_%s" % modele_type.lower()), open=True)
        modele_name = modele_type.replace("PN ", "").replace("STOCK ", "")
        if copy_model:
            file_name = ("SC_MOD_%s_" % modele_name + os.path.splitext(os.path.basename(modele_path))[0]
                         + '_%s.xlsx' % "_".join(self.scenario_name.split("_")[:-2]))
            copy_file_if_not_exist(modele_path,
                                   os.path.join(self.output_dir.format('SC'), self.scenario_name, "MODELES", file_name))

    def check_model_existence(self, scenario_name, scenarios_df):
        scenarii = None
        try:
            if len(scenarios_df) == 0:
                return None

            scenarii = scenarios_df[scenarios_df['NOM SCENARIO'] == scenario_name]
            if len(scenarii) == 0:
                return None

        except IndexError as e:
            logger.error(e, exc_info=True)
            logger.info(
                'Le scénario {} n\'est pas  défini dans la liste des modèles de l\'onglet SC calculateur'.format(
                    scenario_name))
        return scenarii
