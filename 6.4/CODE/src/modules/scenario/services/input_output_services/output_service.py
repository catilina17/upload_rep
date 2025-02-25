import os
import logging
import json
import pandas as pd
import utils.excel_utils as excel_helper
import utils.general_utils as ut
from modules.scenario.utils.work_dir_resolver import copy_file_and_rename, copy_file_if_not_exist
from modules.scenario.services import referential_file_service as ref_services
from utils.excel_utils import try_close_workbook
from modules.scenario.parameters import user_parameters as up
from params import version_params as vp

logger = logging.getLogger(__name__)


def resolve_output_workbook(_input_file_path, output_dir, scenario_name, open=True):
    try:
        os.mkdir(os.path.join(output_dir.format('SC'), scenario_name))
    except OSError:
        pass
    output_file_path = os.path.join(output_dir.format('SC'), scenario_name,
                                    os.path.splitext(os.path.basename(_input_file_path))[0] \
                                    + '_%s.xlsb' % "_".join(scenario_name.split("_")[:-2]))

    copy_file_and_rename(_input_file_path, output_file_path)
    excel_helper.close_workbook_by_name(output_file_path)
    if open:
        wb = excel_helper.try_close_open(output_file_path)
        return wb


def get_output_sc_workbook_and_copy_stock_file(etab, output_dir, scenario_name):
    _stock_file_path = ref_services.get_stock_workbook(etab)
    copy_file_if_not_exist(_stock_file_path, os.path.join(output_dir.format('SC'), os.path.basename(_stock_file_path)))

    _stock_nmd_template_file_path = ref_services.get_stock_nmd_template_workbook(etab)
    copy_file_if_not_exist(_stock_nmd_template_file_path,
                           os.path.join(output_dir.format('SC'), os.path.basename(_stock_nmd_template_file_path)))

    _input_tx_file_path, input_tx_wb = ref_services.get_input_tx_workbook(etab)
    tx_wb = resolve_output_workbook(_input_tx_file_path, output_dir, scenario_name)

    _input_lcr_nsfr_file_path = ref_services.get_input_lcr_nsfr_workbook(etab)
    if _input_lcr_nsfr_file_path != "":
        resolve_output_workbook(_input_lcr_nsfr_file_path, output_dir, scenario_name, open=False)

    stock_file_path = os.path.join(output_dir.format('SC'), os.path.basename(_stock_file_path))

    if input_tx_wb is not None:
        try_close_workbook(input_tx_wb, scenario_name)

    return tx_wb, stock_file_path


def export_scenario_parameters(output_dir, scenario_name, scenario_calc, scenario_dav, scenario_models):
    if scenario_dav.upper() != "SANS SC SURCOUCHE":
        check_and_copy_modele_dav(scenario_dav, output_dir, scenario_name, "DAV")
    check_and_copy_modele(scenario_models, output_dir, scenario_name, "ECH")
    check_and_copy_modele(scenario_models, output_dir, scenario_name, "PEL")
    check_and_copy_modele(scenario_models, output_dir, scenario_name, "NMD")
    create_json_file_with_scenarios_params(output_dir, scenario_calc, scenario_dav, scenario_name)


def check_and_copy_modele_dav(scenario_name, output_dir, sc_global, modele_type, warning_existence=True,
                              copy_model=True):
    scenarii = check_model_existence(scenario_name, up.stress_dav_list)
    if scenarii is None:
        if warning_existence:
            logger.warning(
                "    Vous n'avez pas précisé de scénario de %s pour le scénario %s. Les %s ne pourront pas tourner dans le moteur" % (
                    modele_type, sc_global, modele_type))
        return False

    ut.check_version_templates(up.modele_dav_path, path=up.modele_dav_path,
                               version=eval("vp.version_modele_%s" % modele_type.lower()), open=True)
    modele_name = "DAV"
    if copy_model:
        copy_file_if_not_exist(up.modele_dav_path, os.path.join(output_dir.format('SC'), sc_global,
                                                                "SC_MOD_%s_" % modele_name +
                                                                os.path.splitext(os.path.basename(up.modele_dav_path))[
                                                                    0] \
                                                                + '_%s.xlsx' % "_".join(sc_global.split("_")[:-2])))


def create_json_file_with_scenarios_params(output_dir, scenario_calc_name, scenario_dav_name, scenario_global):
    dic_params = {}
    scenarii_calc = up.scenarii_calc_all[up.scenarii_calc_all["NOM SCENARIO"] == scenario_calc_name].copy()
    if len(scenarii_calc) == 0:
        scenarii_calc = pd.DataFrame([["", ""]], columns=["NOM SCENARIO", "TYPE PRODUIT"])

    scenarii_dav = up.scenarii_dav_all[up.scenarii_dav_all["NOM SCENARIO"] == scenario_dav_name].copy()
    if len(scenarii_dav) > 0:
        scenarii_dav = scenarii_dav.drop("NOM SCENARIO", axis=1).rename(columns={"NOM MODELE": "NOM SCENARIO"})
        scenarii_dav['TYPE PRODUIT'] = "DAV SURCOUCHE"
        scenarii_calc = pd.concat([scenarii_calc, scenarii_dav])

    dic_params["DATA_SC_CALC"] = scenarii_calc.to_dict('records')
    dic_params["MAIN_SCENARIO_EVE"] = up.main_sc_eve
    with open(os.path.join(output_dir, scenario_global, 'scenario_params.json'), 'w') as jsonFile:
        json.dump(dic_params, jsonFile)


def get_output_sc_workbook_pn(etab, output_dir, scenario_name):
    _input_pn_file_path, input_pn_wb = ref_services.get_input_pn_workbook(etab)
    pn_wb = resolve_output_workbook(_input_pn_file_path, output_dir, scenario_name)

    if input_pn_wb is not None:
        try_close_workbook(input_pn_wb, scenario_name)

    return pn_wb


def check_and_copy_modele(scenario_name, output_dir, sc_global, modele_type, warning_existence=True,
                          copy_model=True):
    scenarii = check_model_existence(scenario_name, up.models_list)
    if scenarii is None:
        if warning_existence:
            logger.warning(
                "    Vous n'avez pas précisé de scénario de %s pour le scénario %s. Les %s ne pourront pas tourner dans le moteur" % (
                    modele_type, sc_global, modele_type))
        return False
    scenario_name = scenarii[scenarii["TYPE MODELE"] == modele_type]
    if len(scenario_name) == 0:
        if warning_existence:
            logger.warning(
                "    Vous n'avez pas précisé de scénario de %s pour le scénario %s. Les %s ne pourront pas tourner dans le moteur" % (
                    modele_type, sc_global, modele_type))

    modele = scenario_name["MODELE"].iloc[0]
    modele_path = os.path.join(up.source_dir, "MODELES", modele)
    ut.check_version_templates(modele_path, path=modele_path,
                               version=eval("vp.version_modele_%s" % modele_type.lower()), open=True)
    modele_name = modele_type.replace("PN ", "").replace("STOCK ", "")
    if copy_model:
        copy_file_if_not_exist(modele_path, os.path.join(output_dir.format('SC'), sc_global,
                                                         "SC_MOD_%s_" % modele_name +
                                                         os.path.splitext(os.path.basename(modele_path))[0] \
                                                         + '_%s.xlsx' % "_".join(sc_global.split("_")[:-2])))


def check_model_existence(scenario_name, scenarios_df):
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
