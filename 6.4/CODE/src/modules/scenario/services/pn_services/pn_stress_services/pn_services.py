import logging
from utils import excel_utils
from modules.scenario.parameters import user_parameters as up
from modules.scenario.referentials.general_parameters import *
import modules.scenario.services.referential_file_service as rf
import utils.general_utils as ut
from modules.scenario.referentials.transco_pn import transco_pn_name_to_range_name, transco_pn_num_name_to_range_name
import modules.scenario.referentials.general_parameters as gp
from modules.scenario.controller.pn_controllers.stress_pn_controller import stress_scenario_pn
from modules.scenario.controller.pn_controllers.ajout_pn_controller import ajout_pn_scenario
from modules.scenario.services.excel_template_output_services.pn_output_file_service import print_pn_template_file
logger = logging.getLogger(__name__)

def process_scenarios_stress(wb, pn_type, scenario_stress_pn, etab):
    try:
        stressed_pn_df, non_applied_stress = stress_scenario_pn(wb, pn_type, scenario_stress_pn, etab)
        if len(rf.pn_df[pn_type]) > len(stressed_pn_df) and "%" in pn_type:
            stressed_pn_df.reset_index(inplace=True)
            print_pn_template_file(stressed_pn_df, wb, transco_pn_name_to_range_name[pn_type])
        else:
            stressed_pn_df.reset_index(drop=True, inplace=True)
            print_pn_template_file(stressed_pn_df, wb, transco_pn_num_name_to_range_name[pn_type])


    except Exception as e:
        logger.error('Erreur pendant le traitement des stress PN  de type ' + pn_type)
        logger.error(e, exc_info=True)

def process_scenarios_ajout_pn(wb, pn_type, scenario_stress_pn, mapping_wb, etab):
    try:
        ajout_pn_scenario(wb, pn_type, scenario_stress_pn, mapping_wb, etab)

    except Exception as e:
        logger.error('Erreur pendant le traitement des stress PN  de type ' + pn_type)
        logger.error(e, exc_info=True)

def clean_non_stressed_pn_prcents(wb, type_pn):
    list_type_pn = [item for sublist in [x.split("&") for x in type_pn.split(",")] for item in sublist]
    pn_prcent_to_erase = [x for x in ['PN ECH%', 'NMD%'] if x.replace("PN ", "") not in list_type_pn]
    for pn_type in pn_prcent_to_erase:
        excel_utils.clear_range_content(wb, transco_pn_name_to_range_name[pn_type], 2)
    return pn_prcent_to_erase


def process_scenario_pn_stress(pn_stress_list, scenario_rows, wb, type_pn, etab):
    all_scenarios = scenario_rows.merge(pn_stress_list, left_on="SC PN", right_on="NOM STRESS PN")
    type_pn_unique = all_scenarios[CN_TYPE_PN].unique().tolist()
    pn_prcent_erased = clean_non_stressed_pn_prcents(wb, type_pn)
    type_pn_unique2 = [x for x in type_pn_unique if x not in pn_prcent_erased]
    if len(type_pn_unique) != len(type_pn_unique2):
        logger.info("    Les PNs suivantes n'ont pas été stressées par ces PNs n'ont pas été activées : " + str(
            [x for x in type_pn_unique if x not in type_pn_unique2]))

    if type_pn_unique2 != []:
        for type_pn in type_pn_unique2:
            scenarios = all_scenarios[all_scenarios[CN_TYPE_PN] == type_pn]
            if len(scenarios) > 0:
                logger.info('    Calcul des PN stressées pour le type: ' + type_pn)
                process_scenarios_stress(wb, type_pn, scenarios, etab)
    else:
        logger.info("    Scénario sans stress de PN")



def process_scenario_pn_ajout(pn_ajout_list, scenario_rows, wb, type_pn, mapping_wb, etab):
    all_scenarios = scenario_rows.merge(pn_ajout_list, left_on="SC PN", right_on="NOM SC AJOUT PN")
    type_pn_unique = all_scenarios[CN_TYPE_PN].unique().tolist()
    type_pn_unique2 = [x for x in type_pn_unique if x.replace("PN ","") in type_pn]
    if len(type_pn_unique) != len(type_pn_unique2):
        logger.info("    Les PNs suivantes n'ont pas été ajoutées par ces PNs n'ont pas été activées : " + str(
            [x for x in type_pn_unique if x not in type_pn_unique2]))

    if type_pn_unique2 != []:
        for type_pn in type_pn_unique2:
            scenarios = all_scenarios[all_scenarios[CN_TYPE_PN] == type_pn]
            if len(scenarios) > 0:
                logger.info('    Ajout des PNs pour le type: ' + type_pn)
                process_scenarios_ajout_pn(wb, type_pn, scenarios, mapping_wb, etab)
    else:
        logger.info("    Scénario sans ajout de PN")


def deactivate_non_requested_pns(pn_wb, type_pn):
    all_pns = ["PN ECH", "PN ECH%", "NMD", "NMD%"]
    activated_pns = ut.flatten([pn.split("&") for pn in type_pn.split(",")])
    for pn in all_pns:
        if pn.replace("PN", "").replace(" ","") not in activated_pns:
            pn_wb.Sheets(pn).Unprotect(excel_utils.EXCEL_PASS_WORD)
            pn_wb.Sheets(pn).Rows("2:" + str(pn_wb.Sheets(pn).Rows.Count)).ClearContents()
            try:
                pn_wb.Sheets(pn).Visible = False
            except:
                pass
            pn_wb.Sheets(pn).Protect(excel_utils.EXCEL_PASS_WORD)

def get_type_pn_a_activer(etab):
    etab_params = up.pn_a_activer_df[up.pn_a_activer_df['ENTITE'] == etab]
    if etab_params.empty:
        etab_params = up.pn_a_activer_df[up.pn_a_activer_df['ENTITE'] == 'DEFAULT']
    type_pn = etab_params['PN A ACTIVER'].iloc[0]
    return type_pn

