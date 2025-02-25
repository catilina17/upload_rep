import modules.moteur.index_curves.tx_module as tx
import modules.moteur.services.projection_pn.pn_loader as plo
import modules.moteur.services.projection_pn.pn_launcher as pla
import modules.moteur.mappings.main_mappings as mp
from shutil import copyfile
import modules.moteur.services.projection_stock.stock_preparation as sp
import modules.moteur.services.projection_stock.scenario_dependent_layers as sdl
from  modules.moteur.low_level_services.stock_calculator import Stock_Calculator
from modules.moteur.services.surcouche_dav_service.surcouche_dav_service import DAV_Calculator
from modules.moteur.low_level_services.calculator_params import Calculator_Params
import modules.moteur.mappings.sc_mappings as smp
import modules.moteur.parameters.general_parameters as gp
import os
import utils.excel_utils as ex
import modules.moteur.parameters.user_parameters as up
import modules.moteur.utils.generic_functions as gf
import modules.moteur.services.projection_stock.stock_launcher as sm
import logging
import traceback
import time
import modules.moteur.utils.logger_config as cf
from pathlib import Path
import modules.moteur.services.projection_pn.ajustements.ajustements_module as am

logger = logging.getLogger(__name__)


def launch_scenarios_sim(etab, data_stock, stock_nmd_template, timo):
    for scenario_name in get_sorted_scenarios(etab):
        try:
            am.clear_adjustments()

            logger.info("  CHARGEMENT DU SCÉNARIO: '%s'" % scenario_name)
            scen_path, sc_output_dir = generate_path_and_name_scenario(scenario_name)

            logger.info("     CHARGEMENT DES DONNEES DE scenario")
            cls_pn_loader = load_scenario_data(etab, scenario_name, scen_path)

            logger.info("     COMPILATION DES DONNÉES DE SCÉNARIO de TAUX")
            save_index_rate_data(sc_output_dir, scenario_name)

            logger.info("     LANCEMENT DU CALCULATEUR SUR LE STOCK")
            cls_calc = Calculator_Params()
            cls_calc.load_calculateur_params(stock_nmd_template, data_stock, cls_pn_loader)

            cls_stock_calc = Stock_Calculator(cls_calc)
            cls_stock_calc.run_stock_calculator(scenario_name, etab, data_stock, cls_pn_loader)

            logger.info("     CALCUL DE LA SURCOUCHE")
            cls_dav_calc = DAV_Calculator(cls_calc)
            data_stock, cls_stock_calc.calculated_stock, cls_stock_calc.data_stock_nmd_dt \
                = cls_dav_calc.add_dav_surcouche(etab, scenario_name, cls_pn_loader, data_stock, cls_stock_calc)

            logger.info("     MISE A JOUR DU STOCK")
            data_stock = sdl.update_stock_with_calculator_results(data_stock, cls_stock_calc)

            dic_stock, dic_stock_ind, dic_updated = sp.format_stock(data_stock)

            logger.info("     PROJECTION ET COMPILATION DU STOCK")
            dic_stock_sci, dic_stock_scr = sm.project_loop_stock(dic_stock, dic_stock_ind,
                                                                 dic_updated, scenario_name, sc_output_dir)

            if not (up.indic_sortie["PN"] + up.indic_sortie_eve["PN"] == [] and up.indic_sortie["AJUST"] +
                    up.indic_sortie_eve["AJUST"] == []):
                logger.info("     PROJECTION ET COMPILATION DES PNs")

                pla.project_ech(cls_calc, cls_pn_loader, scenario_name, scen_path, sc_output_dir)
                pla.project_ech_prct(cls_calc, cls_pn_loader, scenario_name, scen_path, sc_output_dir, dic_stock_sci)

                pla.project_nmd_prct(cls_calc, cls_stock_calc, cls_pn_loader, dic_stock_scr,
                                     scenario_name, scen_path, sc_output_dir)
                pla.project_nmd(cls_calc, cls_stock_calc, cls_pn_loader, dic_stock_scr,
                                scenario_name, scen_path, sc_output_dir)

                pla.calculate_ajustements(scenario_name, sc_output_dir)

            logger.info("     DURÉE DE LA SIMULATION POUR LE scenario " + scenario_name + ": " + str(
                round(time.time() - timo, 1)) + "seconds")

            clean_sc_simulation(sc_output_dir, cls_pn_loader.dic_pn_nmd, cls_pn_loader.dic_pn_ech,
                                dic_stock_sci, dic_stock_scr)

            timo = time.time()

        except Exception as e:
            logger.error("     PROBLEME AVEC LE scenario  '" + scenario_name + "' pour l'entité " + etab)
            logger.error(e)
            logger.error(traceback.format_exc())

    mp.map_wb.Close(False)


def get_sorted_scenarios(etab):
    if up.type_simul["EVE"] or  up.type_simul['EVE_LIQ']:
        if up.main_sc_eve in up.list_scenarios[etab]:
            sorted_sc = [up.main_sc_eve] + [x for x in up.list_scenarios[etab] if x != up.main_sc_eve]
        else:
            msg_err = "Le scénario principal pour l'EVE '%s' n'est pas présent dans les scénarios disponibles '%s'" % (
            up.main_sc_eve, up.list_scenarios[etab])
            logger.error(msg_err)
            raise ValueError(msg_err)
    else:
        sorted_sc = up.list_scenarios[etab]

    return sorted_sc


def generate_path_and_name_scenario(scenario_name):
    scen_path = os.path.join(up.input_path, up.nom_etab, scenario_name)
    """ Creation du répertoire de sortie"""
    sc_output_dir = os.path.join(up.output_path_etab, scenario_name if scenario_name != "" else "DEFAULT_SC")
    Path(sc_output_dir).mkdir(parents=True, exist_ok=True)

    return scen_path, sc_output_dir


def load_scenario_data(etab, scenario_name, scen_path):
    logger.info("          CHARGEMENT DES DONNÉES de PN")
    """ Lecture et découpage en x blocs des données de PN pour des questions de mémoire """
    files_sc = up.scenarios_files[etab][scenario_name]
    path_file_sc_vol = os.path.join(scen_path, files_sc["VOLUME"][0])
    wb_vol = ex.try_close_open(path_file_sc_vol, read_only=True)
    cls_pn_loader = plo.PN_LOADER()
    if not (up.indic_sortie["PN"] + up.indic_sortie_eve["PN"] == [] and up.indic_sortie["AJUST"] + up.indic_sortie_eve[
        "AJUST"] == []):
        cls_pn_loader.load_pn_ech(wb_vol)
        cls_pn_loader.load_pn_nmd(wb_vol)

    logger.info("          CHARGEMENT DES MAPPINGS DE SCÉNARIO")
    smp.load_sc_mappings(etab, scenario_name, scen_path, files_sc)

    logger.info("          CHARGEMENT DES COURBES DE TAUX")
    tx.load_tx_curves(files_sc, scen_path, cls_pn_loader)

    wb_vol.Close(False)

    return cls_pn_loader


def save_index_rate_data(sc_output_dir, name_scenario):
    tx.fill_and_save_sc_compil(sc_output_dir, gp.nom_compil_sc, name_scenario, up.bassin_usr)


def clean_sc_simulation(sc_output_dir, dic_pn_nmd, dic_pn_ech, dic_stock_sci, dic_stock_scr):
    """ TRANSFERT DU FICHIER DE LOG VERS LE REPERTOIRE DE SORTIE """
    copyfile(cf.log_file_path, os.path.join(sc_output_dir, "zz_log_" + cf.time_log + ".txt"))
    with open(cf.log_file_path, 'w'):
        pass
    gf.clean_vars([dic_pn_ech, dic_pn_nmd, dic_stock_sci, dic_stock_scr])
    tx.zc_liq_curves = {}
