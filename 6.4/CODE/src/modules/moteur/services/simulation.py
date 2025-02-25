import pandas as pd
import modules.moteur.index_curves.calendar_module as hf
import utils.excel_utils as ex
import modules.moteur.utils.logger_config as cf
import logging
import modules.moteur.mappings.main_mappings as mp
import modules.moteur.parameters.user_parameters as up
import modules.moteur.services.projection_stock.stock_preparation as sp
from pathlib import Path
import modules.moteur.services.simulation_scenario as sc_sim
from modules.moteur.services.indicateurs_liquidite import lcr_module as lcr
from modules.moteur.services.indicateurs_liquidite import nsfr_module as nsfr
import traceback
import time
import os

""" initiation du logger """
logger = logging.getLogger(__name__)
root = logging.getLogger()
root.handlers = []

def load_main_parameters(args):

    up.load_output_path(args)

    cf.load_logger(up.output_path_usr)

    logger.info("CHARGEMENT DES PARAMETRES UTILISATEUR")
    up.load_users_param(args)

    logger.info("CALCUL DES PARAMETRES DE CALENDRIER")
    hf.load_calendar_coeffs()

    up.get_list_etab(args)

def launch_simulation():
    """ Début de la boucle sur ETAB*scenario """

    for etab in up.list_etab_usr:
        try:
            logger.info("*** DÉBUT DE LA SIMULATION POUR '" + etab + "' ***")
            timo = time.time()

            load_etab_paths(etab)

            logger.info("  CHARGEMENT DES DONNEES DU STOCK")
            stock_wb = load_stock_data()
            stock_nmd_template = load_stock_nmd_template_data()

            logger.info("  CHARGEMENT DES MAPPINGS GLOBAUX")
            load_general_mappings(stock_wb, etab)

            logger.info("  LECTURE DES DONNÉES DE STOCK")
            data_stock = sp.read_stock(stock_wb)

            logger.info("  LANCEMENTS DES SCENARIOS")
            sc_sim.launch_scenarios_sim(etab, data_stock, stock_nmd_template, timo)

            logger.info("*** FIN DE LA SIMULATION POUR '" + etab + "' ***")

        except Exception as e:
            logger.error("  PROBLEME pour l'entité " + etab)
            logger.error(e)
            logger.error(traceback.format_exc())

    root = logging.getLogger()
    root.handlers = []
    os.remove(cf.log_file_path)


def load_etab_paths(etab):
    up.nom_etab = etab
    up.bassin_usr = etab
    up.get_stock_path(etab)
    up.output_path_etab = os.path.join(up.output_path_usr, etab)
    Path(up.output_path_etab).mkdir(parents=True, exist_ok=True)

def load_stock_data():
    stock_wb = ex.try_close_open(up.stock_file_path, read_only=True)
    return stock_wb

def load_stock_nmd_template_data():
    stock_nmd_template_data = pd.read_csv(up.stock_nmd_template_file_path, sep=";", decimal=",", low_memory=False)
    return stock_nmd_template_data

def load_general_mappings(stock_wb, etab):
    mp.load_mappings(etab)
    try:
        lcr.load_lcr_params(stock_wb)
        nsfr.load_nsfr_params(stock_wb)
    except:
        pass
