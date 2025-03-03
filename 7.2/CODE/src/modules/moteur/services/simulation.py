import pandas as pd
from modules.moteur.index_curves.calendar_module import Calendar_Manager
import logging
from mappings.mappings_moteur import Mappings
from modules.moteur.parameters.user_parameters import UserParameters
from modules.moteur.services.projection_stock.stock_preparation import Stock_Preparation
from pathlib import Path
from modules.moteur.services.simulation_scenario import Scenario_Simulation
import traceback
import time
import os
from utils import logger_config as lf

""" initiation du logger """
logger = logging.getLogger(__name__)
root = logging.getLogger()
root.handlers = []


class Simulation():
    def __init__(self, cls_sp):
        self.cls_sp = cls_sp

    def load_main_parameters(self, sc_output_dir=""):
        self.up = UserParameters(self.cls_sp, sc_output_dir=sc_output_dir)

        self.up.load_output_path()

        logger.info("CHARGEMENT DES PARAMETRES UTILISATEUR")
        self.up.load_users_param()

        logger.info("  CHARGEMENT DES MAPPINGS GLOBAUX")
        self.mp = Mappings(self.up)
        self.mp.load_mappings()

        logger.info("CALCUL DES PARAMETRES DE CALENDRIER")
        self.cl = Calendar_Manager(self.up, self.mp)
        self.cl.load_calendar_coeffs()

        self.up.get_list_etab()

    def launch_simulation(self, cls_sp, sc_output_dir=""):
        """ Début de la boucle sur ETAB*scenario """
        try:
            self.load_main_parameters(sc_output_dir=sc_output_dir)
            for etab in self.up.list_etab_usr:
                try:
                    logger.info("*** DÉBUT DE LA SIMULATION POUR '" + etab + "' ***")
                    timo = time.time()

                    self.load_etab_paths(etab)

                    logger.info("  CHARGEMENT DES DONNEES DU STOCK")
                    stock_data = self.load_stock_data()
                    stock_nmd_template = self.load_stock_nmd_template_data()

                    logger.info("  LECTURE DES DONNÉES DE STOCK")
                    sp = Stock_Preparation(self.up, self.mp)
                    data_stock = sp.read_stock(stock_data)

                    logger.info("  LANCEMENTS DES SCENARIOS")
                    sc_sim = Scenario_Simulation(self.up, etab, sp, self.mp, self.cl)
                    sc_sim.launch_scenarios_sim(cls_sp, data_stock, stock_nmd_template, timo)

                    logger.info("*** FIN DE LA SIMULATION POUR '" + etab + "' ***")

                except Exception as e:
                    logger.error("  PROBLEME pour l'entité " + etab)
                    logger.error(e)
                    logger.error(traceback.format_exc())


        except Exception:
            raise Exception

        finally:
            try:
                if cls_sp.mode == "MOTEUR":
                    lf.copy_to_dist_dir(self.up.output_path_usr)
            except:
                pass

    def load_etab_paths(self, etab):
        self.up.nom_etab = etab
        self.up.bassin_usr = etab
        self.up.get_stock_path(etab)
        self.up.output_path_etab = os.path.join(self.up.output_path_usr, etab)
        Path(self.up.output_path_etab).mkdir(parents=True, exist_ok=True)

    def load_stock_data(self):
        stock_data = pd.read_csv(self.up.stock_file_path, sep=";", decimal=",", low_memory=False)
        return stock_data

    def load_stock_nmd_template_data(self):
        stock_nmd_template_data = pd.read_csv(self.up.stock_nmd_template_file_path, sep=";", decimal=",",
                                              low_memory=False)
        return stock_nmd_template_data
