from modules.moteur.index_curves.tx_module import Rates_Manager
import modules.moteur.services.projection_pn.pn_loader as plo
import modules.moteur.services.projection_pn.pn_launcher as pla
from modules.moteur.services.intra_groupes.intra_group_module import IG_Manager
from modules.moteur.services.projection_stock.scenario_dependent_layers import Stock_Updater
from modules.moteur.low_level_services.stock_calculator import Stock_Calculator
from modules.moteur.services.surcouche_dav_service.surcouche_dav_service import DAV_Calculator
from modules.moteur.low_level_services.calculator_params import Calculator_Params
import modules.moteur.utils.generic_functions as gf
from modules.moteur.services.projection_pn.pn_compiler import Pn_Compiler
from modules.moteur.services.projection_stock.stock_compiler import Stock_Compiler
import modules.moteur.services.projection_stock.stock_launcher as sm
import logging
import traceback
import time
import os
from pathlib import Path
from modules.moteur.services.projection_pn.ajustements.ajustements_module import Ajustements_Generator

logger = logging.getLogger(__name__)


class Scenario_Simulation():
    def __init__(self, cls_usr, etab, cls_sp, cls_mp, cls_cal):
        self.up = cls_usr
        self.etab = etab
        self.cls_sp = cls_sp
        self.cls_mp = cls_mp
        self.cls_cal = cls_cal

    def set_up_sc_global_classes(self):
        self.ig = IG_Manager(self.up, self.cls_mp)
        self.st_comp = Stock_Compiler(self.up, self.cls_mp, self.ig)
        self.pnc = Pn_Compiler(self.up, self.cls_mp, self.ig)

    def adjust_global_classes_with_sc_params(self):
        self.am = Ajustements_Generator(self.up, self.cls_mp, self.cls_tx)
        self.am.clear_adjustments()
        self.st_comp.set_ajustements(self.am)
        self.st_comp.set_tx_cls(self.cls_tx)
        self.pnc.set_ajustements(self.am)
        self.pnc.set_tx_cls(self.cls_tx)

    def launch_scenarios_sim(self, cls_sp, data_stock, stock_nmd_template, timo):
        self.set_up_sc_global_classes()
        for scenario_name in self.get_sorted_scenarios(self.etab):
            self.scenario_name = scenario_name
            try:
                logger.info("  CHARGEMENT DU SCÉNARIO: " + self.scenario_name)
                self.scen_path, self.sc_output_dir = self.generate_path_and_name_scenario()

                logger.info("     CHARGEMENT DES DONNEES DE SCENARIO")
                cls_pn_loader, self.cls_tx = self.load_scenario_data()

                self.adjust_global_classes_with_sc_params()

                logger.info("     LANCEMENT DU CALCULATEUR SUR LE STOCK")
                cls_calc = Calculator_Params(self.up, self.cls_mp, self.cls_tx, cls_sp)
                cls_calc.load_calculateur_params(stock_nmd_template, data_stock, cls_pn_loader)

                cls_stock_calc = Stock_Calculator(cls_calc, self.etab, self.scenario_name, self.up)
                cls_stock_calc.run_stock_calculator(data_stock, cls_pn_loader)

                logger.info("     CALCUL DE LA SURCOUCHE")
                cls_dav_calc = DAV_Calculator(self.up, cls_calc, self.etab, self.scenario_name)
                data_stock, cls_stock_calc.calculated_stock, cls_stock_calc.data_stock_nmd_dt \
                    = cls_dav_calc.add_dav_surcouche(cls_pn_loader, data_stock, cls_stock_calc)

                logger.info("     MISE A JOUR DU STOCK")
                sdl = Stock_Updater(self.up, self.cls_mp)
                data_stock = sdl.update_stock_with_calculator_results(data_stock, cls_stock_calc)

                dic_stock, dic_stock_ind, dic_updated = self.cls_sp.format_stock(data_stock, self.ig)

                logger.info("     CALCUL DES GAPS et COMPILATION DU STOCK")
                dic_stock_sci, dic_stock_scr \
                    = sm.project_loop_stock(self.st_comp, self.cls_tx, self.cls_cal, self.up, self.cls_mp,
                                            dic_stock, dic_stock_ind, dic_updated, self.scenario_name, self.sc_output_dir)

                if not (self.up.indic_sortie["PN"] + self.up.indic_sortie_eve["PN"] == [] and self.up.indic_sortie[
                    "AJUST"] + self.up.indic_sortie_eve["AJUST"] == []):
                    logger.info("     PROJECTION ET COMPILATION DES PNs")

                    pla.project_ech(self.cls_cal, self.pnc, self.cls_mp, self.up, cls_calc, cls_pn_loader, scenario_name, self.scen_path, self.sc_output_dir)
                    pla.project_ech_prct(self.cls_cal, self.pnc, self.cls_mp, self.up, cls_calc, cls_pn_loader, scenario_name, self.scen_path,
                                         self.sc_output_dir, dic_stock_sci)

                    pla.project_nmd_prct(self.pnc, self.cls_mp, self.up, cls_calc, cls_stock_calc, cls_pn_loader, dic_stock_scr,
                                         scenario_name, self.scen_path, self.sc_output_dir)
                    pla.project_nmd(self.pnc, self.cls_mp, self.up, cls_calc, cls_stock_calc, cls_pn_loader, dic_stock_scr,
                                    scenario_name, self.scen_path, self.sc_output_dir)

                    pla.calculate_ajustements(self.am, self.up, scenario_name, self.sc_output_dir)

                logger.info("     DURÉE DE LA SIMULATION POUR LE SCENARIO " + scenario_name + ": " + str(
                    round(time.time() - timo, 1)) + "seconds")

                self.clean_sc_simulation(cls_pn_loader.dic_pn_nmd, cls_pn_loader.dic_pn_ech,
                                         dic_stock_sci, dic_stock_scr)

                timo = time.time()

            except Exception as e:
                logger.error("     PROBLEME AVEC LE SCENARIO  '" + scenario_name + "' pour l'entité " + self.etab)
                logger.error(e)
                logger.error(traceback.format_exc())

    def get_sorted_scenarios(self, etab):
        if self.up.type_simul["EVE"] or self.up.type_simul['EVE_LIQ']:
            if self.up.main_sc_eve in self.up.list_scenarios[etab]:
                sorted_sc = [self.up.main_sc_eve] + [x for x in self.up.list_scenarios[etab] if
                                                     x != self.up.main_sc_eve]
            else:
                msg_err = "Le scénario principal pour l'EVE '%s' n'est pas présent dans les scénarios disponibles '%s'" % (
                    self.up.main_sc_eve, self.up.list_scenarios[etab])
                logger.error(msg_err)
                raise ValueError(msg_err)
        else:
            sorted_sc = self.up.list_scenarios[etab]

        return sorted_sc

    def generate_path_and_name_scenario(self):
        scen_path = os.path.join(self.up.input_path, self.up.nom_etab, self.scenario_name)
        """ Creation du répertoire de sortie"""
        sc_output_dir = os.path.join(self.up.output_path_etab,
                                     self.scenario_name if self.scenario_name != "" else "DEFAULT_SC")
        Path(sc_output_dir).mkdir(parents=True, exist_ok=True)

        return scen_path, sc_output_dir

    def load_scenario_data(self):
        logger.info("          CHARGEMENT DES DONNÉES de PN")
        """ Lecture et découpage en x blocs des données de PN pour des questions de mémoire """
        files_sc = self.up.scenarios_files[self.etab][self.scenario_name]
        cls_pn_loader = plo.PN_LOADER(self.up, self.cls_mp, self.ig)
        if not (self.up.indic_sortie["PN"] + self.up.indic_sortie_eve["PN"] == [] and
                self.up.indic_sortie["AJUST"] + self.up.indic_sortie_eve["AJUST"] == []):
            cls_pn_loader.load_pn_ech(self.scen_path, files_sc["VOLUME"])
            cls_pn_loader.load_pn_nmd(self.scen_path, files_sc["VOLUME"])

        logger.info("          CHARGEMENT DES COURBES DE TAUX")
        tx = Rates_Manager(self.up, self.cls_cal)
        tx.load_tx_curves(files_sc, self.scen_path, cls_pn_loader)

        return cls_pn_loader, tx

    def clean_sc_simulation(self, dic_pn_nmd, dic_pn_ech, dic_stock_sci, dic_stock_scr):
        gf.clean_vars([dic_pn_ech, dic_pn_nmd, dic_stock_sci, dic_stock_scr])
