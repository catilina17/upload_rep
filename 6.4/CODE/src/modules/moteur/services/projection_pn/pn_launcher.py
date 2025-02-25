import modules.moteur.services.projection_pn.ajustements.ajustements_module as am
from modules.moteur.services.projection_pn.nmd.nmd_pn_projecter import PN_NMD_PROJECTER
from modules.moteur.services.projection_pn.ech.ech_pn_projecter import PN_ECH_PROJECTER
import modules.moteur.services.projection_pn.pn_compiler as pc
import modules.moteur.parameters.user_parameters as up
import utils.general_utils as gu
import logging

logger = logging.getLogger(__name__)


def project_nmd(cls_calc, cls_stock_calc, cls_pn_loader,
                dic_stock_scr, name_scenario, scen_path, new_dir):
    if "nmd" in cls_pn_loader.dic_pn_nmd:
        logger.info("       PROJECTION ET COMPILATION DES PN NMD")
        dic_pn_sc = {}
        cls_nmd = PN_NMD_PROJECTER(cls_calc, cls_stock_calc, cls_pn_loader, scen_path, name_scenario, "")
        cls_nmd.project_pn_nmd(dic_pn_sc, dic_stock_scr)
        pc.compile_pn(dic_pn_sc, "nmd", name_scenario, new_dir, dic_stock_scr=dic_stock_scr)


def project_nmd_prct(cls_calc, cls_stock_calc, cls_pn_loader,
                     dic_stock_scr, name_scenario, scen_path, new_dir):
    if "nmd%" in cls_pn_loader.dic_pn_nmd:
        logger.info("       PROJECTION ET COMPILATION DES PN NMD%")
        dic_pn_sc = {}
        cls_nmd = PN_NMD_PROJECTER(cls_calc, cls_stock_calc, cls_pn_loader, scen_path, name_scenario,
                                   nmd_prct="%")
        cls_nmd.project_pn_nmd(dic_pn_sc, dic_stock_scr)
        pc.compile_pn(dic_pn_sc, "nmd", name_scenario, new_dir, dic_stock_scr=dic_stock_scr)

def project_ech(cls_calc, cls_pn_loader, name_scenario, scen_path, new_dir):
    if "ech" in cls_pn_loader.dic_pn_ech:
        logger.info("       PROJECTION ET COMPILATION DES PN ECH")
        data_ech = gu.chunkized_data(cls_pn_loader.dic_pn_ech["ech"], 4 * 400)  # multiple de 4
        """ Boucle sur les blocs de PN ECH % """
        for i in range(0, len(data_ech)):
            dic_pn_sc = {}
            cls_ech = PN_ECH_PROJECTER(cls_calc, cls_pn_loader, scen_path, name_scenario)
            cls_ech.project_pn_ech(data_ech[i].copy(), dic_pn_sc)
            pc.compile_pn(dic_pn_sc, "ech", name_scenario, new_dir)


def project_ech_prct(cls_calc, cls_pn_loader, name_scenario, scen_path, new_dir, dic_stock_sci):
    if "ech%" in cls_pn_loader.dic_pn_ech:
        logger.info("       PROJECTION ET COMPILATION DES PN ECH%")
        data_ech_prct = gu.chunkized_data(cls_pn_loader.dic_pn_ech["ech%"], 4 * 400)  # multiple de 4
        """ Boucle sur les blocs de PN ECH % """
        for i in range(0, len(data_ech_prct)):
            dic_pn_sc = {}
            cls_ech = PN_ECH_PROJECTER(cls_calc, cls_pn_loader, scen_path, name_scenario, ech_prct="%")
            cls_ech.project_pn_ech(data_ech_prct[i].copy(), dic_pn_sc, dic_stock_sci=dic_stock_sci)
            pc.compile_pn(dic_pn_sc, "ech%", name_scenario, new_dir)


def calculate_ajustements(name_scenario, new_dir):
    logger.info("    PROJECTION ET COMPILATION DES AJUSTEMENTS")
    dic_ajust = {}
    dic_ajust = am.add_adjustments(dic_ajust)
    if up.type_simul["EVE"] or up.type_simul["EVE_LIQ"]:
        dic_ajust = am.add_adjustments(dic_ajust, type="EVE")
    am.calculate_indics(dic_ajust, name_scenario, new_dir)
    am.data_adj = []
