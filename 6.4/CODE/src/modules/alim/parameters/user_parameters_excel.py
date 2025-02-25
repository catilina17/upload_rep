import datetime
import dateutil
import pandas as pd
import modules.alim.parameters.general_parameters as gp
import modules.alim.parameters.user_parameters as up
from params import sources_params as sp
from params import version_params as vp
import logging
from pathlib import Path
import os
import utils.excel_utils as ex
import utils.general_utils as gu
import modules.alim.parameters.RZO_params as rzo_p
import modules.alim.parameters.NTX_SEF_params as ntx_sef_p

logger = logging.getLogger(__name__)
global accueil_ws

def get_user_main_params():
    global accueil_ws
    accueil_ws = ex.xl_interface.Worksheets("ALIM_MAIN")

    vp.version_sources = str(accueil_ws.Range("_type_source").Value).replace(" ", "").lower()

    up.path_map_mis_template_file = '\\'.join(ex.interface_name.split('\\')[0:-1]) + sp.path_map_mis_template_file
    up.path_stock_template_file = '\\'.join(ex.interface_name.split('\\')[0:-1]) + sp.path_stock_template_file
    up.path_sc_tx_template_file = '\\'.join(ex.interface_name.split('\\')[0:-1]) + sp.path_sc_tx_template_file
    up.path_sc_vol_template_file = '\\'.join(ex.interface_name.split('\\')[0:-1]) + sp.path_sc_vol_template_file
    up.path_sc_lcr_nsfr_template_file = '\\'.join(ex.interface_name.split('\\')[0:-1]) + sp.path_sc_lcr_nsfr_template_file

    gu.check_version_templates("MAPPINGS_MANQUANTS.xlsb", path=up.path_map_mis_template_file,  open=True, version=vp.version_AUTRES_TEMP)
    gu.check_version_templates("STOCK_TEMPLATE.xlsb", path=up.path_stock_template_file, open=True, version=vp.version_STOCK_TMP)
    gu.check_version_templates("SC_TAUX_TEMPLATE.xlsb", path=up.path_sc_tx_template_file, open=True, version=vp.version_SC_TX)
    gu.check_version_templates("SC_VOLUME_TEMPLATE.xlsb", path=up.path_sc_vol_template_file, open=True, version=vp.version_SC_VOL)
    gu.check_version_templates("SC_LCR_NSFR_TEMPLATE.xlsb", path=up.path_sc_lcr_nsfr_template_file, open=True, version=vp.version_AUTRES_TEMP)

    up.dar = dateutil.parser.parse(str(accueil_ws.Range("_DAR_ALIM").Value)).replace(tzinfo=None)
    up.name_run = accueil_ws.Range("_Comm").Value if accueil_ws.Range(
        "_Comm").Value is not None else ""

    up.sc_ref_nmd = str(accueil_ws.Range("_SC_REF_NMD").Value)

    up.etabs = str(accueil_ws.Range("_BASSINS").Value).replace(" ", "").split(",")
    up.sources_folder = accueil_ws.Range("_MULTIBASSIN_SOURCES_PATH").Value
    up.map_lcr_nsfr_g = True if accueil_ws.Range("_MAPPER_RAY").Value == "OUI" else False

    up.rate_file_path = os.path.join(up.sources_folder , "RATE_INPUT", accueil_ws.Range("_RATE_INPUT").Value)
    up.gu.check_version_templates(up.rate_file_path.split("\\")[-1], version = vp.version_rate, path=up.rate_file_path, open=True)
    up.liq_file_path = os.path.join(up.sources_folder , "RATE_INPUT", accueil_ws.Range("LIQ_INPUT").Value)
    gu.check_version_templates(up.liq_file_path.split("\\")[-1], version = vp.version_rate, path=up.liq_file_path, open=True)

    if len([x for x in up.etabs if x not in gp.NTX_FORMAT]) > 0:
        up.modele_nmd_file_path = os.path.join(up.sources_folder , "MODELES", accueil_ws.Range("_MODELE_NMD").Value)
        gu.check_version_templates(up.modele_nmd_file_path.split("\\")[-1], version = vp.version_modele_nmd, path=up.modele_nmd_file_path, open=True)

        up.modele_pel_file_path = os.path.join(up.sources_folder , "MODELES", accueil_ws.Range("_MODELE_PEL").Value)
        gu.check_version_templates(up.modele_pel_file_path.split("\\")[-1], version = vp.version_modele_pel, path=up.modele_pel_file_path, open=True)

    up.detail_ig = True if accueil_ws.Range("DET_IG").Value == "OUI" else False

    up.mapp_file = up.sources_folder + sp.mapp_file
    if not os.path.exists(up.mapp_file):
        raise ValueError("Le fichier de MAPPING est absent du dossier des SOURCES: " + str(up.mapp_file))

    up.type_simul = accueil_ws.Range("_TYPE_SIMUL").Value
    up.mois_inter_trim = accueil_ws.Range("_MOIS_INTERTRIMESTRIEL").Value
    up.chemin_compils = accueil_ws.Range("_CHEMIN_COMPILS").Value
    up.output_folder = accueil_ws.Range("_OUTPUT_PATH_ALIM").Value



def generate_etab_parameters_excel(etab):
    if etab in gp.NTX_FORMAT:
        ntx_sef_p.perim_ntx = accueil_ws.Range("_PERIM_NTX").Value
        logger.info("Le périmètre choisi est : %s" % ntx_sef_p.perim_ntx)
        ntx_sef_p.depart_decale = accueil_ws.Range("_DEPART_DECALE_NTX").Value
        logger.info("L'option de départ décalé est : %s" % ntx_sef_p.depart_decale)
        ntx_sef_p.del_enc_dar_zero = accueil_ws.Range("_DAR_ZERO").Value
        logger.info("Les encours nuls en DAR seront filtrés : %s" % ntx_sef_p.del_enc_dar_zero)
