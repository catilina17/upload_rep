# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:01:46 2020

CHARGEMENT DES PARAMETRES UTILISATEURS
"""

import modules.moteur.parameters.general_parameters as gp
import modules.moteur.parameters.authorized_users as au
import getpass
import modules.moteur.utils.generic_functions as gf
import dateutil
import pandas as pd
import json
from pathlib import Path
import datetime
import os
import logging
import re
import modules.moteur.indicators.dependances_indic as di

logger = logging.getLogger(__name__)


class UserParameters():
    def __init__(self, sp_cls, sc_output_dir=""):
        self.simul_cls = sp_cls
        self.om = sp_cls.output_cls
        self.sm = sp_cls.sources_cls
        self.dar = sp_cls.dar
        self.etabs = sp_cls.entities_list
        self.source_path = self.sm.sources_folder
        self.output_folder = self.om.output_folder
        self.config_cls = sp_cls.config_cls
        self.names = self.config_cls.names_ex
        self.nb_mois_proj_out = sp_cls.nb_months_proj
        self.sc_output_dir = sc_output_dir

    def load_users_param(self):
        self.get_input_path()
        self.get_source_path()
        self.load_projection_period_params()
        self.load_ajustements_options()
        self.load_output_indicators()
        self.load_output_axes()
        self.load_conv_ecoulements_options()
        self.load_tla_refixing_params()
        self.load_pn_max_month()
        self.load_compil_format()
        self.get_user_list_etabs()
        self.parse_output_indicators()
        self.display_user_params()

    def get_input_path(self):
        if self.sc_output_dir != "":
            self.input_path = self.sc_output_dir
        else:
            self.input_path = self.config_cls.get_value_from_named_ranged(self.names.NAME_RANGE_SCENARIO_FOLDER)

    def get_source_path(self):
        self.source_path = self.config_cls.get_value_from_named_ranged(self.names.NAME_RANGE_SOURCES_FOLDER)

    def get_stock_path(self, etab):
        list_files = [x[2] for x in os.walk(os.path.join(self.input_path, etab))][0]
        for fname in list_files:
            if gp.stock_tag in fname:
                self.stock_folder_path = os.path.join(self.input_path, etab)
                self.stock_file_path = os.path.join(self.input_path, etab, fname)
            if gp.stock_nmd_template_tag in fname:
                self.stock_nmd_template_file_path = os.path.join(self.input_path, etab, fname)

    def get_user_list_etabs(self):
        self.list_all_etab = self.config_cls.get_dataframe_from_named_ranged(self.names.NAME_RANGE_ALL_ETABS).iloc[:,0].values.tolist()
        self.list_etab_usr = self.config_cls.get_value_from_named_ranged(self.names.NAME_RANGE_ENTITIES)

    def load_output_path(self):
        now = datetime.datetime.now()
        now = now.strftime("%Y%m%d.%H%M.%S")
        new_dir = str(self.dar.year) + str(self.dar.month) + str(self.dar.day) + "_EXEC-" + str(now)
        self.output_path_usr = os.path.join(self.output_folder, "MOTEUR_DAR-" + new_dir)
        Path(self.output_path_usr).mkdir(parents=True, exist_ok=True)


    def load_tla_refixing_params(self):
        """ TX REFIXING """
        self.retraitement_tla = self.config_cls.get_value_from_named_ranged(self.names.ng_tla_retraitement)
        self.retraitement_tla = True if self.retraitement_tla.upper() == "OUI" else False
        self.date_refix_tla = self.config_cls.get_value_from_named_ranged(gp.ng_date_refix)
        self.date_refix_tla = dateutil.parser.parse(str(self.date_refix_tla)).replace(tzinfo=None)
        self.mois_refix_tla = self.date_refix_tla.year * 12 + self.date_refix_tla.month - self.dar.year * 12 - self.dar.month
        try:
            self.freq_refix_tla = int(self.config_cls.get_value_from_named_ranged(gp.ng_tla_retraitement))
        except:
            self.freq_refix_tla = 4
        if self.mois_refix_tla < 1 and self.retraitement_tla:
            logger.error("   La date de refixing TLA est inférieure ou égale à la DAR")
            raise ValueError("   La date de refixing TLA est inférieure ou égale à la DAR")

        return self.retraitement_tla, self.mois_refix_tla, self.freq_refix_tla

    def load_output_indicators(self):
        """ DETERMINATION DES INDICATEURS SORTIES"""
        self.data_indic = self.config_cls.get_dataframe_from_named_ranged(self.config_cls.names_ex.ng_ind_sortie)
        self.type_eve = self.config_cls.get_value_from_named_ranged(self.config_cls.names_ex.nc_type_eve)
        self.data_indic_eve = self.config_cls.get_dataframe_from_named_ranged(self.config_cls.names_ex.ng_ind_sortie_eve)

    def load_ajustements_options(self):
        # global ajust_only, spread_liq, courbes_ajust

        """ NE SORTIR QUE LES AJUSTEMENTS"""
        self.ajust_only = self.config_cls.get_value_from_named_ranged(gp.ng_ajust_only)
        self.ajust_only = False if self.ajust_only == "NON" else True

        """ SPREAD LIQ AJUST """
        self.spread_liq = self.config_cls.get_dataframe_from_named_ranged(self.config_cls.names_ex.ng_spread_ajust)

        """ COURBES AJUST """
        self.courbes_ajust = (self.config_cls.get_dataframe_from_named_ranged(self.config_cls.names_ex.ng_courbes_ajust)
                              .set_index("DEVISE"))

    def load_output_axes(self):
        self.cols_sortie_excl = self.config_cls.get_dataframe_from_named_ranged(self.config_cls.names_ex.ng_col_sortie)
        self.cols_sortie_excl = self.cols_sortie_excl[self.cols_sortie_excl[gp.nc_col_sort_restituer] == "NON"][
            gp.nc_axe_nom].values.tolist()

    def load_pn_max_month(self):
        mois_pn = self.config_cls.get_dataframe_from_named_ranged(self.config_cls.names_ex.nc_nb_mois_pn)
        self.max_month_pn = dict([(str(i).lower(), int(j)) for i, j in zip(mois_pn["PN"], mois_pn["MOIS"])])

    def load_projection_period_params(self):
        if self.nb_mois_proj_out == "" or self.nb_mois_proj_out is None:
            self.nb_mois_proj_out = 60
        try:
            float(str(self.nb_mois_proj_out))
            self.nb_mois_proj_out = int(self.nb_mois_proj_out)
        except ValueError:
            self.nb_mois_proj_out = 60

        if self.nb_mois_proj_out > gp.max_months:
            raise ValueError("La projection ne peut excéder 240 mois")

        self.nb_annees_usr = int(self.nb_mois_proj_out / 12)

    def load_conv_ecoulements_options(self):
        self.force_gp_liq = self.config_cls.get_value_from_named_ranged(self.config_cls.names_ex.ng_force_gp_liq)
        self.force_gp_liq = False if self.force_gp_liq == "NON" else True

        """ OPTION POUR FORCER LES GAPS DES NMDS """
        self.force_gps_nmd = self.config_cls.get_value_from_named_ranged(self.config_cls.names_ex.ng_force_gps_nmd)
        self.force_gps_nmd = False if self.force_gps_nmd == "NON" else True

    def load_compil_format(self):
        compil_format = str(self.config_cls.get_value_from_named_ranged(self.config_cls.names_ex.ng_compil_sep)).strip()
        self.compil_decimal = re.findall(r'"([^"]*)"', compil_format)[1]
        self.compil_sep = re.findall(r'"([^"]*)"', compil_format)[0]
        self.merge_compil = self.config_cls.get_value_from_named_ranged(self.config_cls.names_ex.ng_compil_merge)
        self.merge_compil = self.merge_compil.strip() == "OUI"

    def check_user_credentials(self):
        if True:
            user_win = os.getlogin().upper()
            user_win2 = str(getpass.getuser()).upper()
            if not gf.begin_with_list(au.users_win, user_win) and gf.begin_with_list(au.users_win, user_win2):
                logger.error("YOU ARE NOT AN AUTHORIZED USER for THIS APP")
                raise ValueError("YOU ARE NOT AN AUTHORIZED USER for THIS APP")

    def get_list_etab(self,):
        self.list_scenarios = {}
        self.scenarios_files = {}
        self.scenarios_params = {}

        if not os.path.isdir(self.input_path):
            raise ValueError("Le dossier source " + self.input_path + " n'existe pas")

        if self.list_etab_usr == "" or self.list_etab_usr is None:
            raise ValueError("Veuillez préciser au moins 1 entité à simuler")

        self.list_etab_usr = self.list_etab_usr.replace(" ", "").split(",")

        list_etab_usr2 = [x for x in self.list_etab_usr]

        for etab in list_etab_usr2:
            self.list_scenarios[etab] = []
            self.scenarios_files[etab] = {}
            self.scenarios_params[etab] = {}
            ino = True
            if etab not in self.list_all_etab:
                logger.warning("L'entité " + etab + " n'existe pas dans la liste des établissements pris en charge")
                self.list_etab_usr.remove(etab)
                ino = False

            if not os.path.isdir(os.path.join(self.input_path, etab)) and ino:
                logger.warning("Le dossier relatif à l'entité " + etab + " n'existe pas dans le dossier source indiqué")
                self.list_etab_usr.remove(etab)
                ino = False

            if ino:
                stock_file = [x[2] for x in os.walk(os.path.join(self.input_path, etab))][0]
                self.list_scenarios[etab] = [x[1] for x in os.walk(os.path.join(self.input_path, etab))][0]
                for sc in self.list_scenarios[etab]:
                    list_files = [x for x in os.walk(os.path.join(self.input_path, etab, sc))][0][2]
                    self.scenarios_files[etab][sc] = {}
                    self.scenarios_files[etab][sc]["SC_PARAMS"] = [x for x in list_files if gp.sc_params_tag in x]

                    if len(self.scenarios_files[etab][sc]["SC_PARAMS"]) == 0:
                        msg = "Le fichier de paramètres scénario est manquant"
                        logger.error(msg)
                        raise ValueError(msg)
                    else:
                        file = os.path.join(self.input_path, etab, sc, self.scenarios_files[etab][sc]["SC_PARAMS"][0])
                        with open(file) as f:
                            dic_json = json.load(f)
                            self.scenarios_params[etab][sc] = pd.DataFrame(dic_json[gp.data_sc_tag])
                            # if not args.use_json:
                            self.main_sc_eve = dic_json[gp.main_sc_eve_tag]

                        list_sc_volume = [x for x in os.walk(os.path.join(self.input_path, etab, sc,
                                                                          self.om.sc_volume_folder_tag))][0][2]
                        list_sc_modele = [x for x in os.walk(os.path.join(self.input_path, etab, sc, "MODELES"))][0][2]
                        list_sc_taux = [x for x in os.walk(os.path.join(self.input_path, etab, sc, self.om.sc_taux_folder_tag))][0][2]

                        if os.path.exists(os.path.join(self.input_path, etab, sc, self.om.sc_lcr_nsfr_folder_tag)):
                            list_sc_lcr_nsfr = \
                            [x for x in os.walk(os.path.join(self.input_path, etab, sc,  self.om.sc_lcr_nsfr_folder_tag))][0][2]
                            self.scenarios_files[etab][sc]["LCR_NSFR"] = [x for x in list_sc_lcr_nsfr if
                                                                     gp.sc_lcr_nsfr_tag in x]

                        self.scenarios_files[etab][sc]["MODELE_NMD"] = [x for x in list_sc_modele if
                                                                   gp.sc_modele_nmd_tag in x]
                        self.scenarios_files[etab][sc]["MODELE_ECH"] = [x for x in list_sc_modele if
                                                                   gp.sc_modele_ech_tag in x]
                        self.scenarios_files[etab][sc]["MODELE_DAV"] = [x for x in list_sc_modele if
                                                                   gp.sc_modele_dav_tag in x]

                        self.scenarios_files[etab][sc]["TAUX"] = {}
                        self.scenarios_files[etab][sc]["TAUX"][gp.sc_tx_tag] = [x for x in list_sc_taux if gp.sc_tx_tag in x]
                        self.scenarios_files[etab][sc]["TAUX"][gp.sc_zc_tag] = [x for x in list_sc_taux if gp.sc_zc_tag in x]
                        self.scenarios_files[etab][sc]["TAUX"][gp.sc_tci_tag] = [x for x in list_sc_taux if
                                                                            gp.sc_tci_tag in x]
                        self.scenarios_files[etab][sc]["TAUX"][gp.sc_liq_tag] = [x for x in list_sc_taux if
                                                                            gp.sc_liq_tag in x]
                        self.scenarios_files[etab][sc]["TAUX"][gp.sc_rco_ref_tag] = [x for x in list_sc_taux if
                                                                                gp.sc_rco_ref_tag in x]

                        self.scenarios_files[etab][sc]["VOLUME"] = {}
                        self.scenarios_files[etab][sc]["VOLUME"][gp.sc_vol_nmd_tag] = [x for x in list_sc_volume if
                                                                                  gp.sc_vol_nmd_tag in x
                                                                                  and not gp.sc_vol_nmd_prct_tag in x and not gp.sc_vol_nmd_calage_tag in x]
                        self.scenarios_files[etab][sc]["VOLUME"][gp.sc_vol_ech_tag] = [x for x in list_sc_volume if
                                                                                  gp.sc_vol_ech_tag in x and not gp.sc_vol_pn_ech_prct_tag in x]
                        self.scenarios_files[etab][sc]["VOLUME"][gp.sc_vol_nmd_prct_tag] = [x for x in list_sc_volume if
                                                                                       gp.sc_vol_nmd_prct_tag in x]
                        self.scenarios_files[etab][sc]["VOLUME"][gp.sc_vol_nmd_calage_tag] = [x for x in list_sc_volume if
                                                                                         gp.sc_vol_nmd_calage_tag in x]
                        self.scenarios_files[etab][sc]["VOLUME"][gp.sc_vol_pn_ech_prct_tag] = [x for x in list_sc_volume if
                                                                                          gp.sc_vol_pn_ech_prct_tag in x]

                        if len(self.scenarios_files[etab][sc]["TAUX"]) != 5:
                            msg = "Un des fichier scénario de taux est manquant, le scénario %s ne sera pas traité" % sc
                            logger.error(msg)
                            self.list_scenarios[etab].remove(sc)
                        if len(self.scenarios_files[etab][sc]["VOLUME"]) != 5:
                            msg = "Un dex fichier scénario de volume scénario est manquant, le scénario %s ne sera pas traité" % sc
                            logger.error(msg)
                            self.list_scenarios[etab].remove(sc)
                        if (len(self.scenarios_files[etab][sc]["MODELE_ECH"]) == 0):
                            msg = "Le fichier de modèle ech scénario  est manquant, le scénario %s ne sera pas traité" % sc
                            logger.error(msg)
                            self.list_scenarios[etab].remove(sc)
                        if (len(self.scenarios_files[etab][sc]["MODELE_NMD"]) == 0):
                            msg = "Le fichier de modèle nmd scénario  est manquant, le scénario %s ne sera pas traité" % sc
                            logger.error(msg)
                            self.list_scenarios[etab].remove(sc)
                        if (len(self.scenarios_files[etab][sc]["MODELE_DAV"]) == 0
                                and len(self.scenarios_params[etab][sc][
                                            self.scenarios_params[etab][sc]["TYPE PRODUIT"].str.contains(
                                                "DAV")].copy()) > 0):
                            msg = "Le fichier de modèle dav scénario  est manquant, le scénario %s ne sera pas traité" % sc
                            logger.error(msg)
                            self.list_scenarios[etab].remove(sc)

                        for model in ["MODELE_NMD", "MODELE_ECH", "MODELE_DAV"]:
                            if self.scenarios_files[etab][sc][model] != []:
                                self.scenarios_files[etab][sc][model] = os.path.join(self.input_path, etab, sc, "MODELES",
                                                                                self.scenarios_files[etab][sc][model][0])

                if not gf.begin_in_list2(stock_file, gp.stock_tag):
                    logger.warning(
                        "Le fichier STOCK lié à l'entité " + etab + " n'existe pas dans le dossier source indiqué. L'entité sera exclue")
                    self.list_etab_usr.remove(etab)

    def display_user_params(self, ):
        # global date_refix_tla
        """ RAPPEL DES PARAMETRES UTILISATEURS"""
        logger.info(" PARAMETRES UTILISATEUR FONCTIONNELS:")
        logger.info("  - DATE D'ARRÊTE: %s" % str(datetime.datetime.strftime(self.dar, "%d/%m/%Y")))
        logger.info("  - NB MOIS PROJECTION: %s" % str(self.nb_mois_proj_out))
        logger.info("  - NB MOIS PROJECTION IMPLICITE (OUTLFOW): %s", str(self.nb_mois_proj_usr))
        logger.info("  - MAX MOIS PN: %s" % self.max_month_pn)
        logger.info("  - SPREADS AJUSTEMENTS: %s", str(self.spread_liq.set_index([gp.nc_devise_spread]).to_dict('index')))

        logger.info("  PARAMETRES UTILISATEUR d'EXECUTION:")
        logger.info("  - REPERTOIRE DE SORTIE: %s" % self.output_path_usr)
        logger.info("  - AJUSTEMENTS SLT: %s" % self.ajust_only)
        logger.info("  - APPLIQUER LES CONVS D'ECOULEMENT GAP LIQ: %s" % self.force_gp_liq)
        logger.info("  - APPLIQUER LES CONVS D'ECOULEMENT GAPS NMD: %s" % self.force_gps_nmd)

        logger.info("  PARAMETRES UTILISATEUR DE SORTIE:")
        logger.info("  - IND SORTIE STOCK: %s" % str(self.indic_sortie["ST"] + self.indic_sortie_eve["ST"]))
        logger.info("  - IND SORTIE PN: %s" % str(
            self.indic_sortie["PN"] + self.indic_sortie_eve["PN"] + list(self.indic_sortie["PN_CONTRIB"].keys())))
        if self.ajust_only:
            logger.info(
                "  - IND SORTIE AJUST: %s" % str(self.indic_sortie["AJUST"] + +self.indic_sortie_eve["AJUST"] + list(
                    self.indic_sortie["AJUST_CONTRIB"].keys())))
        logger.info("  - COLONNES DE SORTIE A EXCLURE: %s" % str(self.cols_sortie_excl))
        if self.retraitement_tla:
            logger.info("    UN RETRAITEMENT TLA SERA EFFECTUE A PARTIR DE %s" % self.date_refix_tla.date())

    def parse_output_indicators(self):
        self.type_simul = {}
        if "OUI" in self.data_indic[gp.nc_ind_restituer].values.tolist():
            self.type_simul["LIQ"] = True
        else:
            self.type_simul["LIQ"] = False

        if "OUI" in self.data_indic_eve[gp.nc_ind_restituer].values.tolist() and self.type_eve != "ICAAP":
            self.type_simul["EVE"] = True
        else:
            self.type_simul["EVE"] = False

        if "OUI" in self.data_indic_eve[gp.nc_ind_restituer].values.tolist() and self.type_eve == "ICAAP":
            self.type_simul["EVE_LIQ"] = True
        else:
            self.type_simul["EVE_LIQ"] = False

        data_indic = pd.concat([self.data_indic, self.data_indic_eve])
        data_indic = data_indic[data_indic[gp.nc_ind_restituer] == "OUI"].copy()

        self.indic_sortie = {}
        self.indic_sortie["ST"] = []
        self.indic_sortie["PN"] = []
        self.indic_sortie["PN_CONTRIB"] = {}
        self.indic_sortie["PN_OUTFLOW"] = {}
        self.indic_sortie["ST_OUTFLOW"] = {}

        self.indic_sortie_eve = {}
        self.indic_sortie_eve["ST"] = []
        self.indic_sortie_eve["PN"] = []

        add_on_project = 0

        for i in range(0, len(data_indic)):
            restit = data_indic[gp.nc_ind_restituer].iloc[i]
            step = 1 if data_indic[gp.nc_ind_pas].iloc[i] == "MENSUEL" else (
                3 if data_indic[gp.nc_ind_pas].iloc[i] == "TRIMESTRIEL" else 12)
            m_deb = int(data_indic[gp.nc_ind_deb].iloc[i]) if data_indic[gp.nc_ind_deb].iloc[i] != "-" else 0
            m_fin = int(data_indic[gp.nc_ind_fin].iloc[i]) if data_indic[gp.nc_ind_fin].iloc[i] not in ["-",
                                                                                                        "inf"] else (
                0 if data_indic[gp.nc_ind_fin].iloc[i] == "-" else "inf")
            cat = data_indic[gp.nc_ind_cat].iloc[i]
            cat = cat if cat == "PN" else "ST"
            indic = data_indic[gp.nc_ind_indic].iloc[i]
            typo = data_indic[gp.nc_ind_type].iloc[i]
            """ SEULS LES INDICATEURS A RESTITUER SONT INCLUS"""
            if restit == "OUI":
                if indic not in [gp.outf_sti, gp.outf_pni]:
                    """ TRAITEMENT DES INDICS de TYPE NON OUTFLOW """
                    if typo != "NORMAL":
                        for mois in range(m_deb, m_fin + 1):
                            if (mois - m_deb) // step == (mois - m_deb) / step:
                                if mois <= self.nb_mois_proj_out and not (mois == 0 and cat == "PN"):
                                    self.indic_sortie[cat].append(indic + str(mois))
                        if typo == "CONTRIB":
                            self.indic_sortie["PN_CONTRIB"][indic] = step
                    else:
                        self.indic_sortie[cat].append(indic)
                else:
                    """ TRAITEMENT DES INDICS de TYPE OUTFLOW """
                    self.indic_sortie[cat].append(
                        indic + " " + str(m_deb) + "M-" + ((str(m_fin) + "M") if m_fin != "inf" else "inf"))
                    self.indic_sortie[cat + "_OUTFLOW"][
                        indic + " " + str(m_deb) + "M-" + ((str(m_fin) + "M") if m_fin != "inf" else "inf")] = (
                        m_deb, m_fin)

                    """ ON CALCULE LE NB DE MOIS SUPP DE PROJ POUR POUR PVR CALCULER LES OUTFLOW A N MOIS """
                    if not self.ajust_only:
                        add_on_project = max(add_on_project, m_deb, m_fin if m_fin != "inf" else 0)
                    else:
                        if cat == "PN":
                            add_on_project = max(add_on_project, m_deb, m_fin if m_fin != "inf" else 0)

            """ AJOUT DE CERTAINS OUTFLOWS SI PRESENCE de L'INDIC NSFR"""
        for cat in ["ST", "PN"]:
            delta_rsf_indic = gp.delta_rsf_pni if cat == "PN" else gp.delta_rsf_sti
            rsf_indic = gp.rsf_pni if cat == "PN" else gp.rsf_sti
            delta_asf_indic = gp.delta_asf_pni if cat == "PN" else gp.delta_asf_sti
            asf_indic = gp.asf_pni if cat == "PN" else gp.asf_sti
            if delta_rsf_indic in self.indic_sortie[cat] or rsf_indic in self.indic_sortie[cat] \
                    or delta_asf_indic in self.indic_sortie[cat] or asf_indic in self.indic_sortie[cat]:
                for indic in gp.list_indic_nsfr:
                    if not indic in list(self.indic_sortie[cat + "_" + "OUTFLOW"].keys()):
                        m_deb = int(indic.split(" ")[1].split("-")[0].replace("M", ""))
                        m_fin = indic.split(" ")[1].split("-")[1].replace("M", "")
                        if m_fin != "inf":
                            m_fin = int(m_fin)
                        self.indic_sortie[cat + "_" + "OUTFLOW"][indic] = (m_deb, m_fin)
                        if not self.ajust_only:
                            add_on_project = max(add_on_project, m_deb, m_fin if m_fin != "inf" else 0)
                        else:
                            if cat == "PN":
                                add_on_project = max(add_on_project, m_deb, m_fin if m_fin != "inf" else 0)

            """ AJOUT DE CERTAINS OUTFLOWS SI PRESENCE de L'INDIC LCR"""
        for cat in ["ST", "PN"]:
            delta_nco_indic = gp.delta_nco_pni if cat == "PN" else gp.delta_nco_sti
            nco_indic = gp.nco_pni if cat == "PN" else gp.nco_sti
            if delta_nco_indic in self.indic_sortie[cat] or nco_indic in self.indic_sortie[cat]:
                for indic in gp.list_indic_lcr:
                    if not indic in list(self.indic_sortie[cat + "_" + "OUTFLOW"].keys()):
                        m_deb = int(indic.split(" ")[1].split("-")[0].replace("M", ""))
                        m_fin = indic.split(" ")[1].split("-")[1].replace("M", "")
                        if m_fin != "inf":
                            m_fin = int(m_fin)
                        self.indic_sortie[cat + "_" + "OUTFLOW"][indic] = (m_deb, m_fin)
                        if not self.ajust_only:
                            add_on_project = max(add_on_project, m_deb, m_fin if m_fin != "inf" else 0)
                        else:
                            if cat == "PN":
                                add_on_project = max(add_on_project, m_deb, m_fin if m_fin != "inf" else 0)

        self.indic_sortie["AJUST"] = self.indic_sortie["PN"].copy()
        self.indic_sortie["AJUST_CONTRIB"] = self.indic_sortie["PN_CONTRIB"].copy()
        self.indic_sortie["AJUST_OUTFLOW"] = self.indic_sortie["PN_OUTFLOW"].copy()

        if self.ajust_only:
            self.indic_sortie["ST"] = []
            self.indic_sortie["PN"] = []
            self.indic_sortie["PN_CONTRIB"] = {}
            self.indic_sortie["PN_OUTFLOW"] = {}
            self.indic_sortie["ST_OUTFLOW"] = {}

        """ ON AJOUTE LE NB DE MOIS SUPP DE PROJ POUR POUR PVR CALCULER LES OUTFLOW A N MOIS """
        self.nb_mois_proj_usr = min(gp.max_months, add_on_project + self.nb_mois_proj_out)

        self.indic_sortie_eve["ST"] = [x for x in self.indic_sortie["ST"] if
                                  gf.begin_with_list(self.data_indic_eve["INDIC"].values.tolist(), x)]
        self.indic_sortie_eve["PN"] = [x for x in self.indic_sortie["PN"] if
                                  gf.begin_with_list(self.data_indic_eve["INDIC"].values.tolist(), x)]
        self.indic_sortie_eve["AJUST"] = [x for x in self.indic_sortie["AJUST"] if
                                     gf.begin_with_list(self.data_indic_eve["INDIC"].values.tolist(), x)]

        self.indic_sortie["ST"] = [x for x in self.indic_sortie["ST"] if not x in self.indic_sortie_eve["ST"]]
        self.indic_sortie["PN"] = [x for x in self.indic_sortie["PN"] if not x in self.indic_sortie_eve["PN"]]
        self.indic_sortie["AJUST"] = [x for x in self.indic_sortie["AJUST"] if not x in self.indic_sortie_eve["AJUST"]]

        di.generate_dependencies_indic(self)

