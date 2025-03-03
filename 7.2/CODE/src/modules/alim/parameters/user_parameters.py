import datetime
import pandas as pd
import mappings.general_mappings as mp
import modules.alim.parameters.general_parameters as gp
import modules.alim.parameters.NTX_SEF_params as ntx_sef_p
from params import version_params as vp
import logging
from pathlib import Path
import os
import utils.general_utils as gu
import modules.alim.parameters.RZO_params as rzo_p

logger = logging.getLogger(__name__)


class UsersParameters():
    def __init__(self, sp_cls):
        self.simul_cls = sp_cls
        self.om = sp_cls.output_cls
        self.sm = sp_cls.sources_cls
        self.dar = sp_cls.dar
        self.etabs = sp_cls.entities_list
        self.sources_folder = self.sm.sources_folder
        self.output_folder = self.om.output_folder
        self.config_cls = sp_cls.config_cls
        self.names = self.config_cls.names_ex

    def get_user_main_params(self):
        self.extract_alim_params_from_config_file()
        self.sm.set_templates_paths(self.config_cls.config_base_path)

    def extract_alim_params_from_config_file(self):
        self.sc_ref_nmd = self.config_cls.get_value_from_named_ranged(self.names.NAME_RANGE_SC_REF)

        self.map_lcr_nsfr_g = True if self.config_cls.get_value_from_named_ranged(
            self.names.NAME_RANGE_MAPPER_RAY) == "OUI" else False

        self.rate_file_path = os.path.join(self.sources_folder, "RATE_INPUT",
                                           self.config_cls.get_value_from_named_ranged(
                                               self.names.NAME_RANGE_RATE_INPUT))
        self.liq_file_path = os.path.join(self.sources_folder, "RATE_INPUT",
                                          self.config_cls.get_value_from_named_ranged(self.names.NAME_RANGE_LIQ_INPUT))

        self.tci_file_path = os.path.join(self.sources_folder, "RATE_INPUT",
                                          self.config_cls.get_value_from_named_ranged(self.names.NAME_RANGE_TCI_INPUT))

        self.generate_fwd_sc = str(self.config_cls.get_value_from_named_ranged(self.names.NAME_RANGE_GENERATE_FWD_NAME)).upper() == "OUI"

        self.fwd_sc_name = str(self.config_cls.get_value_from_named_ranged(self.names.NAME_RANGE_FWD_SC_NAME))

        if len([x for x in self.etabs if x not in gp.NTX_FORMAT]) > 0:
            self.modele_nmd_file_path = os.path.join(self.sources_folder, "MODELES",
                                                     self.config_cls.get_value_from_named_ranged(
                                                         self.names.NAME_RANGE_MODELE_NMD))

            gu.check_version_templates(version=vp.version_modele_nmd, path=self.modele_nmd_file_path, open=True)

            if self.generate_fwd_sc:
                self.modele_ech_file_path = os.path.join(self.sources_folder, "MODELES",
                                                         self.config_cls.get_value_from_named_ranged(
                                                             self.names.NAME_RANGE_MODELE_ECH))

                gu.check_version_templates(version=vp.version_modele_ech, path=self.modele_ech_file_path, open=True)

        self.detail_ig = True if self.config_cls.get_value_from_named_ranged(
            self.names.NAME_RANGE_DET_IG) == "OUI" else False

        ntx_sef_p.perim_ntx = self.config_cls.get_value_from_named_ranged(self.names.NAME_RANGE_PERIM_NTX)

        ntx_sef_p.depart_decale = self.config_cls.get_value_from_named_ranged(self.names.NAME_RANGE_DEPART_DECALE)

        ntx_sef_p.del_enc_dar_zero = self.config_cls.get_value_from_named_ranged(self.names.NAME_RANGE_DAR_ZERO_NTX)



    def create_alim_dir(self):
        now = datetime.datetime.now().strftime("%Y%m%d.%H%M.%S")
        self.output_folder \
            = os.path.join(self.om.output_folder,
                           f"ALIM_DAR-{self.dar.year}{self.dar.month:02d}{self.dar.day}_EXEC-{now}")
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

    def get_category(self):
        if self.current_etab in gp.NTX_FORMAT:
            self.category = "NTX_SEF"
        elif self.current_etab in ["ONEY"]:
            self.category = "ONEY_SOCFIM"
        else:
            self.category = "RZO"

    def generate_etab_parameters(self, etab):
        self.current_etab = etab.upper()
        self.get_category()
        self.generate_output_files()

    def generate_output_files(self):
        self.output_folder_etab = os.path.join(self.output_folder, self.current_etab)
        os.makedirs(self.output_folder_etab, exist_ok=True)
        os.makedirs(os.path.join(self.output_folder_etab, self.om.sc_volume_folder_tag), exist_ok=True)
        os.makedirs(os.path.join(self.output_folder_etab, self.om.sc_lcr_nsfr_folder_tag), exist_ok=True)

        name_out = f"{self.current_etab}_{self.dar.year}{self.dar.month:02d}{self.dar.day}_{self.simul_cls.name_run}" \
            if self.simul_cls.name_run else ""
        self.log_path = os.path.join(self.output_folder_etab, f"LOG_{name_out}.txt")

        self.lcr_nsfr_st_output_path = os.path.join(self.output_folder_etab,
                                                    self.om.lcr_nsfr_st_output_file % name_out)
        self.stock_output_path = os.path.join(self.output_folder_etab, self.om.stock_output_file % name_out)
        self.sc_vol_nmd_output_path = os.path.join(self.output_folder_etab, self.om.sc_vol_nmd_output_file % name_out)
        self.sc_vol_ech_output_path = os.path.join(self.output_folder_etab, self.om.sc_vol_ech_output_file % name_out)
        self.sc_vol_nmd_prct_output_path = os.path.join(self.output_folder_etab,
                                                        self.om.sc_vol_nmd_prct_output_file % name_out)
        self.sc_vol_pn_ech_prct_output_path = os.path.join(self.output_folder_etab,
                                                           self.om.sc_vol_pn_ech_prct_output_file % name_out)
        self.sc_vol_nmd_calage_output_path = os.path.join(self.output_folder_etab,
                                                          self.om.sc_vol_nmd_calage_output_file % name_out)
        self.sc_lcr_nsfr_output_path = os.path.join(self.output_folder_etab, self.om.sc_lcr_nsfr_output_file % name_out)
        self.missing_map_output_path = os.path.join(self.output_folder_etab, self.om.missing_map_output_file % name_out)
        self.nmd_template_output_path = os.path.join(self.output_folder_etab,
                                                     self.om.nmd_template_output_file % name_out)

    def get_input_files_name_lcr_nsfr(self):
        nomenclature_lcr_nsfr = mp.nomenclature_lcr_nsfr
        self.update_param_lcr_ncr = True
        self.lcr_nsfr_files_name = {}
        for i in range(0, len(nomenclature_lcr_nsfr)):
            file_path = nomenclature_lcr_nsfr.iloc[i]["CHEMIN"]
            file_path = os.path.join(self.sm.sources_folder, file_path)
            if not os.path.exists(file_path):
                logger.warning(
                    "Le fichier '%s' n'existe pas, les LCR et NSFR DAR et les paramètres d'OUTFLOW ne seront pas mis à jour" % file_path)
                self.update_param_lcr_ncr = False
                return
            else:
                self.lcr_nsfr_files_name[nomenclature_lcr_nsfr.iloc[i]["NOM"]] = file_path
                gu.check_version_templates(path=file_path, version=vp.version_input_lcr_nsfr, open=True)

    def get_input_file_names(self, etab, fwd_output_folder):
        self.prefixes_main_files = []
        self.main_files_name = []
        self.nmd_st_files_name = {}
        self.bcpe_files_name = []
        self.is_ray = False

        if etab in gp.NON_RZO_ETABS:
            nomenclature = mp.nomenclature_stock_ag[mp.nomenclature_stock_ag["ENTITE"] == self.current_etab].copy()
            nomenclature["TYPE CONTRAT"] = ""
        else:
            nomenclature = mp.nomenclature_stock_ag[mp.nomenclature_stock_ag["ENTITE"] == "RZO"].copy()
            nomenclature["CHEMIN"] = nomenclature["CHEMIN"].str.replace("RZO", self.current_etab)

            nomenclature_pn = mp.nomenclature_pn[mp.nomenclature_pn["ENTITE"] == "RZO"].copy()
            nomenclature_pn["CHEMIN"] = nomenclature_pn["CHEMIN"].str.replace("RZO", self.current_etab)

        if self.generate_fwd_sc:
            nomenclature["CHEMIN"] = [os.path.join(fwd_output_folder, os.path.basename(x)) for x in nomenclature["CHEMIN"].values]

        nomenclature = nomenclature[nomenclature["MODULE"] == "ALIM"].copy()

        if not etab in gp.NON_RZO_ETABS:
            nomenclature_pn["TYPE"] = "D"
            nomenclature_pn.rename(columns={"TYPE FICHIER": "NOM INDICATEUR"}, inplace=True)
            nomenclature = pd.concat([nomenclature, nomenclature_pn])

            nomenclature["TYPE CONTRAT"] = nomenclature["TYPE CONTRAT"].fillna("")

        if not etab in gp.NTX_FORMAT + ["ONEY"]:
            nomenclature_nmd = mp.nomenclature_contrats[mp.nomenclature_contrats["TYPE CONTRAT"] == "ST-NMD"].copy()
            nomenclature_nmd = nomenclature_nmd[nomenclature_nmd["ENTITE"] == "RZO"].copy()
            nomenclature_nmd["CHEMIN"] = nomenclature_nmd["CHEMIN"].str.replace("RZO", self.current_etab)
            nomenclature_nmd["TYPE"] = "S"
            nomenclature_nmd.rename(columns={"TYPE FICHIER": "NOM INDICATEUR"}, inplace=True)

            nomenclature = pd.concat([nomenclature, nomenclature_nmd])
            nomenclature["TYPE CONTRAT"] = nomenclature["TYPE CONTRAT"].fillna("")

        nomenclature = nomenclature.sort_values(["ORDRE LECTURE"])
        nomenclature = self.parse_nomenclature_table(nomenclature)

        for i in range(0, nomenclature.shape[0]):
            if nomenclature["TYPE"].iloc[i] == "F":
                self.bcpe_files_name[nomenclature["NOM INDICATEUR"].iloc[i]] = nomenclature["CHEMIN"].iloc[i]
            elif nomenclature["NOM INDICATEUR"].iloc[i] == "RAY":
                self.is_ray = True
                self.lcr_nsfr_file = nomenclature["CHEMIN"].iloc[i]
            elif "PN-" in nomenclature["TYPE CONTRAT"].iloc[i]:
                key = nomenclature["TYPE CONTRAT"].iloc[i] + "-" + nomenclature["NOM INDICATEUR"].iloc[i]
                rzo_p.pn_rzo_files_name[key] = nomenclature["CHEMIN"].iloc[i]
            elif "ST-NMD" in nomenclature["TYPE CONTRAT"].iloc[i]:
                key = nomenclature["TYPE CONTRAT"].iloc[i]
                self.nmd_st_files_name[key] = nomenclature["CHEMIN"].iloc[i]
            else:
                self.prefixes_main_files.append(nomenclature["NOM INDICATEUR"].iloc[i])
                self.main_files_name.append(nomenclature["CHEMIN"].iloc[i])

        if self.is_ray and self.map_lcr_nsfr_g:
            self.map_lcr_nsfr = True
        else:
            self.map_lcr_nsfr = False

    def parse_nomenclature_table(self, nomenclature):
        nomenclature2 = nomenclature.copy()
        already_ignored_dm = False
        already_ignored_fx = False
        already_ignored_dyn_rco = False
        if self.current_etab in gp.NON_RZO_ETABS:
            rzo_p.do_pn = False
        else:
            rzo_p.do_pn = True

        for i in range(0, nomenclature.shape[0]):
            if not os.path.exists(nomenclature["CHEMIN"].iloc[i]):
                if nomenclature["TYPE"].iloc[i] == "D":
                    if not already_ignored_dm:
                        nomenclature2 = nomenclature2[nomenclature2["TYPE"] != "D"]
                        nomenclature2 = nomenclature2[nomenclature2["TYPE"] != "DYN_RCO"]
                        logger.warning("le fichier suivant est manquant: " + nomenclature["CHEMIN"].iloc[i])
                        logger.warning("L'alimentation sera faite en statique")
                        rzo_p.do_pn = False
                        already_ignored_dm = True

                elif "FX-" in nomenclature["NOM INDICATEUR"].iloc[i]:
                    if not already_ignored_fx:
                        nomenclature2 = nomenclature2[nomenclature2["TYPE"] != "F"]
                        logger.warning("Le fichier suivant est manquant : " + str(nomenclature["CHEMIN"].iloc[i]))
                        logger.warning("Les taux de change ne seront pas inclus")
                        already_ignored_fx = True

                elif "DEM" in nomenclature["NOM INDICATEUR"].iloc[i] or "DMN" in nomenclature["NOM INDICATEUR"].iloc[i]:
                    if not already_ignored_dyn_rco:
                        nomenclature2 = nomenclature2[nomenclature2["TYPE"] != "DYN_RCO"]
                        logger.warning("Le fichier suivant est manquant : " + str(nomenclature["CHEMIN"].iloc[i]))
                        logger.warning("Le dynamique de RCO ne sera pas inclus")
                        already_ignored_dyn_rco = True

                elif nomenclature["NOM INDICATEUR"].iloc[i] == "RAY":
                    if self.map_lcr_nsfr_g:
                        logger.error("Le fichier suivant est manquant : " + str(nomenclature["CHEMIN"].iloc[i]))
                        logger.error(
                            "Vous avez activé le indicators RAY et l'établissement sélectionné est mappable par ray")
                        raise ValueError("Le fichier suivant est manquant : " + str(nomenclature["CHEMIN"].iloc[i]))
                    else:
                        nomenclature2 = nomenclature2[nomenclature2["NOM INDICATEUR"] != "RAY"]
                else:
                    logger.error("Le fichier suivant est manquant : " + str(nomenclature["CHEMIN"].iloc[i]))
                    raise ValueError("Le fichier suivant est manquant : " + str(nomenclature["CHEMIN"].iloc[i]))

        return nomenclature2.copy()
