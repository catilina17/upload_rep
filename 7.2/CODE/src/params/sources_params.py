import os
import numpy as np
from params import version_params as vp
import utils.general_utils as gu
from mappings import general_mappings as gmp


class SourcesParams():
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_instance()
        return cls._instance

    def _init_instance(self):
        self.set_templates_relative_paths()
        self.set_sources_relative_paths()

    def set_templates_relative_paths(self):

        self.sc_template_folder_tag = "TEMPLATES"
        self.sc_template_alim_folder_tag = "ALIM"
        self.sc_template_mapping_folder_tag = "MAPPING"
        self.sc_template_stock_sc_tag = "STOCK_SCENARIO"

        self.path_map_mis_template_file = os.path.join(self.sc_template_folder_tag,
                                                       self.sc_template_alim_folder_tag,
                                                       self.sc_template_mapping_folder_tag, "MAPPINGS_MANQUANTS.xlsx")

        self.path_lcr_nsfr_template_file = os.path.join(self.sc_template_folder_tag,
                                                        self.sc_template_alim_folder_tag, self.sc_template_stock_sc_tag,
                                                        "ST_LCR_NSFR_TEMPLATE.xlsx")

        self.path_sc_lcr_nsfr_template_file = os.path.join(self.sc_template_folder_tag,
                                                           self.sc_template_alim_folder_tag,
                                                           self.sc_template_stock_sc_tag,
                                                           "SC_LCR_NSFR_TEMPLATE.xlsx")

    def set_sources_relative_paths(self):
        self.mapp_rel_path = os.path.join("MAPPING", "MAPPING_PASS_ALM.xlsx")
        self.rate_input_folder_name = "RATE_INPUT"

    def set_sources_folder(self, sources_path):
        self.sources_folder = sources_path
        if not os.path.exists(self.sources_folder):
            raise ValueError("Le dossier des sources n'existe pas : " + str(self.sources_folder))

        self.mapp_file_path = os.path.join(self.sources_folder, self.mapp_rel_path)
        if not os.path.exists(self.mapp_file_path):
            raise ValueError("Le fichier de MAPPING est absent du dossier des SOURCES: " + str(self.mapp_file_path))

    def set_templates_paths(self, base_path):
        self.path_map_mis_template_file = base_path / self.path_map_mis_template_file
        self.path_lcr_nsfr_template_st = base_path / self.path_lcr_nsfr_template_file
        self.path_sc_lcr_nsfr_template_file = base_path / self.path_sc_lcr_nsfr_template_file

        gu.check_version_templates(path=self.path_map_mis_template_file, open=True,
                                   version=vp.version_mapping_manquants)
        gu.check_version_templates(path=self.path_lcr_nsfr_template_st, open=True, version=vp.version_STOCK_TMP)

    def get_contract_sources_paths(self, etab, product):
        sources_file = gmp.nomenclature_contrats[gmp.nomenclature_contrats["TYPE CONTRAT"] == product].copy()
        sources_file_etab = sources_file[sources_file["ENTITE"] == etab].copy()
        if len(sources_file_etab) == 0:
            sources_file_etab = sources_file[sources_file["ENTITE"] == "RZO"].copy()
        sources_file_etab["CHEMIN"] = sources_file_etab["CHEMIN"].str.replace("RZO", etab)
        sources_file_etab = sources_file_etab.set_index("TYPE FICHIER")
        sources_file_etab["DELIMITER"] = np.where(sources_file_etab["EXTENSION"] == ".csv", ";", "\t")
        sources_file_etab["DECIMAL"] = np.where(sources_file_etab["EXTENSION"] == ".csv", ",", ".")
        sources_params = sources_file_etab[["CHEMIN", "DELIMITER", "DECIMAL"]].to_dict('index')
        return sources_params
