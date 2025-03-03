import dateutil
from params.config_files.config_excel_manager import ExcelConfigManager
from params.sources_params import SourcesParams
from params.output_params import OutputManager
import os
import logging
import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
class SimulParams():
    def __init__(self, args):
        self.exec_mod = "DEBUG"
        self.mode = args.mode
        if not args.use_json:
            self.config_cls = ExcelConfigManager(args)
        else:
            self.config_cls = None
        self.sources_cls = SourcesParams()
        self.output_cls = OutputManager()
        self.set_dar()
        self.set_output_folder()
        self.set_nb_mois_proj()
        self.set_list_entities()
        self.set_sources_folder()
        self.set_scenario_folder()
        self.set_alim_folder()
        self.set_name_run()
        self.set_batch_size(args)

    def set_batch_size(self, args):
        self.batch_size_ech = int(args.batch_size_ech)
        self.batch_size_nmd = int(args.batch_size_nmd)

    def set_name_run(self):
        self.name_run = self.config_cls.get_value_from_named_ranged(self.config_cls.names_ex.NAME_RUN)

    def set_dar(self):
        self.dar = self.config_cls.get_value_from_named_ranged(self.config_cls.names_ex.NAME_RANGE_DAR)
        self.dar = dateutil.parser.parse(str(self.dar)).replace(tzinfo=None)

    def set_nb_mois_proj(self):
        self.nb_months_proj = (self.config_cls.get_value_from_named_ranged
                               (self.config_cls.names_ex.NAME_RANGE_NB_MONTHS_PROJ))

    def set_sources_folder(self):
        sources_folder = (self.config_cls.get_value_from_named_ranged
                                           (self.config_cls.names_ex.NAME_RANGE_SOURCES_FOLDER))
        self.sources_cls.set_sources_folder(sources_folder)

        self.control_path_existence(sources_folder)

    def set_output_folder(self):
        self.output_cls.output_folder = (self.config_cls.get_value_from_named_ranged
                                         (self.config_cls.names_ex.NAME_RANGE_OUTPUT_FOLDER))
        self.control_path_existence(self.output_cls.output_folder)
        if self.mode == "SCENARIO+MOTEUR":
            now = datetime.datetime.now().strftime("%Y%m%d.%H%M.%S")
            new_dir = str(self.dar.year) + str(self.dar.month) + str(self.dar.day) + "_EXEC-" + str(now)
            self.output_cls.output_folder\
                = os.path.join(self.output_cls.output_folder, "SCENARIO+MOTEUR_DAR-%s" % (new_dir))
            Path(self.output_cls.output_folder).mkdir(parents=True, exist_ok=True)



    def set_list_entities(self):
        self.entities_list = str(self.config_cls.get_value_from_named_ranged
                                 (self.config_cls.names_ex.NAME_RANGE_ENTITIES)).replace(" ", "").split(",")

    def set_scenario_folder(self):
        if self.mode == "MOTEUR":
            self.sources_cls.scenario_folder\
                = self.config_cls.get_value_from_named_ranged(self.config_cls.names_ex.NAME_RANGE_SCENARIO_FOLDER)
            if str(self.sources_cls.scenario_folder) == "" or self.sources_cls.scenario_folder is None:
                raise ValueError("Un répertoire de scénario n'a pas été précisé")

            self.control_path_existence(self.sources_cls.scenario_folder)

    def set_alim_folder(self):
        if "SCENARIO" in self.mode:
            self.sources_cls.alim_folder\
                = self.config_cls.get_value_from_named_ranged(self.config_cls.names_ex.NAME_RANGE_ALIM_FOLDER)
            if str(self.sources_cls.alim_folder) == "" or self.sources_cls.alim_folder is None:
                raise ValueError("Un répertoire d'alim n'a pas été précisé")

            self.control_path_existence(self.sources_cls.alim_folder)

    def control_path_existence(self, path_dir):
        if not os.path.exists(path_dir):
            logger.error("Le chemin suivant n'existe pas : " + path_dir + "  Veuillez vérifier vos paramètres")
            raise ImportError("Le chemin suivant n'existe pas : " + path_dir + " Veuillez vérifier vos paramètres")

