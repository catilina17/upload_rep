import os
import numpy as np
from utils import excel_utils as excel_io

class NomenclatureSaver():
    def __init__(self):
        self._source_nomenclature = None
        self._contrats_nomenclature = None
        self._contrats_pn_nomenclature = None
        self._dar = None
        self._source_dir_path = None
        self._etab = None

    def get_nomenclature_contrats(self, wb_mapping, etab, source_dir, dar):
        if self._contrats_nomenclature is None or self._etab != etab:
            self._etab = etab
            self._source_dir_path = source_dir
            self._dar = dar
            excel_io.make_excel_value(wb_mapping, '_DAR_CONTRATS_', self._dar.strftime("%Y-%m-%d"), protected=True)
            wb_mapping.Names('_DAR_CONTRATS_').RefersToRange.Worksheet.Calculate()
            self._contrats_nomenclature = excel_io.get_dataframe_from_range(wb_mapping, "_NOMENCLATURE_CONTRATS", header=True)
        if ~(self._contrats_nomenclature['ENTITE'] == etab).any():
            self._contrats_nomenclature = self._contrats_nomenclature.replace('RZO', etab, regex=True)

    def get_contracts_files_path(self, etab, indicator_name):
        sources_file = self._contrats_nomenclature[(self._contrats_nomenclature["TYPE CONTRAT"] == indicator_name)][
            ["TYPE FICHIER", "CHEMIN", "EXTENSION"]].copy()
        sources_file = sources_file.set_index("TYPE FICHIER")
        sources_file["CHEMIN"] = sources_file["CHEMIN"].str.replace("\[ENTITE]", etab, regex=True)
        sources_file["CHEMIN"] = [os.path.join(self._source_dir_path, x) for x in sources_file["CHEMIN"]]
        sources_file["DELIMITER"] = np.where(sources_file["EXTENSION"] == ".csv", ";", "\t")
        sources_file["DECIMAL"] = np.where(sources_file["EXTENSION"] == ".csv", ",", ".")
        sources_params = sources_file[["CHEMIN", "DELIMITER", "DECIMAL"]].to_dict('index')
        return sources_params

    def get_nomenclature_pn(self, wb_mapping, etab, source_dir, dar):
        if self._contrats_pn_nomenclature is None or self._etab != etab:
            self._etab = etab
            self._source_dir_path = source_dir
            self._dar = dar
            excel_io.make_excel_value(wb_mapping, '_DAR_CONTRATS_', self._dar.strftime("%Y-%m-%d"), protected=True)
            wb_mapping.Names('_DAR_CONTRATS_').RefersToRange.Worksheet.Calculate()
            self._contrats_pn_nomenclature = excel_io.get_dataframe_from_range(wb_mapping, "_NOMENCLATURE_CONTRATS_PN", header=True)
        if ~(self._contrats_pn_nomenclature['ENTITE'] == etab).any():
            self._contrats_pn_nomenclature = self._contrats_pn_nomenclature.replace('RZO', etab, regex=True)

    def get_pn_files_path(self, etab, indicator_name):
        sources_file = self._contrats_pn_nomenclature[(self._contrats_pn_nomenclature["TYPE CONTRAT"] == indicator_name)][
            ["TYPE FICHIER", "CHEMIN", "EXTENSION"]].copy()
        sources_file = sources_file.set_index("TYPE FICHIER")
        sources_file["CHEMIN"] = sources_file["CHEMIN"].str.replace("\[ENTITE]", etab, regex=True)
        sources_file["CHEMIN"] = [os.path.join(self._source_dir_path, x) for x in sources_file["CHEMIN"]]
        sources_file["DELIMITER"] = np.where(sources_file["EXTENSION"] == ".csv", ";", "\t")
        sources_file["DECIMAL"] = np.where(sources_file["EXTENSION"] == ".csv", ",", ".")
        sources_params = sources_file[["CHEMIN", "DELIMITER", "DECIMAL"]].to_dict('index')
        return sources_params