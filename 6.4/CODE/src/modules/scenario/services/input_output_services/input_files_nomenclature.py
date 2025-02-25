import os
from utils import excel_utils as excel_io
from modules.scenario.parameters import user_parameters as up

class NomenclatureSaver():
    def __init__(self):
        self._source_nomenclature = None
        self._contrats_nomenclature = None
        self._contrats_pn_nomenclature = None
        self._dar = None
        self._source_dir_path = None
        self._etab = None

    def get_source_files_path(self, wb_mapping, etab, indicator_name, dar):
        if self._source_nomenclature is None or self._etab != etab:
            self._etab = etab
            self._source_dir_path = up.source_dir
            self._dar = dar
            excel_io.set_value_to_named_cell(wb_mapping, '_DAR_ALIM_STOCK_', self._dar.strftime("%Y-%m-%d"), protected=True)
            wb_mapping.Names('_DAR_ALIM_STOCK_').RefersToRange.Worksheet.Calculate()
            self._source_nomenclature = excel_io.get_dataframe_from_range(wb_mapping, "_NOMENCLATURE_SOURCES", header=True)
            if ~(self._source_nomenclature['ENTITE'] == etab).any():
                self._source_nomenclature = self._source_nomenclature.replace('RZO', etab, regex=True)
            self._source_nomenclature = self._source_nomenclature.loc[self._source_nomenclature['ENTITE'] == etab, :]
        try:
            sources_file = self._source_nomenclature[(self._source_nomenclature["NOM INDICATEUR"] == indicator_name)]["CHEMIN"]
            sources_file = sources_file.str.replace("\[ENTITE]", etab, regex=True)
            return os.path.join(self._source_dir_path, sources_file.iloc[0])
        except:
            raise AttributeError('Erreur dans le fichier mapping, l\'indicateur {} n\'est pas disponible  pour {}'.format(indicator_name, etab))