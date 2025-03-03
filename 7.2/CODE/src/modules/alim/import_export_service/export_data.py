import ntpath
import modules.alim.parameters.user_parameters as up
import modules.alim.lcr_nsfr_service.lcr_nsfr_module as lcr_nsfr
import utils.excel_openpyxl as ex
import mappings.mapping_functions as mp
from shutil import copyfile
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class Exporter():
    def __init__(self, cls_usr):
        self.up = cls_usr

    
    def export_missing_mappings(self,):
        logger.info("   EXPORTATION DES ERREURS DE MAPPINGS")
        copyfile(self.up.sm.path_map_mis_template_file, self.up.missing_map_output_path)
        full_err = {}
        title_maps = {}
        full_err["NEC"] = [];
        full_err["FAC"] = [];
        title_maps["NEC"] = [];
        title_maps["FAC"] = []
        i = 0
        if len(mp.missing_mapping) > 0:
            for name, val in mp.missing_mapping.items():
                map = val[0].drop_duplicates()
                typo = "NEC" if not val[1] else "FAC"
                title_maps[typo] = title_maps[typo] + [name] + [""] * len(map.columns.tolist())
                if len(full_err[typo]) == 0:
                    full_err[typo] = map
                else:
                    map.insert(0, "", "", allow_duplicates=True)
                    cols = full_err[typo].columns.tolist() + map.columns.tolist()
                    full_err[typo] = pd.concat([full_err[typo].reset_index(drop=True), map.reset_index(drop=True)], axis=1,
                                               ignore_index=True).fillna("")
                    full_err[typo].columns = cols
                i = i + 1
            for typo in list(title_maps.keys()):
                if len(full_err[typo]) > 0:
                    feuille = "MAPPINGS FACULTATIFS MANQUANTS" if typo == "FAC" else "MAPPINGS MANQUANTS"
                    title_maps[typo] = pd.DataFrame(title_maps[typo]).transpose()
                    ex.write_to_excel(title_maps[typo], self.up.missing_map_output_path, feuille, (1, 1), header=False)
                    ex.write_to_excel(full_err[typo], self.up.missing_map_output_path, feuille, (2 + 1 , 1))
    
            msg = "     Attention des MAPPINGS sont manquants. Retrouvez-les" + \
                  " dans le fichier de sortie MAPPINGS_MANQUANTS.xlsb"
            logging.warning(msg)
    
            mp.missing_mapping = {}
    
    
    def export_scenario_file_data(self, PN_ECH, PN_NMD, PN_ECH_pct, PN_NMD_pct, NMD_CALAGE):
        logger.info("   EXPORTATION DES DONNEES de PNs")
        PN_ECH.to_csv(self.up.sc_vol_ech_output_path, decimal=",", sep=";", index=False)
        PN_NMD.to_csv(self.up.sc_vol_nmd_output_path, decimal=",", sep=";", index=False)
        PN_ECH_pct.to_csv(self.up.sc_vol_pn_ech_prct_output_path, decimal=",", sep=";", index=False)
        PN_NMD_pct.to_csv(self.up.sc_vol_nmd_prct_output_path, decimal=",", sep=";", index=False)
        NMD_CALAGE.to_csv(self.up.sc_vol_nmd_calage_output_path, decimal=",", sep=";", index=False)
    
    
    def export_stock_file_data(self, STOCK):
        logger.info("   EXPORTATION DES DONNEES DU STOCK")
        STOCK.to_csv(self.up.stock_output_path, decimal=",", sep=";", index=False)
    
    
    def rename_and_select_cols(self, data, etab, nb_col_qual, col_name=[], etab_col=True):
        if etab_col:
            list_cols = []
            i = 1
            for col in data.columns:
                if etab.upper() == col.upper():
                    list_cols.append(col + str(i))
                    i = i + 1
                else:
                    list_cols.append(col)
    
            data.columns = list_cols
            data = data[list_cols[:nb_col_qual] + [x for x in list_cols[nb_col_qual:] if
                                                   etab.upper() == x.upper()[:-1] and x.upper()[-1:].isnumeric()]]
            if col_name != []:
                data.columns = data.columns.tolist()[:-len(col_name)] + col_name
        else:
            data = data[data["BASSIN"] == self.up.current_etab].copy()
    
        return data.copy()
    
    
    def export_data_generic(self, wb, range_name, final_wb_path, nb_col_qual, file_path, close=True, col_name=[], etab_col=True):
        data_to_export = ex.get_dataframe_from_range(wb, range_name)
        data_to_export = self.rename_and_select_cols(data_to_export, self.up.current_etab, nb_col_qual, col_name=col_name,
                                                etab_col=etab_col)
        if data_to_export.shape[1] <= nb_col_qual:
            logger.error(
                "L'établissement %s n'existe pas dans le fichier %s" % (self.up.current_etab, file_path))
        else:
            ex.write_to_excel(data_to_export, final_wb_path, named_range_name=range_name, header=True)
            if close:
                wb.Close(False)
    
    
    def export_lcr_nsfr_data(self, ):
        if self.up.update_param_lcr_ncr and self.up.map_lcr_nsfr:
            copyfile(self.up.path_sc_lcr_nsfr_template_file, self.up.sc_lcr_nsfr_output_path)
            copyfile(self.up.path_lcr_nsfr_template_st, self.up.lcr_nsfr_st_output_path)
            logger.info("   EXPORTATION DES DONNEES DE LCR ET NSFR")
    
            ex.write_to_excel(lcr_nsfr.table_nco_rl, self.up.sc_lcr_nsfr_output_path, named_range_name="_PARAM_LCR", header=True)
            ex.write_to_excel(lcr_nsfr.table_nco_rl, self.up.lcr_nsfr_st_output_path, named_range_name="_PARAM_LCR", header=True)
    
            logger.info("      Lecture de : %s" % ntpath.basename(self.up.lcr_nsfr_files_name["MODE_CALC"]))
            outflow_wb = ex.load_workbook_openpyxl(self.up.lcr_nsfr_files_name["MODE_CALC"], read_only=True)
            for indic, close in zip(["NSFR", "LCR"], [False, True]):
                range_name = "_OUTFLOW_" + indic
                self.export_data_generic(outflow_wb, range_name, self.up.sc_lcr_nsfr_output_path, 1, self.up.lcr_nsfr_files_name["MODE_CALC"],
                                    close=False, col_name=["APPLIQUER OUTFLOW"])
                self.export_data_generic(outflow_wb, range_name, self.up.lcr_nsfr_st_output_path, 1, self.up.lcr_nsfr_files_name["MODE_CALC"],
                                    close=close, col_name=["APPLIQUER OUTFLOW"])
    
            logger.info("      Lecture de : %s" % ntpath.basename(self.up.lcr_nsfr_files_name["NSFR_COEFF_HISTO"]))
            nsfr_coeff_wb = ex.load_workbook_openpyxl(self.up.lcr_nsfr_files_name["NSFR_COEFF_HISTO"], read_only=True)
            range_name = "_PARAM_NSFR"
            cols = ["Coefficient", "Coeff Outflow 0-6 mois", "Coeff Outflow 6-12 mois", "Coeff Outflow 12-inf mois"]
            self.export_data_generic(nsfr_coeff_wb, range_name, self.up.sc_lcr_nsfr_output_path, 2, self.up.lcr_nsfr_files_name["NSFR_COEFF_HISTO"],
                                close=False, col_name=cols)
            self.export_data_generic(nsfr_coeff_wb, range_name, self.up.lcr_nsfr_st_output_path, 2, self.up.lcr_nsfr_files_name["NSFR_COEFF_HISTO"],
                                col_name=cols)
    
            logger.info("      Lecture de : %s" % ntpath.basename(self.up.lcr_nsfr_files_name["NSFR_DESEMC"]))
            nsfr_coeff_wb = ex.load_workbook_openpyxl(self.up.lcr_nsfr_files_name["NSFR_DESEMC"], read_only=True)
            range_name = "_NSFR_DESEMC"
            self.export_data_generic(nsfr_coeff_wb, range_name, self.up.sc_lcr_nsfr_output_path, 2, self.up.lcr_nsfr_files_name["NSFR_DESEMC"],
                                etab_col=False)
    
            logger.info("      Lecture de : %s" % ntpath.basename(self.up.lcr_nsfr_files_name["PARAM_LCR_SPEC"]))
            lcr_spec_wb = ex.load_workbook_openpyxl(self.up.lcr_nsfr_files_name["PARAM_LCR_SPEC"], read_only=True)
            range_name = "_PARAM_LCR_SPEC"
            cols = ["NCO", "RL"]
            self.export_data_generic(lcr_spec_wb, range_name, self.up.sc_lcr_nsfr_output_path, 3, self.up.lcr_nsfr_files_name["PARAM_LCR_SPEC"],
                                close=False, col_name=cols)
            self.export_data_generic(lcr_spec_wb, range_name, self.up.lcr_nsfr_st_output_path, 3, self.up.lcr_nsfr_files_name["PARAM_LCR_SPEC"],
                                col_name=cols)
    
            for indic, nb_cols_qual, col in zip(["NSFR", "LCR"], [2, 1], [["ASF/RSF officiel"], ["VAL"]]):
                logger.info("      Lecture de : %s" % ntpath.basename(self.up.lcr_nsfr_files_name[indic + "_DAR"]))
                indic_wb = ex.load_workbook_openpyxl(self.up.lcr_nsfr_files_name[indic + "_DAR"], read_only=True)
                range_name = "_" + indic + "_DAR"
                self.export_data_generic(indic_wb, range_name, self.up.lcr_nsfr_st_output_path, nb_cols_qual, self.up.lcr_nsfr_files_name[indic + "_DAR"],
                                    col_name=col)
                ex.close_workbook(indic_wb)
    
            ex.close_workbook(outflow_wb)
            ex.close_workbook(lcr_spec_wb)
            ex.close_workbook(nsfr_coeff_wb)
    
    def export_st_nmd_template_file(self, NMD_TEMPLATE_PROJ):
        if len(NMD_TEMPLATE_PROJ) == 0:
            NMD_TEMPLATE_PROJ = pd.DataFrame(["*"], columns=["RCO_ALLOCATION_KEY"])
        NMD_TEMPLATE_PROJ.to_csv(self.up.nmd_template_output_path, index=False, sep=";", decimal=",")
    
    
    def export_data(self, STOCK, NMD_TEMPLATE_PROJ, NMD_CAlAGE, PN_ECH=[], PN_NMD=[], PN_ECH_pct=[],
                    PN_NMD_pct=[]):
        self.export_lcr_nsfr_data()
        self.export_stock_file_data(STOCK)
        self.export_scenario_file_data(PN_ECH, PN_NMD, PN_ECH_pct, PN_NMD_pct, NMD_CAlAGE)
        self.export_st_nmd_template_file(NMD_TEMPLATE_PROJ)
        self.export_missing_mappings()
