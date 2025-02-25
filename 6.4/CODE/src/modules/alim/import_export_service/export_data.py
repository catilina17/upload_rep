import ntpath
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import modules.alim.parameters.user_parameters as up
import modules.alim.lcr_nsfr_service.lcr_nsfr_module as lcr_nsfr
import utils.excel_utils as ex
import mappings.mapping_module as mp
from shutil import copyfile
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

"""def export_stock_data(STOCK):
    logger.info("   EXPORTATION DES DONNEES DU STOCK")
    db.insert_df_in_databse(STOCK, "STOCK")

def export_sc_data(dic_pn):
    logger.info("   EXPORTATION DES DONNEES DE PN")
    for name, val in dic_pn.items():
        #STOCK = unpivot_data(STOCK)
        #STOCK = order_by_indic(STOCK)
        if len(val)!=0:
            db.insert_df_in_databse(val, name)"""

def export_missing_mappings():
    logger.info("   EXPORTATION DES ERREURS DE mappings")
    full_err = {}
    title_maps = {}
    full_err["NEC"] = []; full_err["FAC"] = [];
    title_maps["NEC"] = []; title_maps["FAC"] = []
    i = 0
    if len(mp.missing_mapping) > 0:
        map_mis_wb = ex.try_close_open(up.missing_map_output_file)
        for name, val in mp.missing_mapping.items():
            map = val[0].drop_duplicates()
            typo = "NEC" if not val[1] else "FAC"
            title_maps[typo] = title_maps[typo] + [name] + [""] * len(map.columns.tolist())
            if len(full_err[typo])==0:
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
                feuille = "mappings FACULTATIFS MANQUANTS" if typo=="FAC" else "mappings MANQUANTS"
                title_maps[typo] = pd.DataFrame(title_maps[typo]).transpose()
                fill_template(map_mis_wb, title_maps[typo], [], feuille, (2, 1), header=False)
                fill_template(map_mis_wb, full_err[typo], [], feuille, (2 + 2, 1))

        msg = "     Attention des mappings sont manquants. Retrouvez-les" + \
              " dans le fichier de sortie MAPPINGS_MANQUANTS.xlsb"
        logging.warning(msg)

        map_mis_wb.Save()
        map_mis_wb.Close(False)

        mp.missing_mapping = {}


def fill_template(final_xl, data, num_cols, name_sheet, adrs, header=True, named_range=False):
    if named_range:
        rango = final_xl.Names(adrs).RefersToRange
        adrs = (rango.Row, rango.Column)
    ws = final_xl.Sheets(name_sheet)
    data = data.copy()
    str_variables = [x for x in data.columns if x not in num_cols]
    for i in range(0, len(data.columns)):
        if data.columns[i] in str_variables:
            data[data.columns[i]] = data[data.columns[i]].astype(str)
        else:
            if np.issubdtype(data[data.columns[i]].values.dtype, np.number) and pa.NC_PA_TX_CIBLE not in data.columns[i]:
                data[data.columns[i]] = data[data.columns[i]].astype(float).values

    data = data.replace(np.nan, '', regex=True)
    data.columns = ["" if "Unnamed" in str(x) else x for x in data.columns]

    try:
        ws.Unprotect(Password="LAX_MILGRAM")
    except:
        pass

    sizo = data.shape[0]
    chunk_size = 20000
    data_list = [data[i:i + chunk_size].copy() for i in range(0, sizo, chunk_size)]
    rowo = adrs[0]
    for k in range(0, len(data_list)):
        data_part = data_list[k]
        if k == 0 and header:
            header_p = True
        else:
            header_p = False

        if header_p:
            vals = (tuple(["'" + str(x) for x in data_part.columns]),) + tuple(
                data_part.itertuples(index=False, name=None))
        else:
            vals = tuple(data_part.itertuples(index=False, name=None))

        ws.AutoFilterMode = False

        if header_p:
            ws.Range(ws.Cells(rowo, adrs[1]), ws.Cells(rowo + len(data_part), len(data_part.columns))).Value = vals
        else:
            ws.Range(ws.Cells(rowo, adrs[1]), ws.Cells(rowo + len(data_part) - 1, len(data_part.columns))).Value = vals

        ex.empty_clipboard()
        if header_p:
            rowo = rowo + len(data_part) + 1
        else:
            rowo = rowo + len(data_part)

    try:
        ws.Protect(Password="LAX_MILGRAM")
    except:
        pass


def export_scenario_file_data(PN_ECH, PN_NMD,PN_ECH_pct, PN_NMD_pct, NMD_CAlAGE):
    logger.info("   EXPORTATION DES DONNEES de PNs")

    sc_final_xl = ex.try_close_open(up.sc_volume_output_path)

    for data, name in zip([PN_ECH, PN_NMD, PN_ECH_pct, PN_NMD_pct, NMD_CAlAGE],\
                      ["PN ECH", "NMD", "PN ECH%", "NMD%", "NMD_CALAGE", "NMD_TEMPLATES"]):
        if len(data) > 0:
            fill_template(sc_final_xl, data, pa.NC_PA_COL_SORTIE_NUM_PN, name, (2, 1), header=False)
        else:
            sc_final_xl.Sheets(name).Visible = False
    sc_final_xl.Save()
    sc_final_xl.Close(False)


def export_stock_file_data(STOCK, stock_wb):
    logger.info("   EXPORTATION DES DONNEES DU STOCK")
    fill_template(stock_wb, STOCK, pa.NC_PA_COL_SORTIE_NUM_ST, "STOCK", (2, 1), header=False)
    stock_wb.Save()
    stock_wb.Close(False)

def rename_and_select_cols(data, etab, nb_col_qual, col_name=[], etab_col=True):
    if etab_col:
        list_cols=[]
        i=1
        for col in data.columns:
            if etab.upper()==col.upper():
                list_cols.append(col+str(i))
                i=i+1
            else:
                list_cols.append(col)

        data.columns = list_cols
        data = data[list_cols[:nb_col_qual] + [x for x in list_cols[nb_col_qual:] if etab.upper()==x.upper()[:-1] and x.upper()[-1:].isnumeric()]]
        if col_name!=[]:
            data.columns = data.columns.tolist()[:-len(col_name)] + col_name
    else:
        data = data[data["BASSIN"]==up.current_etab].copy()

    return data.copy()


def export_data_generic(wb, range_name, final_wb, nb_col_qual, file_path, close=True, col_name=[], etab_col=True):
    data_to_export = ex.get_dataframe_from_range(wb, range_name)
    data_to_export = rename_and_select_cols(data_to_export, up.current_etab, nb_col_qual, col_name=col_name, etab_col=etab_col)
    if data_to_export.shape[1] <= nb_col_qual:
        logger.error(
            "L'établissement %s n'existe pas dans le fichier %s" % (up.current_etab, file_path))
    else:
        ex.write_df_to_range_name(data_to_export, range_name, final_wb, header=True)
        if close:
            wb.Close(False)


def export_lcr_nsfr_data():
    stock_wb = ex.try_close_open(up.stock_output_path)
    if up.update_param_lcr_ncr and up.map_lcr_nsfr:
        copyfile(up.path_sc_lcr_nsfr_template_file, up.sc_lcr_nsfr_output_path)
        logger.info("   EXPORTATION DES DONNEES DE LCR ET NSFR")
        sc_ln_final_xl = ex.try_close_open(up.sc_lcr_nsfr_output_path)

        ex.write_df_to_range_name(lcr_nsfr.table_nco_rl, "_PARAM_LCR", sc_ln_final_xl, header=True)
        ex.write_df_to_range_name(lcr_nsfr.table_nco_rl, "_PARAM_LCR", stock_wb, header=True)

        logger.info("      Lecture de : %s" % ntpath.basename(up.lcr_nsfr_files_name["MODE_CALC"]))
        outflow_wb = ex.try_close_open(up.lcr_nsfr_files_name["MODE_CALC"], read_only=True)
        for indic, close in zip(["NSFR","LCR"],[False, True]):
            range_name="_OUTFLOW_" + indic
            export_data_generic(outflow_wb, range_name, sc_ln_final_xl, 1, up.lcr_nsfr_files_name["MODE_CALC"], close=False, col_name=["APPLIQUER OUTFLOW"])
            export_data_generic(outflow_wb, range_name, stock_wb, 1, up.lcr_nsfr_files_name["MODE_CALC"], close=close, col_name=["APPLIQUER OUTFLOW"])

        logger.info("      Lecture de : %s" % ntpath.basename(up.lcr_nsfr_files_name["NSFR_COEFF_HISTO"]))
        nsfr_coeff_wb =  ex.try_close_open(up.lcr_nsfr_files_name["NSFR_COEFF_HISTO"], read_only=True)
        range_name = "_PARAM_NSFR"
        cols = ["Coefficient", "Coeff Outflow 0-6 mois", "Coeff Outflow 6-12 mois", "Coeff Outflow 12-inf mois"]
        export_data_generic(nsfr_coeff_wb, range_name, sc_ln_final_xl, 2, up.lcr_nsfr_files_name["NSFR_COEFF_HISTO"], close=False, col_name=cols)
        export_data_generic(nsfr_coeff_wb, range_name, stock_wb, 2, up.lcr_nsfr_files_name["NSFR_COEFF_HISTO"], col_name=cols)

        logger.info("      Lecture de : %s" % ntpath.basename(up.lcr_nsfr_files_name["NSFR_DESEMC"]))
        nsfr_coeff_wb =  ex.try_close_open(up.lcr_nsfr_files_name["NSFR_DESEMC"], read_only=True)
        range_name = "_NSFR_DESEMC"
        export_data_generic(nsfr_coeff_wb, range_name, sc_ln_final_xl, 2, up.lcr_nsfr_files_name["NSFR_DESEMC"], etab_col=False)

        logger.info("      Lecture de : %s" % ntpath.basename(up.lcr_nsfr_files_name["PARAM_LCR_SPEC"]))
        lcr_spec_wb =  ex.try_close_open(up.lcr_nsfr_files_name["PARAM_LCR_SPEC"], read_only=True)
        range_name = "_PARAM_LCR_SPEC"
        cols = ["NCO", "RL"]
        export_data_generic(lcr_spec_wb, range_name, sc_ln_final_xl, 3, up.lcr_nsfr_files_name["PARAM_LCR_SPEC"], close=False, col_name=cols)
        export_data_generic(lcr_spec_wb, range_name, stock_wb, 3, up.lcr_nsfr_files_name["PARAM_LCR_SPEC"], col_name=cols)

        for indic, nb_cols_qual, col in zip(["NSFR","LCR"],[2,1],[["ASF/RSF officiel"], ["VAL"]]):
            logger.info("      Lecture de : %s" % ntpath.basename(up.lcr_nsfr_files_name[indic + "_DAR"]))
            indic_wb =  ex.try_close_open(up.lcr_nsfr_files_name[indic + "_DAR"], read_only=True)
            range_name= "_" + indic + "_DAR"
            export_data_generic(indic_wb, range_name, stock_wb, nb_cols_qual, up.lcr_nsfr_files_name[indic + "_DAR"], col_name=col)

        sc_ln_final_xl.Save()
        sc_ln_final_xl.Close(False)

    else:
        stock_wb.Sheets("LCR DAR").Visible = False
        stock_wb.Sheets("NSFR DAR").Visible = False

    return stock_wb


def copy_template_file_in_ouput_dir():
    copyfile(up.path_map_mis_template_file, up.missing_map_output_file)
    copyfile(up.path_sc_tx_template_file, up.sc_taux_output_path)
    copyfile(up.path_sc_vol_template_file, up.sc_volume_output_path)
    copyfile(up.path_stock_template_file, up.stock_output_path)

def get_map_file():
    xl=ex.xl
    try:
        map_wb = xl.Workbooks(ntpath.basename(up.mapp_file))
    except:
        map_wb= xl.Workbooks.Open(up.mapp_file)

    return map_wb

def export_st_nmd_template_file(NMD_TEMPLATE_PROJ):
    if len(NMD_TEMPLATE_PROJ) == 0:
        NMD_TEMPLATE_PROJ = pd.DataFrame(["*"], columns = ["RCO_ALLOCATION_KEY"])
    NMD_TEMPLATE_PROJ.to_csv(up.nmd_template_output_path, index=False, sep=";", decimal=",")



def export_data(STOCK, NMD_TEMPLATE_PROJ, NMD_CAlAGE, PN_ECH=[], PN_NMD=[], PN_ECH_pct=[],
                PN_NMD_pct=[]):
    copy_template_file_in_ouput_dir()
    stock_wb = export_lcr_nsfr_data()
    export_stock_file_data(STOCK, stock_wb)
    export_scenario_file_data(PN_ECH, PN_NMD, PN_ECH_pct, PN_NMD_pct, NMD_CAlAGE)
    export_st_nmd_template_file(NMD_TEMPLATE_PROJ)
    export_missing_mappings()


