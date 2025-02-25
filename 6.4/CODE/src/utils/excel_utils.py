# -*- coding: utf-8 -*-
"""
Created on Sun May 31 11:06:57 2020

@author: Hossayne
"""
import win32com.client as win32
from params import simul_params as sp
import ntpath
import pandas as pd
import sys
import logging
import dateutil
import pywintypes
import win32process
import psutil
import win32clipboard as wc
import os
import numpy as np
from datetime import datetime
from ctypes import windll

global xl, xl_interface, interface_name, xl_open

logger = logging.getLogger(__name__)

EXCEL_PASS_WORD = "LAX_MILGRAM"

""" CONSTANTES EXCEL """
xltoright = -4161
xldown = -4121
xlPasteFormats = -4122
xlNone = -4142
xlPasteValues = -4163
xlManual = -4135
xlAutomatic = -4105
xlPasteAll = -4104


def able_Excel(able):
    global xl
    try:
        # xl_moteur.Sheets(gp.param_gen_wsh).Activate()
        xl.EnableEvents = able
        xl.DisplayAlerts = able
        xl.ScreenUpdating = able
        xl.Visible = able
        xl.Calculation = xlAutomatic
        xl.AskToUpdateLinks = able
    except:
        pass


def kill_excel():
    global excel_process_id
    psutil.Process(excel_process_id).terminate()


def clear_clipboard():
    wc.OpenClipboard()
    wc.EmptyClipboard()
    wc.CloseClipboard()



def load_excel():
    global xl, excel_process_id
    clear_clipboard()
    """ Appel de l'appli Excel existante"""
    try:
        xl = win32.DispatchEx('Excel.Application')
        t, excel_process_id = win32process.GetWindowThreadProcessId(xl.Hwnd)

    except:
        import os
        import re
        import shutil
        # Remove cache and try again.
        MODULE_LIST = [m.__name__ for m in sys.modules.values()]
        for module in MODULE_LIST:
            if re.match(r'win32com\.gen_py\..+', module):
                del sys.modules[module]
        shutil.rmtree(os.path.join(os.environ.get('LOCALAPPDATA'), 'Temp', 'gen_py'))
        xl = win32.DispatchEx('Excel.Application')
        t, excel_process_id = win32process.GetWindowThreadProcessId(xl.Hwnd)

    """ Désactivation des alertes Excel """
    able_Excel(False)


def load_excel_interface(args):
    global xl, xl_interface, interface_name, excel_process_id, xl_open
    """ Chargement du nom du fichier Excel d'Interface """
    if not args.use_json:
        xl_open = win32.GetObject(Pathname=args.ref_file_path)
        xl_open = xl_open.Application

        try:
            interface_name = args.ref_file_path
        except:
            raise ValueError("L'interface EXCEL est introuvable")

        """ Chargement du fichier Excel d'Interface """
        try:
            xl_interface = xl_open.Workbooks(ntpath.basename(interface_name))
        except:
            raise NameError("L'interface Excel " + interface_name + " n'est pas ouverte, Veuillez l'ouvrir!")

        """ Défiltrage des feuilles Excel de l'interface """
        unfilter_all_sheets(xl_interface)


def excel_to_python_date(excel_date):
    dt = datetime.fromordinal(datetime(1900, 1, 1).toordinal() + excel_date - 2)
    return dt


def get_cell_value_from_range(wb, name_range):
    rango = wb.Names(name_range).RefersToRange
    return get_excel_value_from_range(rango)


def try_close_open(file_path, read_only=False):
    global xl
    try:
        xl.Workbooks(ntpath.basename(file_path)).Close(False)
    except:
        pass

    return xl.Workbooks.Open(file_path, None, read_only)


def empty_clipboard():
    try:
        if windll.user32.OpenClipboard(None):
            windll.user32.EmptyClipboard()
            windll.user32.CloseClipboard()
    except:
        pass

def unfilter_all_sheets(wb):
    for i in range(1, wb.Worksheets.count + 1):
        try:
            wb.Worksheets(i).Unprotect(Password=EXCEL_PASS_WORD)
            wb.Worksheets(i).ShowAllData()
            wb.Worksheets(i).Protect(Password=EXCEL_PASS_WORD)
        except:
            pass


def make_excel_value(wb, name_range, value, protected=False):
    try:
        if protected:
            worksheet = wb.Names(name_range).RefersToRange.Worksheet
            worksheet.Unprotect(EXCEL_PASS_WORD)
        wb.Names(name_range).RefersToRange.Value = value
        if protected:
            worksheet.Protect(EXCEL_PASS_WORD)
    except:
        raise ValueError("La Range nommée  " + name_range + " ne peut être trouvée")


def export_dataframe(wb, name_range, value):
    try:
        wb.Names(name_range).RefersToRange.Value = value
    except:
        raise ValueError("La Range nommée  " + name_range + " ne peut être trouvée")


def get_value_from_named_ranged(wb, name_range, alert=""):
    try:
        value = wb.Names(name_range).RefersToRange.Value
    except:
        raise ValueError("La Range nommée  " + name_range + " ne peut être trouvée")
    if alert != "" and (value == "" or value == None):
        logger.error("win_alert:" + alert)
    else:
        return "" if value == None else value


def get_excel_value_from_range(rango):
    value = rango.Value
    return "" if value == None else value


def clear_excel_range(wb, name_range):
    wb.Names(name_range).RefersToRange.Clear()


def export_df_to_xl_with_range_name(wb, data, range_name_choc, protected=False, header=True, offset=1):
    write_df_to_range_name(data, range_name_choc, wb, protected=protected, header=header, offset=offset)


def write_df_to_range_name(data, range_name, wb, header=True, protected=False, offset=1, end_xldown=False):
    range = wb.Names(range_name).RefersToRange
    if offset > 1:
        range = range.Offset(offset)
    address = range.Row, range.Column
    worksheet = range.Worksheet
    if protected:
        worksheet.Unprotect(EXCEL_PASS_WORD)
    export_dataframe_to_xl2(data, worksheet, [], address, header=header, end_xldown=end_xldown)
    if protected:
        worksheet.Protect(EXCEL_PASS_WORD)


def export_dataframe_to_xl2(data, ws, str_variables, address, header=True, end_xldown=False):
    data.loc[:, str_variables] = data.loc[:, str_variables].astype(str)
    data = data.replace(np.nan, '', regex=True)
    if header:
        vals = (tuple(["'" + str(x) for x in data.columns]),) + tuple(data.itertuples(index=False, name=None))
    else:
        vals = tuple(data.itertuples(index=False, name=None))

    rowo = address[0]

    if end_xldown:
        if ws.Cells(address[0], address[1]).Value is not None and ws.Cells(address[0] + 1,
                                                                           address[1]).Value is not None:
            rowo = ws.Range(ws.Cells(address[0], address[1]), ws.Cells(address[0], address[1])).End(xldown).Row + 1
        elif ws.Cells(address[0], address[1]).Value is not None and ws.Cells(address[0] + 1, address[1]).Value is None:
            rowo = address[0] + 1

    if header:
        ws.Range(ws.Cells(rowo, address[1]),
                 ws.Cells(rowo + len(data), address[1] + len(data.columns) - 1)).Value = vals
    else:
        ws.Range(ws.Cells(rowo, address[1]),
                 ws.Cells(rowo + len(data) - 1, address[1] + len(data.columns) - 1)).Value = vals


def export_dataframe_to_xl_chunk(data, ws, str_variables, address, header=True, chunk=10000, end_xldown=False):
    data.loc[:, str_variables] = data.loc[:, str_variables].astype(str)
    data = data.replace(np.nan, '', regex=True)

    sizo = data.shape[0]
    chunk_size = chunk
    data_list = [data[i:i + chunk_size].copy() for i in range(0, sizo, chunk_size)]
    rowo = address[0]

    if end_xldown:
        if ws.Cells(address[0], address[1]).Value is not None and ws.Cells(address[0] + 1,
                                                                           address[1]).Value is not None:
            rowo = ws.Range(ws.Cells(address[0], address[1]), ws.Cells(address[0], address[1])).End(xldown).Row + 1
        elif ws.Cells(address[0], address[1]).Value is not None and ws.Cells(address[0] + 1, address[1]).Value is None:
            rowo = address[0] + 1

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

        if header_p:
            ws.Range(ws.Cells(rowo, address[1]), ws.Cells(rowo + len(data_part), len(data_part.columns))).Value = vals
        else:
            ws.Range(ws.Cells(rowo, address[1]),
                     ws.Cells(rowo + len(data_part) - 1, len(data_part.columns))).Value = vals

        if header_p:
            rowo = rowo + len(data_part) + 1
        else:
            rowo = rowo + len(data_part)


def try_close_workbook(wb, name, save=False):
    try:
        wb.Close(SaveChanges=save)
    except:
        logger.info('Erreur pendant la fermeture du fichier: {}'.format(name))


def close_workbook_by_name(workbook_name):
    global xl
    if os.path.basename(workbook_name) in [wb.Name for wb in xl.Workbooks]:
        try_close_workbook(xl.Workbooks[os.path.basename(workbook_name)], False)


def set_value_to_named_cell(wb, name_range, value, protected=False):
    range = wb.Names(name_range).RefersToRange
    worksheet = range.Worksheet
    if protected:
        worksheet.Unprotect(EXCEL_PASS_WORD)
        wb.Names(name_range).RefersToRange.Value = value
        worksheet.Protect(EXCEL_PASS_WORD)
    else:
        wb.Names(name_range).RefersToRange.Value = value
    worksheet.Calculate()


def get_sheet_name_from_named_range(wb, name_range):
    try:
        return wb.Names(name_range).RefersToRange.Worksheet.Name
    except:
        raise ValueError("La Range nommée  " + name_range + " ne peut être trouvée")


def clear_range_content(wb, range_name, offset=1):
    rango = wb.Names(range_name).RefersToRange
    sheet = rango.Worksheet
    deb_col = rango.Column
    if get_excel_value_from_range(rango.Offset(1, 2)) == "":
        max_col = deb_col
    else:
        max_col = rango.End(xltoright).Column

    deb_row = rango.Row
    if get_excel_value_from_range(rango.Offset(2, 1)) == "" or get_excel_value_from_range(rango) == "":
        max_row = deb_row
    else:
        max_row = rango.End(xldown).Row
    rango = rango.Offset(offset)
    sheet.Unprotect(EXCEL_PASS_WORD)
    sheet.Range(rango, rango.Offset(max_row - deb_row + 1, max_col - deb_col + 1)).ClearContents()
    sheet.Protect(EXCEL_PASS_WORD)


def get_dataframe_from_range(wb, name_range, header=True, alert="", replace_datetime=False):
    try:
        rango = wb.Names(name_range).RefersToRange
    except:
        raise ValueError("La Range nommée  " + name_range + " ne peut être trouvée")

    sheet = rango.Worksheet
    deb_col = rango.Column
    if get_excel_value_from_range(rango.Offset(1, 2)) == "":
        max_col = deb_col
    else:
        max_col = rango.End(xltoright).Column

    deb_row = rango.Row
    if get_excel_value_from_range(rango.Offset(2, 1)) == "" or get_excel_value_from_range(rango) == "":
        max_row = deb_row
    else:
        max_row = rango.End(xldown).Row

    data = sheet.Range(rango, rango.Offset(max_row - deb_row + 1, max_col - deb_col + 1)).Value

    return RangeToDataframe(data, header=header, replace_datetime=replace_datetime)


def get_dataframe_from_range_chunk_np(wb, name_range, header=True, replace_datetime=False):
    try:
        rango = wb.Names(name_range).RefersToRange
    except:
        raise ValueError("La Range nommée  " + name_range + " ne peut être trouvée")

    sheet = rango.Worksheet
    deb_col = rango.Column
    if get_excel_value_from_range(rango.Offset(1, 2)) == "":
        max_col = deb_col
    else:
        max_col = rango.End(xltoright).Column

    deb_row = rango.Row
    if get_excel_value_from_range(rango.Offset(2, 1)) == "" or get_excel_value_from_range(rango) == "":
        max_row = deb_row
    else:
        max_row = rango.End(xldown).Row

    if max_row == deb_row:
        data = sheet.Range(rango, rango.Offset(max_row - deb_row + 1, max_col - deb_col + 1)).Value
        data = RangeToDataframe(data, header=header, replace_datetime=replace_datetime)
        return data
    else:
        chunk = 30000
        data = []
        for i in range(1, max_row - deb_row + 2, chunk):
            data_tmp = sheet.Range(rango.Offset(i),
                                   rango.Offset(min(i + chunk - 1, max_row - deb_row + 1), max_col - deb_col + 1)).Value
            if len(data) == 0:
                cols = list(data_tmp[0])
                data.append(RangeToDataframe(data_tmp, header=True))
            else:
                data.append(RangeToDataframe(data_tmp, header=False, cols=cols))

        data = pd.concat(data)
        return data


def sheet_in_wb(wb, namo_ws):
    for ws in wb.Sheets:
        if ws.Name == namo_ws:
            return True
    return False


def is_already_open(xl, name_wb):
    for wb in xl.Workbooks:
        if wb.Name == name_wb:
            return True
    return False


def open_wb_and_get_data_from_named_range(xl, full_path_file, name_range_data):
    if is_already_open(xl, ntpath.basename(full_path_file)):
        MSG = "PLEASE CLOSE FILE ntpath.basename(full_path_file) and relaunch the program"
        logger.error(MSG)
        raise ValueError(MSG)

    wb = xl.Workbooks.Open(full_path_file, None, True)

    data = get_dataframe_from_range(wb, name_range_data, header=True)

    wb.Close()

    return data

def write_df_to_address_chunk(data, sheet, address, header=True, chunk=10000):
    address = address[0], address[1]
    export_dataframe_to_xl_chunk(data, sheet, [], address, header=header, chunk=chunk)


def write_df_to_range_name_chunk(data, range_name, wb, header=True, offset=0, chunk=10000, end_xldown=False):
    range = wb.Names(range_name).RefersToRange.Offset(offset + 1)
    address = range.Row, range.Column
    worksheet = range.Worksheet
    worksheet.Unprotect(EXCEL_PASS_WORD)
    export_dataframe_to_xl_chunk(data, worksheet, [], address, header=header, chunk=chunk,end_xldown=end_xldown)
    worksheet.Protect(EXCEL_PASS_WORD)

def RangeToDataframe(data, header=False, replace_datetime=False, cols=[]):
    df = [list(x) for x in data]

    if replace_datetime:
        replace_datetime_data(df[1:])

    if header:
        df = pd.DataFrame(df[1:], columns=df[0])
    else:
        df = pd.DataFrame(df)
        if cols != []:
            df.columns = cols

    return df


def replace_datetime_data(data):
    for i in range(0, len(data)):
        for j in range(0, len(data[i])):
            if isinstance(data[i][j], pywintypes.TimeType):
                data[i][j] = dateutil.parser.parse(str(data[i][j])).replace(tzinfo=None)


def write_df_to_range_name2(data, range_name, wb, header=True, offset=1, end_xldown=False):
    range = wb.Names(range_name).RefersToRange.Offset(offset)
    address = range.Row, range.Column
    worksheet = range.Worksheet
    worksheet.Unprotect(EXCEL_PASS_WORD)
    export_dataframe_to_xl2(data, worksheet, [], address, header=header, end_xldown=end_xldown)
    worksheet.Protect(EXCEL_PASS_WORD)
