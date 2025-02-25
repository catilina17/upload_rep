import logging
import utils.general_utils as ut
import os
from modules.scenario.rate_services import tx_referential as tx_ref
from modules.scenario.parameters import user_parameters as up
import modules.scenario.referentials.general_parameters as gp
import params.version_params as vp
import utils.excel_utils as excel_helper

logger = logging.getLogger(__name__)

global stock_data
global pn_df
pn_df = {}
stock_data = []


def get_bassin_name(scenario_rows):
    if len(scenario_rows['ETABLISSEMENTS'].unique()) > 1:
        raise ValueError(
            "La liste des établissements du scénario:  {},  n'est pas la même dans toute les lignes".format(
                scenario_rows[gp.CN_NOM_SCENARIO]))

    etabs_liste = get_etabs_from_string_liste(scenario_rows['ETABLISSEMENTS'].iloc[0])
    valide_liste = check_if_etab_data_are_in_alim_output_dir(etabs_liste)
    return valide_liste


def check_if_etab_data_are_in_alim_output_dir(etabs_liste, warning=False):
    valide_etab_liste = [x for x in etabs_liste if os.path.exists(os.path.join(up.alim_dir_path, x))]
    not_found_etabs = [x for x in etabs_liste if x not in valide_etab_liste]
    if warning:
        if len(not_found_etabs) >= 1:
            logger.warning("Les établissements suivants ne sont pas disponibles dans le dossier de l'ALIM : {}".format(
                ','.join(not_found_etabs)))
    return valide_etab_liste


def get_etabs_from_string_liste(string_liste):
    bassins_liste = [x.strip() for x in string_liste.split(',')]

    for i, field in enumerate(bassins_liste):
        if field in up.codification_etab.columns and field not in up.all_etabs:
            etab_liste = list(up.codification_etab.loc[:, field].dropna())
            bassins_liste[i:i + 1] = etab_liste
    unique_liste = []
    [unique_liste.append(x) for x in bassins_liste if x not in unique_liste and x in up.all_etabs]
    return unique_liste


def get_calc_curves(mapping_wb):
    return excel_helper.get_dataframe_from_range(mapping_wb, '_COURBES_PRICING_PN')


def get_rate_code_curve_mapping(mapping_wb):
    return excel_helper.get_dataframe_from_range(mapping_wb, '_MAP_RATE_CODE_CURVE')


def get_filtering_tenor_curves(mapping_wb):
    return excel_helper.get_dataframe_from_range(mapping_wb, '_FILTRE_TENOR')


def get_curves_to_interpolate(mapping_wb):
    return excel_helper.get_dataframe_from_range(mapping_wb, 'COURBES_A_INTERPOLER')


def set_empty_curves_to_zero(df):
    df.iloc[df[df.isnull().sum(axis=1) > 238].index, 5:] = 0
    return df


def get_pn_df(input_wb, pn_range_name, etab):
    logger.debug('      Lecture des PN à partir de {}'.format(pn_range_name))
    if etab == "NTX":
        df = excel_helper.get_dataframe_from_range_chunk(input_wb, pn_range_name)
    else:
        df = excel_helper.get_dataframe_from_range(input_wb, pn_range_name)

    columns_list = list(df.columns)
    index_columns = [x for x in columns_list if x not in ["M" + str(i) for i in range(0, tx_ref.NB_PROJ_TAUX + 1)]
                     and x != "IND03"]
    df.set_index(index_columns, inplace=True)
    return df


def get_template_df(input_wb, pn_range_name):
    logger.debug('      Lecture des PN à partir de {}'.format(pn_range_name))
    df = excel_helper.get_dataframe_from_range(input_wb, pn_range_name)
    return df


def get_pass_alm_referential(mapping_wb):
    logger.debug('      Récupération du référentiel de courbes')
    ref_pass_alm = excel_helper.get_dataframe_from_range(mapping_wb, gp.RN_REF_PASS_ALM_FST_CELL)
    ref_pass_alm.drop_duplicates(inplace=True)
    return ref_pass_alm


def get_input_tx_workbook(etab):
    _input_file_path = _get_input_file_path(etab, 'SC_TAUX')
    input_wb = excel_helper.try_close_open(_input_file_path, read_only=True)
    return _input_file_path, input_wb


def get_input_pn_workbook(etab):
    _input_file_path = _get_input_file_path(etab, 'SC_VOLUME')
    input_wb = excel_helper.try_close_open(_input_file_path, read_only=True)
    return _input_file_path, input_wb


def get_input_lcr_nsfr_workbook(etab):
    _input_file_path = ""
    try:
        _input_file_path = _get_input_file_path(etab, 'SC_LCR_NSFR')
    except:
        pass
    return _input_file_path


def get_zc_data(file_path):
    zc_wb = excel_helper.try_close_open(file_path, read_only=True)
    ut.check_version_templates(file_path.split("\\")[-1], version=vp.version_zc, wb=zc_wb, open=False, warning=True)
    zc_data = excel_helper.get_dataframe_from_range(zc_wb, '_ZC_DATA_EVE')
    excel_helper.try_close_workbook(zc_wb, file_path.split("\\")[-1])
    return zc_data


def get_stock_workbook(etab):
    _stock_file_path = _get_input_file_path(etab, 'STOCK_AG_')
    return _stock_file_path


def get_stock_nmd_template_workbook(etab):
    _stock_nmd_template_file_path = _get_input_file_path(etab, 'STOCK_NMD_TEMPLATE')
    return _stock_nmd_template_file_path


def _get_input_file_path(etab, file_substring):
    if not os.path.isdir(os.path.join(up.alim_dir_path, etab)):
        raise IOError("Le dossier relatif à l'entité " + etab + " n'existe pas dans le dossier de sortie de l'ALIM")
    etab_dir_file_path = os.path.join(up.alim_dir_path, etab)
    list_files = [x[2] for x in os.walk(etab_dir_file_path)][0]
    sc_files = [file for file in list_files if file.startswith(file_substring) and '~' not in file]

    if len(sc_files) != 1:
        raise IOError("Le dossier relatif à l'entité " + etab + " contient %s fichier de type %s" % (
            len(sc_files), file_substring))

    return os.path.join(etab_dir_file_path, sc_files[0])
