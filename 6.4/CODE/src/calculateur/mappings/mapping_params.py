from utils import excel_utils as ex
from . import others_mapping as rim
from . import mapping_products as mpp
from . import mapping_pass_alm as mpa
from ..rates_transformer.swap_rates_interpolator import Rate_Interpolator
import mappings.general_mappings as gmp
import os
import pandas as pd
import numpy as np
import logging

global input_file_map, rate_input_map, model_file_map, rate_input_files_path, curve_accruals_map, pricing_curves_map
global index_curve_tenor_map, mp_bassins, map_pass_alm
global gap_tx_params

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


def load_mappings(sources_folder, dar):
    global input_file_map, model_file_map, rate_input_files_path, curve_accruals_map, pricing_curves_map
    global index_curve_tenor_map, mp_bassins, map_pass_alm
    nom_map_file_path = os.path.join(sources_folder, "MAPPING", "MAPPING_NOMENCLATURE.xlsx")
    rc_map_file_path = os.path.join(sources_folder, "MAPPING", "MAPPING_PASS_ALM.xlsx")
    input_file_map, model_file_map, rate_input_files_path = get_mapping_input_files(nom_map_file_path, dar)

    gmp.load_general_mappings(rc_map_file_path, dar)

    map_wb = ex.try_close_open(rc_map_file_path, read_only=True)
    curve_accruals_map, pricing_curves_map, index_curve_tenor_map, mp_bassins \
        = rim.load_other_mappings(map_wb)

    map_pass_alm = mpa.lecture_mapping_pass_alm(map_wb)

    map_wb.Close(False)
    map_wb = None


def get_mapping_input_files(map_file_path, dar):
    mapping_wb = ex.try_close_open(map_file_path, read_only=True)
    ex.make_excel_value(mapping_wb, "_DAR_", str(dar.date()))
    mapping_wb.Sheets("NOMENCLATURE_SOURCES").Calculate()
    input_file_map = ex.get_dataframe_from_range(mapping_wb, "_NOMENCLATURE_SOURCES")
    model_file_map = ex.get_dataframe_from_range(mapping_wb, "_NOMENCLATURE_MODELES")
    rate_input_file_map = ex.get_dataframe_from_range(mapping_wb, "_NOMENCLATURE_COURBES_TX")
    mapping_wb.Close(False)
    mapping_wb = None
    return input_file_map, model_file_map, rate_input_file_map


def get_contract_sources_paths(etab, product, sources_folder):
    sources_file = input_file_map[(input_file_map["TYPE CONTRAT"] == mpp.products_map[product])][
        ["TYPE FICHIER", "CHEMIN", "EXTENSION"]].copy()
    sources_file = sources_file.set_index("TYPE FICHIER")
    sources_file["CHEMIN"] = sources_file["CHEMIN"].str.replace("\[ENTITE]", etab, regex=True)
    sources_file["CHEMIN"] = [os.path.join(sources_folder, x) for x in sources_file["CHEMIN"]]
    sources_file["DELIMITER"] = np.where(sources_file["EXTENSION"] == ".csv", ";", "\t")
    sources_file["DECIMAL"] = np.where(sources_file["EXTENSION"] == ".csv", ",", ".")
    sources_params = sources_file[["CHEMIN", "DELIMITER", "DECIMAL"]].to_dict('index')
    return sources_params


def get_model_file_path(model, sources_folder):
    global model_file_map
    models_files = model_file_map[["TYPE FICHIER", "CHEMIN"]].copy()
    models_files =  models_files[models_files["TYPE FICHIER"] == model].copy()
    models_files["CHEMIN"] = [os.path.join(sources_folder, x) for x in models_files["CHEMIN"]]
    models_file_path = models_files["CHEMIN"].iloc[0]
    return models_file_path

def get_model_wb(model_file_path):
    model_wb = None
    # ut.check_version_templates(model_file_path.split("\\")[-1], version="4.1", path=model_file_path,
    #                           open=True, warning=False)
    model_wb = ex.try_close_open(model_file_path, read_only=True)
    ex.unfilter_all_sheets(model_wb)
    return model_wb

def get_rate_file_path(rate_file, sources_folder):
    global rate_input_files_path
    rate_files = rate_input_files_path[["TYPE FICHIER", "CHEMIN"]].copy()
    rate_files =  rate_files[rate_files["TYPE FICHIER"] == rate_file].copy()
    rate_files["CHEMIN"] = [os.path.join(sources_folder, x) for x in rate_files["CHEMIN"]]
    rate_file_path = rate_files["CHEMIN"].iloc[0]
    return rate_file_path

def load_rate_params(rate_file_path, scenario, sources_folder, interpolate_curves):
    global rate_input_files_path, rate_input_map, curve_accruals_map, mp_bassins
    global gap_gestion_index, gap_indexes, gap_gestion_reg

    logger.info("   LOADING '%s' RATE PARAMETERS" % scenario)

    tx_curves = pd.read_csv(rate_file_path, sep=";", decimal=",")

    tx_curves = tx_curves[tx_curves["SCENARIO"] == scenario].copy()

    tx_tci_path = os.path.join(sources_folder, rate_input_files_path["CHEMIN"].iloc[2])
    tx_tci_values = pd.read_csv(tx_tci_path, sep=";", decimal=",")
    tx_tci_values = tx_tci_values.fillna("*").drop(["dar", "all_t"], axis=1)

    cle = ["reseau", "company_code", "devise", "contract_type", "family", "rate_category"]
    tx_tci_values['new_key'] = tx_tci_values[cle].apply(lambda row: '$'.join(row.values.astype(str)), axis=1)
    tx_tci_values = tx_tci_values.drop_duplicates(subset=cle).set_index('new_key').drop(columns=cle, axis=1).copy()

    try:
        zc_curves_path = os.path.join(sources_folder, rate_input_files_path["CHEMIN"].iloc[1])
        zc_curves_data = pd.read_csv(zc_curves_path, sep=";", decimal=",")
    except:
        logger.info("NO ZC CURVES AVAILABLE")
        zc_curves_data = []

    tx_params = {"curves_df": {"data": tx_curves, "cols": rim.cols_ri, "max_proj": rim.max_taux_cms,
                               "curve_code": "CODE COURBE", "tenor": "TENOR",
                               "maturity_to_days": rim.maturity_to_days, "curve_name_taux_pel": "TAUX_PEL",
                               "tenor_taux_pel": "12M1D"},
                 "accrual_map": {'data': curve_accruals_map, "accrual_conv_col": "ACCRUAL_CONVERSION",
                                 "type_courbe_col": "TYPE DE COURBE", "accrual_method_col": "ACCRUAL_METHOD",
                                 "alias": "ALIAS",
                                 "standalone_const": "Standalone index", "curve_name": "CURVE_NAME"},
                 "ZC_DATA": {"data": zc_curves_data},
                 "map_pricing_curves": {"data": pricing_curves_map,
                                        "col_pricing_curve": "COURBE PRICING"},
                 "map_index_curve_tenor": {"data": index_curve_tenor_map,
                                           "col_curve": "CURVE_NAME", "col_tenor": "TENOR"},
                 "tci_vals": {"data": tx_tci_values}}

    rate_int_cls = Rate_Interpolator()
    if interpolate_curves:
        tx_params["dic_tx_swap"] = rate_int_cls.interpolate_curves(tx_params)
    else:
        tx_params["dic_tx_swap"] = []

    return tx_params


def load_gap_tx_params(map_wb):
    global gap_tx_params
    mapping_data = ex.get_dataframe_from_range(map_wb, "_GAP_GESTION_INDEX")
    gap_index_coeff = pd.melt(mapping_data, id_vars=["GAP"], value_vars=["TLA", "CEL", "TLB", "LEP"],
                              var_name="INDEX_AG", value_name="PRCT").set_index("INDEX_AG")

    gap_index_coeff_tf = gap_index_coeff[gap_index_coeff["GAP"] == "GAP TX TF"].copy()

    gap_index_coeff_inf = gap_index_coeff[gap_index_coeff["GAP"] == "GAP TX INF"].copy()

    mapping_data = ex.get_dataframe_from_range(map_wb, "_INDEX_REG")
    index_params = pd.melt(mapping_data, value_vars=["TLA", "CEL", "TLB", "LEP"],
                           var_name="INDEX_AG", value_name="INDEX").set_index("INDEX")
    index_params = index_params[~index_params.index.isnull()].copy()

    gep_reg_coeff = ex.get_dataframe_from_range(map_wb, "_GAP_GESTION_INDEX")

    gap_tx_params = {"INDEXS": index_params, "GAP_TF_COEFF_INDEX": gap_index_coeff_tf,
                     "GAP_INF_COEFF_INDEX": gap_index_coeff_inf,
                     "GAP REG": gep_reg_coeff, "COL_INDEX_AG": "INDEX_AG", "COL_INDEX": "INDEX",
                     "COL_VAL": "PRCT", "GAP_TF": "TF", "GAP_INF": "INF"}
