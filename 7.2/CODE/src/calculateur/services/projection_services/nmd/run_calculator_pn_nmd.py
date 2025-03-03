from os import path
import pandas as pd
import numpy as np
from mappings.pass_alm_fields import PASS_ALM_Fields
from calculateur.data_transformers.data_in.nmd.data_formater.class_nmd_pn_formater import Data_NMD_PN_Formater
from calculateur.models.agregation.class_agregation import Agregation
from calculateur.models.main_calculator import Calculator
from calculateur.models.data_manager.data_format_manager.class_data_formater import Data_Formater
from calculateur.models.data_manager.data_format_manager.class_fields_manager import Data_Fields
from calculateur.data_transformers.data_out.nmd.calculate_nmd_pn import NMD_PN_PROJ



import logging
logger = logging.getLogger(__name__)


def run_calculator_pn_nmd(dar, horizon, source_data,  name_product, etab, cls_nmd_tmp,
                          compiled_indics_st, cls_nmd_spreads = None, tx_params=[],
                          agregation_level="AG",
                          exit_indicators_type=["GPLIQ"], gap_tx_params = [],
                          with_dyn_data = False, type_rm = "NORMAL",
                          max_pn=60, batch_size = 5000, tci_contract_perimeter=[],
                          output_data_type = "cls_ag", output_mode="dump", exec_mode="simul"):

    cls_pn_nmd = load_pn_data(source_data["PN"], dar, source_data["MODELS"]["NMD"]["DATA"],
                              horizon, tx_params, etab, type_rm, cls_nmd_tmp, max_pn, cls_nmd_spreads,
                              batch_size, exec_mode)

    cls_ag = Agregation(agregation_level, cls_pn_nmd.cls_fields, horizon, output_mode=output_mode)

    if output_data_type == "pn_monhtly_flow":
        pn_to_generate = []
        get_flux_pn = True
        get_sc_variables = False
        custom_report = False
    elif output_data_type == "pn_proj":
        dic_sc_all = {}
        get_sc_variables = True
        custom_report = False
        get_flux_pn = False
    elif output_data_type == "cls_ag":
        get_flux_pn = False
        get_sc_variables = False
        custom_report = True

    static_data = load_static_data(source_data, cls_nmd_spreads)

    calc_cls = Calculator(static_data, dar, horizon, name_product, tx_params=tx_params, agregation_level="NMD_DT",
                          exit_indicators_type = exit_indicators_type + ["PN_NMD"], gap_tx_params=gap_tx_params,
                          with_dyn_data = with_dyn_data, output_mode = "dataframe",
                          tci_contract_perimeter=tci_contract_perimeter)

    for batch in cls_pn_nmd.dic_data_nmd.keys():
        logger.debug("Traitement du batch des entités : %s" % str(batch))
        dynamic_data = load_dynamic_data(cls_pn_nmd.dic_data_nmd[batch], batch_size)
        calc_cls.init_main_data(dynamic_data)
        cls_ag_pn_nmd = calc_cls.run_calculator()

        calc_nmd_pn_f = NMD_PN_PROJ(cls_ag_pn_nmd.compiled_indics, compiled_indics_st,
                                    cls_ag_pn_nmd.keep_vars_dic, horizon, cls_pn_nmd, etab,
                                    get_flux_pn=get_flux_pn, custom_report=custom_report,
                                    get_sc_variables = get_sc_variables,
                                    cls_ag=cls_ag)

        if output_data_type == "pn_monhtly_flow":
            pn_to_generate_batch = calc_nmd_pn_f.calculate_pn_nmd()
            if len(pn_to_generate) == 0:
                pn_to_generate = pn_to_generate_batch.copy()
            else:
                pn_to_generate = pd.concat([pn_to_generate, pn_to_generate_batch])
        elif output_data_type == "pn_proj" :
            dic_sc = calc_nmd_pn_f.calculate_pn_nmd()
            if dic_sc_all == {}:
                dic_sc_all = dic_sc.copy()
            else:
                for key in dic_sc_all.keys():
                    if key != "data_index":
                        dic_sc_all[key] = np.vstack([dic_sc_all[key], dic_sc[key]])
                    else:
                        dic_sc_all[key] = np.concatenate([dic_sc_all[key], dic_sc[key]], axis=0)
        else:
            calc_nmd_pn_f.calculate_pn_nmd()

    if output_data_type == "pn_monhtly_flow":
        return pn_to_generate
    elif output_data_type == "pn_proj":
        return dic_sc_all
    else:
        return cls_ag


#@profile
def load_pn_data(source_data, dar_usr, model_wb, horizon, tx_params, etab,
                 type_rm, cls_nmd_tmp, max_pn, cls_nmd_spreads, batch_size, exec_mode):
    cls_data = Data_Fields()
    cls_format = Data_Formater(cls_data)
    cls_pa_fields = PASS_ALM_Fields()

    check_locations_existence(source_data, cls_nmd_spreads)

    cls_nmd = Data_NMD_PN_Formater(cls_data, cls_format, cls_pa_fields, source_data, dar_usr, model_wb, horizon,
                                   tx_params, etab, cls_nmd_tmp, max_pn,  type_rm = type_rm, batch_size=batch_size,
                                   exec_mode=exec_mode)

    cls_nmd.generate_templated_pn_data()

    return cls_nmd

def check_locations_existence(source_data, cls_nmd_spreads):
    if not "DATA" in source_data["LDP"]:
        input_pn = source_data["LDP"]["CHEMIN"]
        if not path.isfile(input_pn):
            logger.error("    Le fichier " + input_pn + " n'existe pas")
            raise ImportError("    Le fichier " + input_pn + " n'existe pas")

    if cls_nmd_spreads is None:
        input_marges = source_data["MARGE-INDEX"]["CHEMIN"]
        if not path.isfile(input_marges):
            logger.error("    Le fichier " + input_marges + " n'existe pas")
            raise ImportError("    Le fichier " + input_marges + " n'existe pas")

def load_dynamic_data(pn_data, batch_size):
    alim_data = {}
    alim_data["LDP"] = chunkized_data(pn_data, batch_size)
    alim_data["PAL"] = []
    alim_data["CF"] = []
    return alim_data

def load_static_data(source_data, cls_nmd_spreads):
    alim_data = {}
    if cls_nmd_spreads is None:
        alim_data["MARGE-INDEX"] = pd.read_csv(source_data["MARGE-INDEX"]["CHEMIN"], delimiter="\t", decimal=",",
                                               engine='python', encoding="ISO-8859-1")
    else:
        alim_data["MARGE-INDEX"] = cls_nmd_spreads.data_tx_spread
        alim_data["TARGET-RATES"] = cls_nmd_spreads.data_tx_cible

    alim_data["modele_wb"] = source_data["MODELS"]["NMD"]["DATA"]

    return alim_data

def chunkized_data(dic_data_nmd, chunk):
    data_ldp_all = []
    for key, data_part in dic_data_nmd.items():
        nb_parts = int(float(key.split("_")[0]))
        real_chunk = nb_parts * (chunk//nb_parts)
        data_ldp_all = data_ldp_all + [data_part.iloc[i:i + real_chunk] for i in range(0, len(data_part), real_chunk)]
    return data_ldp_all