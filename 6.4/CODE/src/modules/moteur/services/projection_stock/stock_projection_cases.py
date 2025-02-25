import numpy as np


def generate_specific_cases(dic_cases, dic_gen, dic_mni, pv, comp_data):
    gen_case_tv_ech_floor(dic_cases, dic_gen, dic_mni, pv, comp_data)
    gen_case_tv_ech_nonfloor(dic_cases, dic_gen, dic_mni, pv)
    gen_case_other(dic_cases, dic_gen, dic_mni, pv)

def gen_case_tv_ech_floor(dic_cases, dic_gen, dic_mni, pv, comp_data):
    cas = "TV & ECH & FLOOR"
    dic_cases[cas] = (~dic_gen["IS TF"]) & (~dic_gen["IS NOT ECH"]) \
                                    & (dic_gen["IS FLOOR"])

    dic_mni[cas] = pv["lmn_j"] + ((pv["enc_ref_j"] - pv["tem_j"]) * (np.maximum(comp_data["floor_values"], \
                  pv["tx_sc_j"]) - np.maximum(comp_data["floor_values"], pv["tx_ref_j"]))) * pv["bsc_j"]

def gen_case_tv_ech_nonfloor(dic_cases, dic_gen, dic_mni, pv):
    cas = "TV & ECH & ~FLOOR"
    dic_cases[cas] = (~dic_gen["IS TF"]) & (~dic_gen["IS NOT ECH"]) & (~dic_gen["IS FLOOR"])

    dic_mni[cas] = pv["lmn_j"] + ((pv["enc_ref_j"] - pv["tem_j"]) * (pv["tx_sc_j"] - pv["tx_ref_j"])) * pv["bsc_j"]

def gen_case_other(dic_cases, dic_gen, dic_mni, pv):
    cas = "OTHER"
    dic_cases[cas] = (dic_gen["IS TF"]) | (dic_gen["IS NOT ECH"])
    dic_mni[cas] = pv["lmn_j"]
