global max_taux_cms, NC_CODE, cols_ri, maturity_to_days, rate_input_map, NC_INDEX_CALC
from utils import excel_utils as ex, general_utils as gu
from modules.scenario.rate_services import maturities_referential as mr

def load_other_mappings(map_wb):
    global NC_INDEX_CALC
    rate_input_map = {}
    load_rate_input_file_parameters()

    name_ranges = ["_MP_CURVE_ACCRUALS2", "_DATA_PN_PRICING", "_MAP_RATE_CODE_CURVE",
                   "_MP_BASSIN_SOUS_BASSIN"]
    renames = [{}, {}, {}, {},{}]
    keys = [["CURVE_NAME", "TENOR_BEGIN"], ["CONTRAT", "DIM2", "BILAN", "TF/TV", "MARCHE", "DEVISE", "INDEX CALC"],
            ["CCY_CODE" , "RATE_CODE"], ["SOUS-BASSIN"]]
    useful_cols = [["ACCRUAL_METHOD", "ACCRUAL_CONVERSION", "TYPE DE COURBE", "ALIAS"],
                   ["COURBE PRICING"], ["CURVE_NAME", "TENOR"], ["SUR-BASSIN"]]
    joinkeys = [False, True, False, False]
    force_int_str = [False, True, False, False]
    upper_content = [True, True, True, True]
    drop_duplicates = [True, True, True, True]
    mappings_full_name = ["MAPPING DES CONVENTIONS DE CALCUL DES COURBES",
                          "MAPPING DES COURBES DE PRICING", "MAPPING INDEX - NOM COURBE & TENOR",
                          "MAPPING BASSINS"]
    est_facultatif = [False, False, False, False]

    for i in range(0, len(name_ranges)):
        mapping_data = ex.get_dataframe_from_range(map_wb, name_ranges[i])
        if len(renames[i]) != 0:
            mapping_data = mapping_data.rename(columns=renames[i])

        rate_input_map[mappings_full_name[i]] = gu.gen_mapping(keys[i], useful_cols[i], mappings_full_name[i], mapping_data,
                              est_facultatif[i], joinkeys[i], force_int_str=force_int_str[i],
                                                               upper_content=upper_content[i], drop_duplicates=drop_duplicates[i])

    return (rate_input_map["MAPPING DES CONVENTIONS DE CALCUL DES COURBES"]["TABLE"],
            rate_input_map["MAPPING DES COURBES DE PRICING"]["TABLE"],
            rate_input_map["MAPPING INDEX - NOM COURBE & TENOR"]["TABLE"],
            rate_input_map["MAPPING BASSINS"]["TABLE"])


def load_rate_input_file_parameters():
    global max_taux_cms, NC_CODE, cols_ri, maturity_to_days
    max_taux_cms = 300
    NC_CODE = "CODE"
    cols_ri = ["M" + str(i) for i in range(0, max_taux_cms + 1)]
    maturity_to_days = mr.maturity_to_days_360