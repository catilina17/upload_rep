from utils import excel_utils as ex, general_utils as gu

NC_CONTRAT = "CONTRAT"
NC_POSTE = "POSTE"
NC_PALIER = "PALIER"
NC_DEVISE = "DEVISE"
NC_INDEX_CALC = "INDEX CALC"



def get_mapping_from_wb(map_wb, name_range):
    mapping_data = ex.get_dataframe_from_range(map_wb, name_range)
    return mapping_data

def rename_cols_mapping(mapping_data, rename):
    mapping_data = mapping_data.rename(columns=rename)
    return mapping_data

def gen_mapping(keys, useful_cols, mapping_full_name, mapping_data, est_facultatif, joinkey, drop_duplicates=False,
                force_int_str=False):
    mapping = {}
    mapping_data = gu.strip_and_upper(mapping_data, keys)

    if len(keys)>0:
        mapping_data = mapping_data.drop_duplicates(subset=keys).copy()

    if force_int_str:
        for col in keys + useful_cols:
            mapping_data = gu.force_integer_to_string(mapping_data, col)

    if joinkey:
        mapping_data["KEY"] = mapping_data[keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        keys = ["KEY"]

    if len(keys)>0:
        mapping["TABLE"] = mapping_data.set_index(keys)
    else:
        mapping["TABLE"] = mapping_data

    mapping["OUT"] = useful_cols
    mapping["FULL_NAME"] = mapping_full_name

    mapping["est_facultatif"] = est_facultatif

    return mapping
def lecture_mapping_pass_alm(map_wb):
    global exceptions_missing_mappings
    general_mapping = {}

    mappings_name = ["CONTRATS","PALIER"]
    name_ranges = ["_MAP_GENERAL", "_MapPalier"]
    renames = [{"CONTRAT": "CONTRAT_INIT", "CONTRAT PASS": NC_CONTRAT, "POSTE AGREG": NC_POSTE}, \
               {"MAPPING": NC_PALIER}]
    keys = [["CATEGORY", "CONTRAT_INIT"],["PALIER CONSO"]]
    useful_cols = [[NC_CONTRAT], [NC_PALIER]]
    mappings_full_name = ["MAPPING CONTRATS PASSALM", "MAPPING CONTREPARTIES"]
    est_facultatif = [False, False]

    joinkeys = [False] * len(keys)

    force_int_str = [False] + [True]

    for i in range(0, len(mappings_name)):
        mapping_data = get_mapping_from_wb(map_wb, name_ranges[i])
        if len(renames[i]) != 0:
            mapping_data = rename_cols_mapping(mapping_data, renames[i])

        mapping = gen_mapping(keys[i], useful_cols[i], mappings_full_name[i], mapping_data, \
                              est_facultatif[i], joinkeys[i], force_int_str=force_int_str[i])
        general_mapping[mappings_name[i]] = mapping

    return general_mapping

