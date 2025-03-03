from modules.scenario.parameters import general_parameters as gp


def get_available_entities(codification_etab, all_etabs, scenario_rows):
    if len(scenario_rows['ETABLISSEMENTS'].unique()) > 1:
        raise ValueError(
            "La liste des établissements du scénario:  {},  n'est pas la même dans toute les lignes".format(
                scenario_rows[gp.CN_NOM_SCENARIO]))

    entities_list = get_etabs_from_string_liste(codification_etab, all_etabs, scenario_rows['ETABLISSEMENTS'].iloc[0])
    return entities_list


def get_etabs_from_string_liste(codification_etab, all_etabs, string_liste):
    bassins_liste = [x.strip() for x in string_liste.split(',')]

    for i, field in enumerate(bassins_liste):
        if field in codification_etab.columns and field not in all_etabs:
            etab_liste = list(codification_etab.loc[:, field].dropna())
            bassins_liste[i:i + 1] = etab_liste
    unique_liste = []
    [unique_liste.append(x) for x in bassins_liste if x not in unique_liste and x in all_etabs]
    return unique_liste
