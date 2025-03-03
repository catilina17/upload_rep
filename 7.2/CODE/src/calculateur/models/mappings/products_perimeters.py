
class Perimeters():
    perimetre_pr_per = ["A-PR-PERSO", "AHB-NS-PR-PER"]
    perimetre_ptz = ["A-PTZ", "A-PTZ+", "AHB-NS-CR-PTZ"]
    perimetre_cr_eq = ["A-CR-EQ-MUL", "A-CR-EQ-AIDE", "A-CR-EQ-STR", "A-CR-EQ-STD", "A-CR-EQ-CPLX", "AHB-NS-CR-EQ",
                       "AHB-NS-CR-EQA"]
    perimetre_cap_floor = ["AHB-CAP", "PHB-CAP", "AHB-FLOOR", "PHB-FLOOR"]
    perimetre_habitat = ["A-CR-HAB-LIS", "A-CR-HAB-STD", "A-CR-HAB-MOD", "A-CR-HAB-AJU", "A-PR-STARDEN",
                         "AHB-NS-CR-HAB"]
    perimetre_habitat_casden = ["A-CR-HAB-BON", "AHB-NS-CR-HBN"]
    perimetre_PEL = ["P-PEL", "P-PEL-C"]
    perimetre_P_CAT = ["P-CAT-STD", "P-CAT-CORP", "P-CAT-MIXTE", "P-CAT-PELP", "P-CAT-VIE", "P-CAT-PROG"]
    perimetre_A_INTBQ = ["A-PR-INTBQ"]
    perimetre_P_INTBQ = ["P-EMP-INTBQ"]
    perimetre_A_AUTRES_TF = ["A-CR-AV", "A-EFT-CRECOM", "A-PR-PATRI", "A-CR-REL-HAB", "A-CR-TRESO",
                             "A-CR-BAIL", "A-CR-HAB-BON", "A-CR-LBO", "A-PR-CEL", "A-LIGNE-TRES",
                             "A-PR-PEL", "A-PR-LDD"]
    perimetre_habitat_not_casden = ["A-CR-HAB-BON"]

    perimetre_ptz_rco = ["A-PTZ", "A-PTZ+", "HB-NS-CR-PTZ"]
    perimetre_cap_floor_rco = ["HB-CAP", "HB-FLOOR"]

    perimetre_swap = ["HB-SW-SIMPLE", "HB-SW-ASS", "HB-CHANGE-SW", "HB-SW-STRUCT", "HB-SW-DEV"]

    perimetre_titres = ['A-OBLIG-2','A-OBLIG-1','A-OBLIG-3','A-OBLIG-4',
                        'A-T-SUB-DD','A-OBLIG-6','A-OBLIG-5','A-T-SUB-DI',
                        'P-TIT-OBLIG', "P-TCN-FI", "A-ACTION", "P-EMP-OBL"]

    perimetre_change = ["HB-CHANGE-SW", "HB-CHANGE"]

    produits_marche = perimetre_swap + perimetre_cap_floor_rco + perimetre_titres + perimetre_change
