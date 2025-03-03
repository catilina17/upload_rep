class PASS_ALM_Fields():
    """
    Classe permettant de gérer les champs d'entrée, de sortie ainsi que les indicateurs de sortie
    """
    NC_PA_IND03 = "IND03"
    NC_PA_ETAB = "ETAB"
    NC_PA_BASSIN = "BASSIN"
    NC_PA_DEVISE = "DEVISE"
    NC_PA_CONTRACT_TYPE = "CONTRAT"
    NC_PA_MARCHE = "MARCHE"
    NC_PA_RATE_CODE = "RATE CODE"
    NC_PA_PALIER = "PALIER"
    NC_PA_BILAN = "BILAN"
    NC_PA_MATUR = "MATUR"
    NC_PA_INDEX = "INDEX"
    NC_PA_ACCRUAL_BASIS = 'BASE_CALC'
    NC_PA_JR_PN = 'JrPN'
    NC_PA_AMORTIZING_TYPE = 'PROFIL AMOR'
    NC_PA_MATURITY_DURATION = 'DUREE'
    NC_PA_AMORTIZING_PERIODICITY = 'PERIODE AMOR'
    NC_PA_PERIODICITY = 'PERIODE INTERETS'
    NC_PA_FIXING_PERIODICITY = 'PERIODE FIXING'
    NC_PA_COMPOUND_PERIODICITY = 'PERIODE CAPI'
    NC_PA_RELEASING_RULE = "REGLE DEBLOCAGE"

    NC_PA_CLE = 'CLE'
    NC_PA_DIM2 = 'DIM2'
    NC_PA_DIM3 = 'DIM3'
    NC_PA_DIM4 = 'DIM4'
    NC_PA_DIM5 = 'DIM5'
    NC_PA_DIM6 = 'DIM6'
    NC_PA_POSTE = 'POSTE'
    NC_PA_INDEX_AGREG = 'INDEX AGREG'
    NC_PA_GESTION = 'GESTION'
    NC_PA_BOOK = 'BOOK'
    NC_PA_CUST = 'CUST'
    NC_PA_PERIMETRE = 'PERIMETRE'

    NC_PA_Affectation_Social = 'Affectation Social'
    NC_PA_Affectation_Social_2 = 'Affectation Social 2'
    NC_PA_LCR_TIERS_SHARE = 'LCR_TIERS_SHARE'
    NC_PA_DIM_NSFR_1 = 'DIM NSFR 1'
    NC_PA_DIM_NSFR_2 = 'DIM NSFR 2'
    NC_PA_Regroupement_1 = 'Regroupement 1'
    NC_PA_Regroupement_2 = 'Regroupement 2'
    NC_PA_Regroupement_3 = 'Regroupement 3'
    NC_PA_Bilan_Cash = 'Bilan Cash'
    NC_PA_Bilan_Cash_Detail = 'Bilan Cash Detail'
    NC_PA_Bilan_Cash_CTA = 'Bilan Cash CTA'
    NC_PA_METIER = "MÉTIER"
    NC_PA_SOUS_METIER = "SOUS-MÉTIER"
    NC_PA_ZONE_GEO = "ZONE GÉO"
    NC_PA_SOUS_ZONE_GEO = "SOUS-ZONE GÉO"
    NC_PA_TOP_MNI = "TOP_MNI"
    NC_PA_LCR_TIERS = 'LCR TIERS'
    NC_PA_SCOPE = 'SCOPE'

    NC_PA_LEF = "LEF"
    NC_PA_LEM = "LEM"
    NC_PA_LMN = "LMN"
    NC_PA_LMN_EVE = "LMN EVE"
    NC_PA_TEF = "TEF"
    NC_PA_TEM = "TEM"
    NC_PA_DEM_RCO = "DEM"
    NC_PA_DMN_RCO = "DMN"
    NC_PA_DEM = "FlEM"
    NC_PA_DEM_CIBLE = "ECM"
    NC_PA_MG_CO = "MargeCo(bps)"
    NC_PA_TX_SP = "TxSpPN(bps)"
    NC_PA_TX_CIBLE = "TxProdCible(bps)"
    NC_PA_isECH = "IsECH"
    NC_PA_SCENARIO_REF = "scenario REF"

    MAX_MONTHS_ST = 300
    MAX_MONTHS_PN = 60
    ALL_NUM_STOCK = [x for x in range(0, MAX_MONTHS_ST + 1)]

    NC_PA_COL_SORTIE_NUM_ST = ["M" + str(i) for i in range(0, MAX_MONTHS_ST + 1)]

    NC_PA_CLE_OUTPUT = [NC_PA_BASSIN, NC_PA_ETAB, NC_PA_CONTRACT_TYPE, NC_PA_MATUR,
                        NC_PA_DEVISE, NC_PA_RATE_CODE, NC_PA_MARCHE, NC_PA_GESTION,
                        NC_PA_PALIER, NC_PA_BOOK, NC_PA_CUST, NC_PA_PERIMETRE,
                        NC_PA_TOP_MNI, NC_PA_LCR_TIERS, NC_PA_SCOPE]

    NC_PA_COL_SORTIE_QUAL = [NC_PA_CLE, NC_PA_BASSIN, NC_PA_ETAB, NC_PA_BILAN,
                             NC_PA_DIM2, NC_PA_DIM3, NC_PA_DIM4, NC_PA_DIM5,
                             NC_PA_CONTRACT_TYPE, NC_PA_MATUR, NC_PA_DEVISE, NC_PA_RATE_CODE,
                             NC_PA_INDEX_AGREG, NC_PA_MARCHE, NC_PA_GESTION, NC_PA_PALIER,
                             NC_PA_BOOK, NC_PA_CUST, NC_PA_PERIMETRE, NC_PA_POSTE,
                             NC_PA_Affectation_Social, NC_PA_Affectation_Social_2,
                             NC_PA_DIM_NSFR_1, NC_PA_DIM_NSFR_2, NC_PA_Regroupement_1,
                             NC_PA_Regroupement_2, NC_PA_Regroupement_3, NC_PA_Bilan_Cash, NC_PA_Bilan_Cash_Detail,
                             NC_PA_Bilan_Cash_CTA, NC_PA_TOP_MNI, NC_PA_METIER, NC_PA_SOUS_METIER, NC_PA_ZONE_GEO,
                             NC_PA_SOUS_ZONE_GEO, NC_PA_LCR_TIERS, NC_PA_LCR_TIERS_SHARE, NC_PA_SCOPE,
                             NC_PA_INDEX, NC_PA_IND03]

    NC_PA_COL_SORTIE_NUM_PN = ["M" + str(i) for i in range(0, MAX_MONTHS_PN + 1)]
    NC_PA_COL_SORTIE_NUM_PN2 = ["M" + str(i) for i in range(1, MAX_MONTHS_PN + 1)]

    NC_PA_COL_SPEC_ECH = [NC_PA_JR_PN, NC_PA_AMORTIZING_TYPE, NC_PA_MATURITY_DURATION,
                          NC_PA_ACCRUAL_BASIS, NC_PA_AMORTIZING_PERIODICITY, NC_PA_PERIODICITY,
                          NC_PA_FIXING_PERIODICITY, NC_PA_COMPOUND_PERIODICITY, NC_PA_RELEASING_RULE]

    NC_PA_COL_SORTIE_QUAL_ECH = NC_PA_COL_SORTIE_QUAL[:-1] + NC_PA_COL_SPEC_ECH + [NC_PA_IND03]
