from modules.alim.parameters.general_parameters import *
from mappings.pass_alm_fields import PASS_ALM_Fields as pa

""" PARAM NTX"""
global perim_ntx
global del_enc_dar_zero
global depart_decale

""" DATA NTX """
NC_NTX_DATA_CCY = "CCY_CODE"
NC_NTX_DATA_RATEC = "RATE_CODE"
NC_NTX_DATA_FAMILY = "FAMILY"
NC_NTX_DATA_FLAG_MNI = "FLAG_MNI"
NC_NTX_DATA_SWAP_TAG = "SWAP_TAG"
NC_NTX_DATA_FLAG_LIQ_GAP = "FLAG_LIQ_GAP"
NC_NTX_DATA_ID = "IDENT"
NC_NTX_DATA_PALIER = "PALIER_CONSO"
NC_NTX_DATA_METIER = "METIER"
NC_NTX_DATA_BALE = "BALE"
NC_NTX_DATA_DEPARTD = "DEPART_DECALE"
NC_NTX_DATA_DAR = "DAR"
NC_NTX_DATA_CONTRAT = "CONTRACT_TYPE"
NC_NTX_DATA_GESTION = "INTENTIONS_GESTION"
NC_NTX_DATA_MATUR = "MATURITE_INITIALE"
NC_NTX_DATA_DIM1 = "DIM1"
NC_NTX_DATA_TYPE_EMETTEUR = "TYPE_EMETTEUR"


cols_to_keep = [NC_CONTRACT_TEMP, pa.NC_PA_RATE_CODE, pa.NC_PA_DEVISE, \
                pa.NC_PA_MARCHE, NC_GESTION_TEMP, NC_PALIER_TEMP, \
                NC_MATUR_TEMP, pa.NC_PA_TOP_MNI, NC_NTX_DATA_DEPARTD, pa.NC_PA_BOOK,
                NC_NTX_DATA_FLAG_LIQ_GAP, pa.NC_PA_BILAN, pa.NC_PA_LCR_TIERS,\
                pa.NC_PA_PERIMETRE, pa.NC_PA_CUST, NC_NTX_DATA_TYPE_EMETTEUR]

agreg_vars = [x for x in cols_to_keep if x not in [NC_NTX_DATA_DEPARTD,NC_NTX_DATA_TYPE_EMETTEUR]] + [pa.NC_PA_SCOPE]

lcr_tiers_actif_change_contract=["A-CC-CLI-KN", "A-CPT-DEB-KN", "A-CR-TRES-KN", "A-CR-EXP-KN",\
                                 "A-CR-DOC-KN", "A-CR-EQ-KN", "A-CR-REV-KN", "A-EFT-COM-KN", "A-PR-IBQ-KN"]
contract_change_actif="A-CPT-BC-KN"
lcr_tiers_passif_change_contract=["P-CAT-STD-KN", "P-CC-CLI-KN", "P-CPT-CRD-KN", "P-EMP-IBQ-KN", "P-LIV-ORD-KN"]
contract_change_passif="P-EMP-BC-KN"

""" MAPPING COLS"""
NC_MAP_NTX_TRAITEMENT_REG_ALM = "Traitement réglementaire défaut ALM"