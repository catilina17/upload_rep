""" Paramètres spécifiques des établissements type RZO"""
from modules.alim.parameters.general_parameters import *
from mappings.pass_alm_fields import PASS_ALM_Fields as pa

""" PRESENCE DE PN """
global do_pn

""" LISTE FICHIERS PN"""
pn_rzo_files_name = {}

""" COLS INPUT FILES"""
NC_RZO_DATA_DAR = "DARNUM"
NC_RZO_DATA_RATEC = "RATE_CODE"
NC_RZO_DATA_CCY = "CCY_CODE"
NC_RZO_DATA_MONTANT = "MONTANT"
NC_RZO_DATA_CONTRACT = "CONTRACT_TYPE"
NC_RZO_DATA_MATUR = "MATUR_INI"
NC_RZO_DATA_DIM1 = "CATEGORY_CODE"
NC_RZO_DATA_DATE = "DATEFIN_PERIOD"
NC_RZO_DATA_FAMILY = "FAMILY"
NC_RZO_DATA_PALIER = "PALIER_CONSO"
NC_RZO_DATA_GESTION = "INT_GEST"
NC_RZO_DATA_BOOK =  "BOOK_CODE"
NC_RZO_DATA_ETAB =  "ETAB"
NC_RZO_DATA_COMPONENT_TYPE = "COMPONENT_TYPE"


cols_to_keep = [NC_CONTRACT_TEMP, pa.NC_PA_RATE_CODE, pa.NC_PA_DEVISE, \
                pa.NC_PA_MARCHE, NC_GESTION_TEMP, NC_PALIER_TEMP, \
                pa.NC_PA_BILAN, NC_DATE_MONTANT, NC_MATUR_TEMP, pa.NC_PA_BOOK, pa.NC_PA_ETAB]

agreg_vars = [x for x in cols_to_keep if x != NC_DATE_MONTANT]

