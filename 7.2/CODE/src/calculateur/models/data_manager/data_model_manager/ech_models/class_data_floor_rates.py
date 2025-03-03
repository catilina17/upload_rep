from utils import excel_openpyxl as ex
import logging
from calculateur.calc_params.model_params import *

logger = logging.getLogger(__name__)

class Data_RATES_FLOOR():
    def __init__(self, model_wb, name_product):
        self.model_wb = model_wb
        self.name_product = name_product
        self.load_columns()
        self.load_floors()
    def load_columns(self):
        self.NR_FLOOR_FOR_RATES = "_FLOORS_TX_SC"
        self.NC_CONTRAT = "CONTRAT"

    def load_floors(self):
        if self.name_product not in (models_nmd_st + models_nmd_pn):
            floors = ex.get_dataframe_from_range(self.model_wb, self.NR_FLOOR_FOR_RATES)
            self.rates_floors = floors.set_index(self.NC_CONTRAT)
        else:
            self.rates_floors = 0
