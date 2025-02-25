import numpy as np
from utils import excel_utils as ex
import logging
from calculateur.models.utils import utils as ut

logger = logging.getLogger(__name__)

class Data_Douteux_Model_Params():
    def __init__(self, model_wb, cls_fields, name_product):
        self.model_wb = model_wb
        self.cls_fields = cls_fields
        self.profil_douteux = {}
        self.has_douteux = True
        self.name_product = name_product

    def load_douteux_model_params(self):
        self.NR_EVOl_DOUTEUX = "_EVOL_DOUTEUX"
        self.NC_MOD_CTRT = "CONTRACT_TYPE"

        if self.name_product not in ["nmd_st", "nmd_pn"]:
            self.size_ecoul_douteux = 72

            """ EVOL DOUTEUX """
            data_profil_douteux = ex.get_dataframe_from_range(self.model_wb, self.NR_EVOl_DOUTEUX)
            data_profil_douteux = ut.explode_dataframe(data_profil_douteux, columns=[self.NC_MOD_CTRT])

            for contrat in data_profil_douteux[self.NC_MOD_CTRT].unique():
                self.profil_douteux[contrat]\
                    = np.array(data_profil_douteux[data_profil_douteux[self.NC_MOD_CTRT]==contrat].iloc[:, 1:].copy())

        else:
            self.size_ecoul_douteux = 0







