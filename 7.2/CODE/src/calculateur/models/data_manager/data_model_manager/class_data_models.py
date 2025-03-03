from ...data_manager.data_model_manager.ech_models.class_ra_rn_model_params import Data_RA_RN_Model_Params
from ...data_manager.data_model_manager.ech_models.class_douteux_model_params import Data_Douteux_Model_Params
from ...data_manager.data_model_manager.pel_models.class_ra_pel_model_params import Data_RA_PEL_Model_Params
from ...data_manager.data_model_manager.ech_models.class_deblocage_model_params import Data_Deblocage_Model_Params
from ...data_manager.data_model_manager.pel_models.class_data_versements import Data_Versements_PEL
from ...data_manager.data_model_manager.ech_models.class_data_floor_rates import Data_RATES_FLOOR
from ...data_manager.data_model_manager.nmd_models.class_nmd_flow_params import Data_NMD_Model_Params
import logging
from calculateur.calc_params.model_params import *

logger = logging.getLogger(__name__)

class Data_Model_Params():
    def __init__(self, cls_hz_params, cls_fields, name_product, modele_wb):
        self.name_product = name_product
        self.fichier_modele = modele_wb
        self.cls_hz_params = cls_hz_params
        self.cls_fields = cls_fields
        self.create_all_model_types()

    def create_all_model_types(self):
        if self.name_product not in (models_nmd_st + models_nmd_pn):
            self.cls_ra_rn_params = Data_RA_RN_Model_Params(self.fichier_modele, self.cls_hz_params.nb_months_proj)
            self.cls_ra_rn_params.load_rarn_model_params()
            self.cls_versement_params = None
            self.cls_flow_params = None

        elif self.name_product in (models_nmd_st + models_nmd_pn):
            self.cls_ra_rn_params = Data_RA_PEL_Model_Params(self.fichier_modele,
                                                             self.cls_hz_params.nb_months_proj)
            self.cls_ra_rn_params.load_pel_model_params()

            self.cls_versement_params = Data_Versements_PEL(self.fichier_modele, self.cls_fields,
                                                            self.cls_hz_params, self.cls_ra_rn_params)

            self.cls_versement_params.load_versements_model_params()

            self.cls_flow_params = Data_NMD_Model_Params(self.fichier_modele, self.cls_hz_params.dar_usr, self.cls_fields,
                                                         self.cls_hz_params)
            self.cls_flow_params.load_models_params()

        self.cls_debloc_params = Data_Deblocage_Model_Params(self.fichier_modele, self.cls_fields, self.name_product)
        self.cls_debloc_params.load_deblocage_params()

        self.cls_douteux_params = Data_Douteux_Model_Params(self.fichier_modele, self.cls_fields, self.name_product)
        self.cls_douteux_params.load_douteux_model_params()

        self.cls_rates_floor = Data_RATES_FLOOR(self.fichier_modele, self.name_product)

        if self.name_product not in (models_nmd_st + models_nmd_pn):
            try:
                self.fichier_modele.Close(False)
            except:
                pass
            self.fichier_modele = None