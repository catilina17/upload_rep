from utils import excel_openpyxl as ex
import pandas as pd

class Data_Versements_PEL():
    def __init__(self, model_wb, cls_fields, cls_hz_params, cls_ra_model_params):
        self.dar_usr = cls_hz_params.dar_usr
        self.cls_hz_params = cls_hz_params
        self.model_wb = model_wb
        self.cls_fields = cls_fields
        self.nb_mois_proj = cls_hz_params.nb_months_proj
        self.cls_ra_model_params = cls_ra_model_params
        self.mapping_contrats_cle = self.cls_ra_model_params.mapping_contrats_cle
        self.max_pn = cls_ra_model_params.max_pn

    def load_versements_model_params(self):
        self.mtx_vers_dict = {}
        self.max_age_versement = 0
        self.NR_MTX_VERSEMENT_PEL = "_MTX_VERSEMENT_PEL"
        self.NR_MTX_VERSEMENT_PEL_CAT = "_MTX_VERSEMENT_PEL"
        self.NAME_RANGE_TX_STRUCT = ["MTX_VRST_PEL", "MTX_VRST_PELC"]

        self.NC_AGE = "AGE"
        self.NC_CONTRAT = "CONTRAT"
        self.NC_CLE_CC = "CLE"

        """ MATRICES VERSEMENTS et TX STRUCTURELS """
        self.mtx_versements = self.get_versement_matrices(self.mtx_vers_dict, self.model_wb)

    def get_versement_matrices(self, mtx_vers_dict, model_wb):
        names_mtx = self.NAME_RANGE_TX_STRUCT
        names_range = ["_MTX_VERSEMENT_PEL", "_MTX_VERSEMENT_PEL_CAT"]

        for i in range(0, len(names_mtx)):
            mtx_vers_dict[names_mtx[i]] = ex.get_dataframe_from_range(model_wb, names_range[i])
            self.max_age_versement = max(self.max_age_versement , mtx_vers_dict[names_mtx[i]].shape[0])

        mtx_versements_temp = None

        for key, mtx in mtx_vers_dict.items():
            mtx = pd.melt(mtx, id_vars=[self.NC_AGE], value_vars=[x for x in mtx.columns if x != self.NC_AGE],
                          var_name=self.NC_CLE_CC,
                          value_name='VRSMT_STRUCT')
            mtx[self.NC_CONTRAT] = "P-PEL-C" if "PELC" in key else "P-PEL"
            if mtx_versements_temp is None:
                mtx_versements_temp = mtx
            else:
                mtx_versements_temp = pd.concat([mtx_versements_temp, mtx])

        mtx_versements_temp[self.NC_AGE] = [str(int(x)) for x in mtx_versements_temp[self.NC_AGE]]
        mtx_versements_temp.set_index([self.NC_CONTRAT, self.NC_AGE, self.NC_CLE_CC], inplace=True)

        return mtx_versements_temp