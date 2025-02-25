import numpy as np
import pandas as pd
import datetime
import logging

logger = logging.getLogger(__name__)


class Data_Palier_Manager():
    """
    Formate les données
    """
    def __init__(self, cls_format, cls_fields, cls_hz_params, data_palier):
        self.cls_format = cls_format
        self.cls_fields = cls_fields
        self.cls_hz_params = cls_hz_params
        self.mapping_paliers = []
        self.data_palier = data_palier.copy()

    def format_pal_file(self):
        data_palier = self.data_palier.copy()
        if len(data_palier) >0 :
            logging.debug('    Lecture du fichier PAL')
            data_palier = self.cls_format.rename_cols(data_palier)
            data_palier = self.cls_format.upper_columns_names(data_palier)#DATA_FORMATER

            data_palier = self.cls_format.create_unvailable_variables(data_palier, self.cls_fields.pal_vars,
                                                                      self.cls_fields.default_pal_vars, type="pal")

            data_palier = self.cls_format.parse_date_list(data_palier, [self.cls_fields.NC_PAL_BEGIN_DATE,
                                                                          self.cls_fields.NC_PAL_FIXING_NEXT_DATE],
                                                           in_month=[True, True])#DATA_FOMRATER


            data_palier = self.cls_format.format_ldp_num_cols(data_palier, num_cols=[self.cls_fields.NC_PAL_RATE,
                                                                                     self.cls_fields.NC_PAL_MKT_SPREAD,
                                                                                     self.cls_fields.NC_PAL_CAP_STRIKE,
                                                                                     self.cls_fields.NC_PAL_FLOOR_STRIKE,
                                                                                     self.cls_fields.NC_PAL_MULT_SPREAD,
                                                                                     self.cls_fields.NC_PAL_FIXING_NB_DAYS],
                                                              divider=[1200, 10000, 100, 100, 1, 1, 10000 * 12],
                                                              na_fills=[-100 * 1200, 0, 100, 0, 1, 0, np.nan])#DATA_FOMRATER

            data_palier = self.cls_format.generate_freq_int(data_palier, self.cls_fields.NC_PAL_FREQ_INT)#DATA_FOMRATER
            data_palier = self.cls_format.get_supsension_or_interets_capitalization_var(data_palier, self.cls_fields.NC_PAL_FREQ_INT, self.cls_hz_params.dar_mois,
                                                                            capi_rate_col = self.cls_fields.NC_PAL_CAPITALIZATION_RATE,
                                                                            type_data="pal")
            data_palier = self.cls_format.generate_freq_amor(data_palier, self.cls_fields.NC_PAL_FREQ_AMOR,  self.cls_fields.NC_PAL_FREQ_INT)#DATA_FOMRATER #PROJECTION
            data_palier = self.cls_format.generate_amortissement_profil(data_palier, self.cls_fields.NC_LDP_TYPE_AMOR)#DATA_FOMRATER
            data_palier = self.cls_format.generate_freq_fixing_periodicity(data_palier,
                                                                                     self.cls_fields.NC_PAL_FIXING_PERIODICITY,
                                                                                     self.cls_fields.NC_PAL_FREQ_INT)#DATA_FOMRATER
            data_palier = self.cls_format.generate_freq_curve_tenor(data_palier, self.cls_fields.NC_PAL_TENOR,
                                                                              self.cls_fields.NC_LDP_FIXING_PERIODICITY)

            data_palier = self.cls_format.generate_freq_curve_name(data_palier, self.cls_fields.NC_PAL_CURVE_NAME,
                                                                    self.cls_fields.NC_PAL_RATE_TYPE)


            data_palier.rename(columns={self.cls_fields.NC_PAL_CONTRAT: self.cls_fields.NC_CONTRAT_REF_PALIER,
                                         self.cls_fields.NC_PAL_BEGIN_DATE: self.cls_fields.NC_MOIS_DATE_PALIER,
                                         self.cls_fields.NC_PAL_BEGIN_DATE + "_REAL": self.cls_fields.NC_DATE_PALIER_REAL,
                                         self.cls_fields.NC_PAL_VAL_EUR: self.cls_fields.NC_VAL_PALIER,
                                         self.cls_fields.NC_PAL_RATE: self.cls_fields.NC_RATE_PALIER,
                                         self.cls_fields.NC_PAL_MKT_SPREAD: self.cls_fields.NC_MKT_SPREAD_PALIER,
                                         self.cls_fields.NC_PAL_FLOOR_STRIKE: self.cls_fields.NC_FLOOR_STRIKE_PALIER,
                                         self.cls_fields.NC_PAL_CAP_STRIKE: self.cls_fields.NC_CAP_STRIKE_PALIER,
                                         self.cls_fields.NC_PAL_MULT_SPREAD: self.cls_fields.NC_MULT_FACT_PALIER,
                                         self.cls_fields.NC_FIXING_PERIODICITY_NUM:self.cls_fields.NC_FIXING_PERIODICITY_NUM_PALIER,
                                         self.cls_fields.NC_PAL_FIXING_NEXT_DATE + "_REAL": self.cls_fields.NC_FIXING_NEXT_DATE_PALIER,
                                         self.cls_fields.NC_PAL_RATE_CODE: self.cls_fields.NC_RATE_CODE_PALIER,
                                         self.cls_fields.NC_PAL_RATE_TYPE: self.cls_fields.NC_RATE_TYPE_PALIER,
                                         self.cls_fields.NC_PAL_FREQ_INT : self.cls_fields.NC_FREQ_INT_PALIER,
                                         self.cls_fields.NC_FREQ_INT: self.cls_fields.NC_FREQ_INT_NUM_PALIER, self.cls_fields.NC_PAL_TENOR:self.cls_fields.NC_TENOR_PALIER,
                                         self.cls_fields.NC_LDP_ACCRUAL_BASIS: self.cls_fields.NC_ACCRUAL_BASIS_PALIER,
                                         self.cls_fields.NC_PAL_CURVE_NAME: self.cls_fields.NC_CURVE_NAME_PALIER,
                                         self.cls_fields.NC_PAL_FIXING_PERIODICITY: self.cls_fields.NC_FIXING_PERIODICITY_PALIER,
                                         self.cls_fields.NC_PAL_FIXING_NB_DAYS: self.cls_fields.FIXING_NB_DAYS_PAL,
                                         self.cls_fields.NC_PAL_FIXING_RULE: self.cls_fields.FIXING_RULE_PAL},
                                         inplace=True)#DATA_FOMRATER

            data_palier = data_palier.set_index(self.cls_fields.NC_CONTRAT_REF_PALIER)
            """ AJOUT DE COLONNES ICI ET POTENTIEL FORMATAGE, 
            => FIXING_PERIODICITY, NEXT_FIXING_DATE, RATE_CODE, Mkt_Spread, Num_Spread
            """
            self.mapping_paliers = data_palier[[self.cls_fields.NC_MOIS_DATE_PALIER, self.cls_fields.NC_DATE_PALIER_REAL, self.cls_fields.NC_VAL_PALIER, self.cls_fields.NC_RATE_PALIER,
                                        self.cls_fields.NC_SUSPEND_AMOR_OR_CAPITALIZE_INTERESTS,
                                        self.cls_fields.NC_FREQ_AMOR, self.cls_fields.NC_PROFIL_AMOR, self.cls_fields.NC_MKT_SPREAD_PALIER,
                                        self.cls_fields.NC_MULT_FACT_PALIER, self.cls_fields.NC_FLOOR_STRIKE_PALIER,
                                        self.cls_fields.NC_CAP_STRIKE_PALIER, self.cls_fields.NC_FIXING_PERIODICITY_PALIER,
                                        self.cls_fields.NC_FIXING_NEXT_DATE_PALIER, self.cls_fields.NC_RATE_CODE_PALIER,
                                        self.cls_fields.NC_RATE_TYPE_PALIER, self.cls_fields.NC_FREQ_INT_PALIER, self.cls_fields.NC_FREQ_INT_NUM_PALIER,
                                        self.cls_fields.NC_TENOR_PALIER, self.cls_fields.NC_TENOR_NUM, self.cls_fields.NC_CURVE_NAME_PALIER,
                                        self.cls_fields.NC_FIXING_PERIODICITY_NUM_PALIER, self.cls_fields.NC_ACCRUAL_BASIS_PALIER,
                                                self.cls_fields.FIXING_NB_DAYS_PAL, self.cls_fields.FIXING_RULE_PAL]].copy()
        else:
            self.mapping_paliers = pd.DataFrame([], columns=[self.cls_fields.NC_MOIS_DATE_PALIER, self.cls_fields.NC_DATE_PALIER_REAL, self.cls_fields.NC_VAL_PALIER, self.cls_fields.NC_RATE_PALIER,
                                        self.cls_fields.NC_SUSPEND_AMOR_OR_CAPITALIZE_INTERESTS,
                                        self.cls_fields.NC_FREQ_AMOR, self.cls_fields.NC_PROFIL_AMOR, self.cls_fields.NC_MKT_SPREAD_PALIER,
                                        self.cls_fields.NC_MULT_FACT_PALIER, self.cls_fields.NC_FLOOR_STRIKE_PALIER,
                                        self.cls_fields.NC_CAP_STRIKE_PALIER, self.cls_fields.NC_FIXING_PERIODICITY_PALIER,
                                        self.cls_fields.NC_FIXING_NEXT_DATE_PALIER, self.cls_fields.NC_RATE_CODE_PALIER,
                                        self.cls_fields.NC_RATE_TYPE_PALIER, self.cls_fields.NC_FREQ_INT_PALIER, self.cls_fields.NC_FREQ_INT_NUM_PALIER,
                                        self.cls_fields.NC_TENOR_PALIER, self.cls_fields.NC_TENOR_NUM, self.cls_fields.NC_CURVE_NAME_PALIER,
                                        self.cls_fields.NC_FIXING_PERIODICITY_NUM_PALIER, self.cls_fields.NC_ACCRUAL_BASIS_PALIER,
                                                             self.cls_fields.FIXING_NB_DAYS_PAL, self.cls_fields.FIXING_RULE_PAL])

