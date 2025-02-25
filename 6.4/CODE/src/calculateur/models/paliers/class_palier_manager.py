import numpy as np
import pandas as pd
import datetime
import logging

logger = logging.getLogger(__name__)


class Palier_Manager():
    def __init__(self, cls_fields, cls_format, cls_cal, nb_months_proj):
        self.cls_fields = cls_fields
        self.cls_format = cls_format
        self.nb_months_proj = nb_months_proj
        self.cls_cal = cls_cal
    ###########@profile
    def prepare_palier_data(self, clas_proj, cls_data_palier, cls_ra_rn_params):

        self.get_palier_schedule(cls_data_palier, clas_proj.data_ldp, clas_proj.cls_cal.mois_depart,
                                 clas_proj.cls_cal.period_begin_date, clas_proj.contracts_updated,
                                 clas_proj.n, clas_proj.t)


    def get_palier_schedule(self, cls_data_palier, data_hab, mois_depart, period_end_date,
                            contracts_updated, n, t):
        data_palier_formated = self.format_data_before_joining_with_palier_mapping(data_hab, mois_depart, period_end_date, n, t)
        dic_palier = self.generate_palier_schedule(data_palier_formated, contracts_updated, cls_data_palier)
        self.dic_palier = dic_palier

    def format_data_before_joining_with_palier_mapping(self, data_hab, mois_depart, period_begin_date, n, t):
        """
        AJOUTER LA NEXT FIXING DATE
        """
        data_ldp_mp = data_hab.copy()
    
        #RENAME DE COLONNES POUR LES PALIERS => FIXING_PERIODICITY, NEXT_FIXING_DATE, RATE_CODE, Mkt_Spread, Num_Spread
        """ 
        SUPPRIME FTP_RATE, FIXING_NEXT_DATE
        RAJOUTER FLOOR et CAP, RATE_TYPE
        """
        data_ldp_mp = data_ldp_mp.rename(columns={self.cls_fields.NC_LDP_CONTRAT: self.cls_fields.NC_CONTRAT_REF_PALIER,
                                                  self.cls_fields.NC_LDP_ECHEANCE_VAL: self.cls_fields.NC_VAL_PALIER,
                                                  self.cls_fields.NC_LDP_RATE: self.cls_fields.NC_RATE_PALIER,
                                                  self.cls_fields.NC_FIXING_PERIODICITY_NUM:self.cls_fields.NC_FIXING_PERIODICITY_NUM_PALIER,
                                                  self.cls_fields.NC_PAL_FIXING_PERIODICITY: self.cls_fields.NC_FIXING_PERIODICITY_PALIER,
                                                  self.cls_fields.NC_LDP_RATE_CODE:self.cls_fields.NC_RATE_CODE_PALIER,
                                                  self.cls_fields.NC_LDP_MKT_SPREAD:self.cls_fields.NC_MKT_SPREAD_PALIER,
                                                  self.cls_fields.NC_LDP_FLOOR_STRIKE:self.cls_fields.NC_FLOOR_STRIKE_PALIER,
                                                  self.cls_fields.NC_LDP_CAP_STRIKE:self.cls_fields.NC_CAP_STRIKE_PALIER,
                                                  self.cls_fields.NC_LDP_FIXING_NEXT_DATE + "_REAL":self.cls_fields.NC_FIXING_NEXT_DATE_PALIER,
                                                  self.cls_fields.NC_LDP_MULT_SPREAD:self.cls_fields.NC_MULT_FACT_PALIER,
                                                  self.cls_fields.NC_LDP_RATE_TYPE: self.cls_fields.NC_RATE_TYPE_PALIER,
                                                  self.cls_fields.NC_FREQ_INT: self.cls_fields.NC_FREQ_INT_NUM_PALIER,
                                                  self.cls_fields.NC_LDP_FREQ_INT: self.cls_fields.NC_FREQ_INT_PALIER,
                                                  self.cls_fields.NC_DECALAGE_AMOR_CAP : self.cls_fields.NC_DECALAGE_AMOR_PALIER,
                                                  self.cls_fields.NC_LDP_TENOR: self.cls_fields.NC_TENOR_PALIER,
                                                  self.cls_fields.NC_LDP_ACCRUAL_BASIS:self.cls_fields.NC_ACCRUAL_BASIS_PALIER,
                                                  self.cls_fields.NC_LDP_CURVE_NAME: self.cls_fields.NC_CURVE_NAME_PALIER,
                                                  self.cls_fields.NC_LDP_FIXING_NB_DAYS: self.cls_fields.FIXING_NB_DAYS_PAL,
                                                  self.cls_fields.NC_LDP_FIXING_RULE : self.cls_fields.FIXING_RULE_PAL})
    
        data_ldp_mp[self.cls_fields.NC_MOIS_PALIER_AMOR] = data_ldp_mp[self.cls_fields.NC_DATE_DEBUT_AMOR].values
        data_ldp_mp[self.cls_fields.NC_MOIS_PALIER_DEBUT] = data_ldp_mp[self.cls_fields.NC_DATE_DEBUT].values
        data_ldp_mp[self.cls_fields.NC_DATE_PALIER_REAL] = period_begin_date[np.arange(0, n).reshape(n, 1),
                                                                     np.minimum(mois_depart - 1, t - 1)]
        data_ldp_mp[self.cls_fields.NC_DATE_PALIER_REAL] = pd.to_datetime(data_ldp_mp[self.cls_fields.NC_DATE_PALIER_REAL])
        data_ldp_mp[self.cls_fields.NC_DATE_PALIER_REAL] =  data_ldp_mp[self.cls_fields.NC_DATE_PALIER_REAL].replace(pd.NaT,self.cls_format.default_date)
        data_ldp_mp[self.cls_fields.NC_MOIS_DATE_PALIER] = self.cls_format.nb_months_in_date_pd(data_ldp_mp[self.cls_fields.NC_DATE_PALIER_REAL])
    
        #RAJOUT ICI
        data_ldp_mp = data_ldp_mp[[self.cls_fields.NC_CONTRAT_REF_PALIER, self.cls_fields.NC_VAL_PALIER,
                                   self.cls_fields.NC_MOIS_DATE_PALIER, self.cls_fields.NC_MOIS_PALIER_AMOR, self.cls_fields.NC_DATE_PALIER_REAL,
                                   self.cls_fields.NC_LDP_MATUR_DATE, self.cls_fields.NC_LDP_CLE,
                                   self.cls_fields.NC_RATE_PALIER, self.cls_fields.NC_FREQ_AMOR,
                                   self.cls_fields.NC_SUSPEND_AMOR_OR_CAPITALIZE_INTERESTS,self.cls_fields.NC_PROFIL_AMOR,
                                   self.cls_fields.NC_FIXING_PERIODICITY_PALIER, self.cls_fields.NC_FIXING_PERIODICITY_NUM_PALIER,
                                   self.cls_fields.NC_RATE_CODE_PALIER, self.cls_fields.NC_MKT_SPREAD_PALIER,
                                   self.cls_fields.NC_FLOOR_STRIKE_PALIER, self.cls_fields.NC_CAP_STRIKE_PALIER,
                                   self.cls_fields.NC_FIXING_NEXT_DATE_PALIER, self.cls_fields.NC_MULT_FACT_PALIER,
                                   self.cls_fields.NC_RATE_TYPE_PALIER, self.cls_fields.NC_FREQ_INT_NUM_PALIER, self.cls_fields.NC_FREQ_INT_PALIER,
                                   self.cls_fields.NC_LDP_FIRST_AMORT_DATE, self.cls_fields.NC_LDP_FIRST_AMORT_DATE_REAL,
                                   self.cls_fields.NC_LDP_MATUR_DATE_REAL, self.cls_fields.NC_LDP_BROKEN_PERIOD,
                                   self.cls_fields.NC_DECALAGE_AMOR_PALIER, self.cls_fields.NC_MOIS_PALIER_DEBUT, self.cls_fields.NC_CAPITALIZE,
                                   self.cls_fields.NC_LDP_CAPITALIZATION_RATE, self.cls_fields.NC_TENOR_PALIER, self.cls_fields.NC_ACCRUAL_BASIS_PALIER,
                                   self.cls_fields.NC_TENOR_NUM, self.cls_fields.NC_CURVE_NAME_PALIER, self.cls_fields.NC_LDP_CURRENCY,
                                   self.cls_fields.NC_LDP_PERFORMING, self.cls_fields.NC_FREQ_CAP,
                                   self.cls_fields.NC_DECALAGE_INT_CAP, self.cls_fields.FIXING_NB_DAYS_PAL,
                                   self.cls_fields.NC_PRODUCT_TYPE, self.cls_fields.NC_LDP_IS_FUTURE_PRODUCT,
                                   self.cls_fields.NC_LDP_ETAB, self.cls_fields.FIXING_RULE_PAL,
                                   self.cls_fields.NC_LDP_TRADE_DATE_REAL, self.cls_fields.NC_LDP_TRADE_DATE,
                                   self.cls_fields.NC_LDP_TARGET_RATE, self.cls_fields.NC_LDP_FTP_FUNDING_SPREAD,
                                   self.cls_fields.NC_DATE_FIN_AMOR, self.cls_fields.NC_DATE_FIN,
                                   self.cls_fields.NC_LDP_PALIER, self.cls_fields.NC_LDP_CONTRACT_TYPE]].copy()
    
        return data_ldp_mp

    def generate_palier_schedule(self, data_hab, contracts_updated, cls_data_palier):
        """"""
        mapping_paliers = cls_data_palier.mapping_paliers
        dic_palier = {}
        """ PARSE DATA WITH PALIER"""
        data_palier = data_hab.copy()
        if len(mapping_paliers) > 0:
            data_palier = data_palier.rename(columns={self.cls_fields.NC_MOIS_DATE_PALIER: self.cls_fields.NC_MOIS_DATE_PALIER + "_FIRST"})
            data_palier = data_palier[[self.cls_fields.NC_LDP_CLE, self.cls_fields.NC_CONTRAT_REF_PALIER, self.cls_fields.NC_LDP_MATUR_DATE,
                                       self.cls_fields.NC_MOIS_DATE_PALIER + "_FIRST", self.cls_fields.NC_LDP_FIRST_AMORT_DATE,
                                       self.cls_fields.NC_LDP_FIRST_AMORT_DATE_REAL, self.cls_fields.NC_LDP_MATUR_DATE_REAL,
                                       self.cls_fields.NC_LDP_BROKEN_PERIOD, self.cls_fields.NC_LDP_CAPITALIZATION_RATE,
                                       self.cls_fields.NC_FREQ_CAP, self.cls_fields.NC_PRODUCT_TYPE]]\
                .join(mapping_paliers, on=self.cls_fields.NC_CONTRAT_REF_PALIER, how="inner")


            cond_mat = pd.to_datetime(data_palier[self.cls_fields.NC_LDP_MATUR_DATE_REAL],format='%d/%m/%Y')\
                       > pd.to_datetime(data_palier[self.cls_fields.NC_DATE_PALIER_REAL],format='%d/%m/%Y')
            data_palier = data_palier[cond_mat].copy()

            if len(data_palier) > 0:
                dar_mois = cls_data_palier.cls_hz_params.dar_mois
                value_date_month = data_palier[self.cls_fields.NC_MOIS_DATE_PALIER].values
                maturity_date_month = data_palier[self.cls_fields.NC_LDP_MATUR_DATE].values
                val_date = data_palier[self.cls_fields.NC_DATE_PALIER_REAL]
                mat_date = data_palier[self.cls_fields.NC_LDP_MATUR_DATE_REAL]
                first_amor_date = data_palier[self.cls_fields.NC_LDP_FIRST_AMORT_DATE].values
                broken_period = data_palier[self.cls_fields.NC_LDP_BROKEN_PERIOD].values
                data_palier[self.cls_fields.NC_MOIS_PALIER_AMOR] = \
                    self.cls_cal.calculate_amortizing_and_interest_payment_begin_month(value_date_month, maturity_date_month, val_date, mat_date,
                                                        first_amor_date, broken_period, dar_mois)[0]

                data_palier[self.cls_fields.NC_MOIS_PALIER_DEBUT] = \
                    self.cls_cal.calculate_amortizing_and_interest_payment_begin_month(value_date_month, maturity_date_month, val_date, mat_date,
                                                        first_amor_date, broken_period, dar_mois, typo="normal")[0]

                data_palier[self.cls_fields.NC_DECALAGE_AMOR_PALIER] \
                    = self.cls_cal.calculate_month_shift_for_non_monthly_amor(data_palier, self.cls_fields.NC_FREQ_AMOR, maturity_date_month,
                                                                     data_palier[self.cls_fields.NC_MOIS_PALIER_AMOR].values)

                data_palier[self.cls_fields.NC_DECALAGE_INT_CAP] \
                    = self.cls_cal.calculate_month_shift_for_non_monthly_amor(data_palier,
                                                                                             self.cls_fields.NC_FREQ_CAP,
                                                                                             maturity_date_month,
                                                                                             data_palier[self.cls_fields.NC_MOIS_PALIER_DEBUT].values)


                data_palier[self.cls_fields.NC_CAPITALIZE] = data_palier[self.cls_fields.NC_SUSPEND_AMOR_OR_CAPITALIZE_INTERESTS]\
                                                & (data_palier[self.cls_fields.NC_LDP_CAPITALIZATION_RATE] != 0) \
                                                & (data_palier[self.cls_fields.NC_FREQ_CAP] != 0)

                if len(contracts_updated) > 0:
                    palier_to_update = data_palier[self.cls_fields.NC_LDP_CLE].isin(contracts_updated.index.values.tolist())
                    cle_to_update = data_palier[palier_to_update][self.cls_fields.NC_LDP_CLE].values.tolist()
                    data_palier.loc[palier_to_update, self.cls_fields.NC_MOIS_DATE_PALIER] = data_palier[palier_to_update][self.cls_fields.NC_MOIS_DATE_PALIER].values +\
                                                                       contracts_updated.loc[cle_to_update]["MONTH_ADJ"].values.astype(int)

                # envisager de créer un palier pour ceux aui capitalisent avec une AMOR DATE supérieure à DAR, quelques contrats seulement

                dic_palier["is_palier"] = data_hab[self.cls_fields.NC_LDP_CLE].isin(data_palier[self.cls_fields.NC_LDP_CLE].unique().tolist())
                data_first_palier = data_hab[dic_palier["is_palier"]].copy()

                """ CONCAT ALL DATA"""
                data_palier = data_palier.drop([self.cls_fields.NC_LDP_MATUR_DATE_REAL, self.cls_fields.NC_LDP_BROKEN_PERIOD, self.cls_fields.NC_LDP_FIRST_AMORT_DATE, self.cls_fields.NC_LDP_MATUR_DATE,
                                                self.cls_fields.NC_MOIS_DATE_PALIER + "_FIRST",self.cls_fields.NC_LDP_CAPITALIZATION_RATE,
                                                self.cls_fields.NC_LDP_FIRST_AMORT_DATE_REAL, self.cls_fields.NC_FREQ_CAP], axis=1).reset_index(drop=True)
                data_first_palier = data_first_palier.drop([self.cls_fields.NC_LDP_MATUR_DATE_REAL, self.cls_fields.NC_LDP_BROKEN_PERIOD,
                                                            self.cls_fields.NC_LDP_FIRST_AMORT_DATE, self.cls_fields.NC_LDP_MATUR_DATE,
                                                            self.cls_fields.NC_LDP_FIRST_AMORT_DATE_REAL, self.cls_fields.NC_LDP_CAPITALIZATION_RATE,
                                                            self.cls_fields.NC_LDP_CURRENCY, self.cls_fields.NC_FREQ_CAP,
                                                            self.cls_fields.NC_LDP_TRADE_DATE_REAL, self.cls_fields.NC_LDP_TRADE_DATE,
                                                            self.cls_fields.NC_LDP_TARGET_RATE, self.cls_fields.NC_LDP_FTP_FUNDING_SPREAD],
                                                           axis=1).reset_index(drop=True)
                data_first_palier["PRIORITY"] = 1
                data_palier["PRIORITY"] = 2

                data_palier = pd.concat([data_first_palier, data_palier]) \
                    .sort_values([self.cls_fields.NC_LDP_CLE, self.cls_fields.NC_CONTRAT_REF_PALIER, "PRIORITY", self.cls_fields.NC_MOIS_DATE_PALIER])

                """ CREATE SCHEDULE"""
                data_palier[self.cls_fields.NC_NB_PALIER] = data_palier[[self.cls_fields.NC_LDP_CLE, self.cls_fields.NC_MOIS_DATE_PALIER]].groupby(
                    data_palier[self.cls_fields.NC_LDP_CLE]).cumcount() + 1

                dic_palier["max_palier"] = data_palier[self.cls_fields.NC_NB_PALIER].max()

                data_palier = data_palier.drop([self.cls_fields.NC_CONTRAT_REF_PALIER], axis=1)
                data_palier[self.cls_fields.NC_VAL_PALIER] = data_palier[self.cls_fields.NC_VAL_PALIER].fillna(-10000)

                palier_schedule = data_palier.set_index([self.cls_fields.NC_LDP_CLE, self.cls_fields.NC_NB_PALIER]).unstack(self.cls_fields.NC_NB_PALIER)

                palier_schedule.columns = [str(col[0]) + str(col[1]) for col in palier_schedule.columns.values]

                palier_schedule = self.format_palier_schedule(data_hab, palier_schedule, dic_palier["max_palier"], dar_mois)

                dic_palier["palier_schedule"] = palier_schedule.sort_index()
            else:
                dic_palier["max_palier"] = 1
                dic_palier["is_palier"] = np.full(data_hab.shape[0], False)

        else:
            dic_palier["max_palier"] = 1
            dic_palier["is_palier"] = np.full(data_hab.shape[0], False)

        return dic_palier

    def format_palier_schedule(self, data_hab, palier_schedule, max_palier, dar_mois):
        n = palier_schedule.shape[0]
        t = self.nb_months_proj
        mois_fin_amor = data_hab.set_index([self.cls_fields.NC_LDP_CLE]).loc[palier_schedule.index][self.cls_fields.NC_DATE_FIN_AMOR].values - dar_mois
        mois_fin_int = data_hab.set_index([self.cls_fields.NC_LDP_CLE]).loc[palier_schedule.index][self.cls_fields.NC_DATE_FIN].values - dar_mois
        palier_schedule = self.format_date_palier(palier_schedule, max_palier, self.cls_fields.NC_MOIS_PALIER_AMOR, dar_mois, n, mois_fin_amor)
        palier_schedule = self.format_date_palier(palier_schedule, max_palier, self.cls_fields.NC_MOIS_PALIER_DEBUT, dar_mois, n, mois_fin_int)

        curr_palier = data_hab.set_index([self.cls_fields.NC_LDP_CLE]).loc[palier_schedule.index][self.cls_fields.NC_LDP_CURRENCY].values
        curr_palier = pd.DataFrame(curr_palier, index=palier_schedule.index, columns=[self.cls_fields.NC_CURRENCY_PALIER])

        future_product_palier = data_hab.set_index([self.cls_fields.NC_LDP_CLE]).loc[palier_schedule.index][self.cls_fields.NC_LDP_IS_FUTURE_PRODUCT].values
        future_product_palier = pd.DataFrame(future_product_palier, index=palier_schedule.index, columns=[self.cls_fields.NC_LDP_IS_FUTURE_PRODUCT])

        etab_palier = data_hab.set_index([self.cls_fields.NC_LDP_CLE]).loc[palier_schedule.index][self.cls_fields.NC_LDP_ETAB].values
        etab_palier = pd.DataFrame(etab_palier, index=palier_schedule.index, columns=[self.cls_fields.NC_LDP_ETAB])

        trade_date_palier = data_hab.set_index([self.cls_fields.NC_LDP_CLE]).loc[palier_schedule.index][self.cls_fields.NC_LDP_TRADE_DATE_REAL].values
        trade_date_palier = pd.DataFrame(trade_date_palier, index=palier_schedule.index, columns=[self.cls_fields.NC_LDP_TRADE_DATE_REAL])

        trade_date_month_palier = data_hab.set_index([self.cls_fields.NC_LDP_CLE]).loc[palier_schedule.index][self.cls_fields.NC_LDP_TRADE_DATE].values
        trade_date_month_palier = pd.DataFrame(trade_date_month_palier, index=palier_schedule.index, columns=[self.cls_fields.NC_LDP_TRADE_DATE])

        funding_spread_palier = data_hab.set_index([self.cls_fields.NC_LDP_CLE]).loc[palier_schedule.index][self.cls_fields.NC_LDP_FTP_FUNDING_SPREAD].values
        funding_spread_palier = pd.DataFrame(funding_spread_palier, index=palier_schedule.index, columns=[self.cls_fields.NC_LDP_FTP_FUNDING_SPREAD])

        target_rate_palier = data_hab.set_index([self.cls_fields.NC_LDP_CLE]).loc[palier_schedule.index][self.cls_fields.NC_LDP_TARGET_RATE].values
        target_rate_palier = pd.DataFrame(target_rate_palier, index=palier_schedule.index, columns=[self.cls_fields.NC_LDP_TARGET_RATE])

        counterparty_code_palier = data_hab.set_index([self.cls_fields.NC_LDP_CLE]).loc[palier_schedule.index][self.cls_fields.NC_LDP_PALIER].values
        counterparty_code_palier = pd.DataFrame(counterparty_code_palier, index=palier_schedule.index, columns=[self.cls_fields.NC_LDP_PALIER])

        contract_type_code_palier = data_hab.set_index([self.cls_fields.NC_LDP_CLE]).loc[palier_schedule.index][self.cls_fields.NC_LDP_CONTRACT_TYPE].values
        contract_type_code_palier = pd.DataFrame(contract_type_code_palier, index=palier_schedule.index, columns=[self.cls_fields.NC_LDP_CONTRACT_TYPE])

        palier_schedule = pd.concat([palier_schedule, curr_palier, future_product_palier, etab_palier, trade_date_palier,
                                     trade_date_month_palier, funding_spread_palier, target_rate_palier,
                                     counterparty_code_palier, contract_type_code_palier], axis=1)


        cols_fix_per = [self.cls_fields.NC_FIXING_PERIODICITY_PALIER + str(pal_nb) for pal_nb in range(1, max_palier + 1)]
        palier_schedule[cols_fix_per] = palier_schedule[cols_fix_per].ffill(axis=1)
    
        cols_fix_per_num = [self.cls_fields.NC_FIXING_PERIODICITY_NUM_PALIER + str(pal_nb) for pal_nb in range(1, max_palier + 1)]
        palier_schedule[cols_fix_per_num] = palier_schedule[cols_fix_per_num].ffill(axis=1)
    
        cols_freq_amor = [self.cls_fields.NC_FREQ_AMOR + str(pal_nb) for pal_nb in range(1, max_palier + 1)]
        palier_schedule[cols_freq_amor] = palier_schedule[cols_freq_amor].ffill(axis=1)
    
        cols_mult_fact = [self.cls_fields.NC_MULT_FACT_PALIER + str(pal_nb) for pal_nb in range(1, max_palier + 1)]
        palier_schedule[cols_mult_fact] = palier_schedule[cols_mult_fact].ffill(axis=1)
    
        cols_freq_palier = [self.cls_fields.NC_FREQ_INT_NUM_PALIER + str(pal_nb) for pal_nb in range(1, max_palier + 1)]
        palier_schedule[cols_freq_palier] = palier_schedule[cols_freq_palier].ffill(axis=1)
    
        cols_rate_code_palier = [self.cls_fields.NC_RATE_CODE_PALIER + str(pal_nb) for pal_nb in range(1, max_palier + 1)]
        palier_schedule[cols_rate_code_palier] = palier_schedule[cols_rate_code_palier].astype(str).ffill(axis=1)
    
        cols_tenor_palier = [self.cls_fields.NC_TENOR_PALIER + str(pal_nb) for pal_nb in range(1, max_palier + 1)]
        palier_schedule[cols_tenor_palier] = palier_schedule[cols_tenor_palier].ffill(axis=1)
    
        cols_curve_name_palier = [self.cls_fields.NC_CURVE_NAME_PALIER + str(pal_nb) for pal_nb in range(1, max_palier + 1)]
        palier_schedule[cols_curve_name_palier] = palier_schedule[cols_curve_name_palier].astype(str).ffill(axis=1)
    
        cols_tenor_num_palier = [self.cls_fields.NC_TENOR_NUM + str(pal_nb) for pal_nb in range(1, max_palier + 1)]
        palier_schedule[cols_tenor_num_palier] = palier_schedule[cols_tenor_num_palier].ffill(axis=1)
    
        cols_accrual_basis_palier = [self.cls_fields.NC_ACCRUAL_BASIS_PALIER + str(pal_nb) for pal_nb in range(1, max_palier + 1)]
        palier_schedule[cols_accrual_basis_palier] = palier_schedule[cols_accrual_basis_palier].ffill(axis=1)
    
        return palier_schedule

    def format_date_palier(self, palier_schedule, max_palier, col_to_format, dar_mois, n, mois_fin):
        cols_date = [col_to_format + str(x) for x in range(1, max_palier + 1)]
        cols_date_palier = np.array(palier_schedule[cols_date]).reshape(n, len(cols_date))
        cols_date_palier = cols_date_palier - dar_mois
        cols_date_palier = np.where(np.isnan(cols_date_palier), mois_fin.reshape(n, 1), cols_date_palier)
        cols_date_palier = cols_date_palier.astype(int)
        palier_schedule[cols_date] = cols_date_palier
        return palier_schedule




