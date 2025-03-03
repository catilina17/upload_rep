import numexpr as ne
import numpy as np
import pandas as pd
from calculateur.models.run_offs.ech_amortization.amortissement_scenario.class_amortization_renegociated import Amortization_Renegociated
from calculateur.models.utils import utils as ut

try:
    import cupy as cnp
    is_cupy = True
except:
    import numpy as cnp
    is_cupy = False


class Immo_Renegotiation():
    """
    Load les données utilisateurs
    """
    def __init__(self, cls_stat_amor, cls_static_ind, cls_model_params):
        self.cls_amor_reneg = Amortization_Renegociated(cls_stat_amor)
        self.cls_proj = cls_stat_amor.cls_proj
        self.cls_cal = cls_stat_amor.cls_cal
        self.cls_palier = cls_stat_amor.cls_proj.cls_palier
        self.cls_fields = cls_stat_amor.cls_fields
        self.cls_static_ind = cls_static_ind
        self.cls_model_params = cls_model_params
        self.cls_ra_rn_params = cls_model_params.cls_ra_rn_params
        self.load_cupy()

    def get_reneg_indics(self, cls_static_ind, data_hab, remaining_capital, cls_rarn, mois_depart_amor, mois_fin_amor,
                         drac_amor, current_month, mois_depart, interests_calc_periods, dar_mois, douteux, dic_palier,
                         n, t):

        (self.reneg_leg_capital, self.avg_reneg_leg_capital,
         self.reneg_leg_mni, self.reneg_leg_ftp_mni, self.reneg_rate) = (np.zeros((n, t)).copy(),
                                                                          np.zeros((n, t)).copy(),
                                                                          np.zeros((n, t)).copy(),
                                                                          np.zeros((n, t)).copy(), np.zeros((n, t)).copy())

        if (cls_static_ind.cls_proj.name_product not in ['nmd_st', "nmd_pn"]):
            is_reneg = (~np.isnan(cls_rarn.rate_renego_immo)).any(axis=1)
            _n = is_reneg[is_reneg].shape[0]
            if _n > 0:
                is_palier = dic_palier["is_palier"][is_reneg].copy()
                reneg_leg_capital, avg_reneg_leg_capital, reneg_leg_ir, reneg_leg_ftp_ir, reneg_rate \
                    = self.get_reneg_capital_and_mni(data_hab[is_reneg].copy(), remaining_capital[is_reneg],
                                                              cls_static_ind.effet_rarn_cum[is_reneg],
                                                              cls_rarn.tx_rn[is_reneg], cls_rarn.rate_renego_immo[is_reneg],
                                                              mois_depart_amor[is_reneg], mois_fin_amor[is_reneg],
                                                              drac_amor[is_reneg], current_month[is_reneg],
                                                              mois_depart[is_reneg],
                                                              interests_calc_periods[is_reneg],
                                                              dar_mois, douteux[is_reneg], is_palier,
                                                              dic_palier, _n, t)

                self.reneg_leg_capital[is_reneg] = reneg_leg_capital.copy()
                self.avg_reneg_leg_capital[is_reneg] = avg_reneg_leg_capital.copy()
                self.reneg_leg_mni[is_reneg] = reneg_leg_ir.copy()
                self.reneg_leg_ftp_mni[is_reneg] = reneg_leg_ftp_ir.copy()
                self.reneg_rate[is_reneg] = reneg_rate.copy()

    ############@profile
    def get_reneg_capital_and_mni(self, data_hab, remaining_capital, rarn_effect, tx_rn, rate_renego, mois_depart_amor,
                                         mois_fin_amor, drac_amor, current_month, mois_depart,
                                         interests_calc_periods, dar_mois, douteux, is_palier, dic_palier, n, t):

        amount_renegociated = self.calculate_renegociated_amount(remaining_capital, rarn_effect, tx_rn, current_month,
                                                                 mois_depart)

        reneg_leg_capital = np.zeros((n, t))
        avg_reneg_leg_capital = np.zeros((n, t))
        reneg_leg_ir = np.zeros((n, t))
        reneg_leg_ftp_ir = np.zeros((n, t))
        reneg_rate = np.zeros((n, t))

        rate_renego = np.nan_to_num(rate_renego)

        data_hab = self.force_to_infine_if_capitalization_without_palier_or_linear_ech(data_hab, is_palier)

        is_infine = np.array(data_hab[self.cls_fields.NC_PROFIL_AMOR] == "INFINE")

        mois_depart_amor, drac_amor =\
            self.get_amor_deb_for_palier_with_capi(data_hab, is_palier, dic_palier, dar_mois, mois_depart_amor, drac_amor, n, t)

        if amount_renegociated.sum() == 0:
            t_l = 0
        else:
            t_l = int(np.where(amount_renegociated > 0, current_month, 0).max())

        filter_reneg = amount_renegociated.sum(axis=1) != 0
        n_r = filter_reneg[filter_reneg].shape[0]

        if t_l > 0:
            self.amount_renegociated = cnp.asarray(amount_renegociated[filter_reneg].copy())
            self.mois_depart_amor = cnp.asarray(mois_depart_amor[filter_reneg].copy())
            self.drac_amor = cnp.asarray(drac_amor[filter_reneg].copy())
            self.is_infine = cnp.asarray(is_infine[filter_reneg].copy())
            self.reneg_leg_capital_nr = cnp.zeros((n_r, t))
            self.rate_reneg_ag_nr = cnp.zeros((n_r, t))
            self.avg_reneg_leg_capital_nr = cnp.zeros((n_r, t))
            self.reneg_leg_ir_nr = cnp.zeros((n_r, t))
            self.reneg_leg_ftp_ir_nr = cnp.zeros((n_r, t))
            self.rate_renego =  cnp.asarray(rate_renego[filter_reneg].copy())
            self.douteux =  cnp.asarray(douteux[filter_reneg])
            self.mois_fin_amor = cnp.asarray(mois_fin_amor[filter_reneg].copy())
            self.current_month = cnp.asarray(current_month[filter_reneg].copy())
            self.is_base_calc_30 = cnp.asarray(data_hab.loc[filter_reneg, self.cls_fields.NC_LDP_ACCRUAL_BASIS].isin(["30/360", "30E/360","30A/360"]).values)
            self.nb_days = cnp.asarray(data_hab.loc[filter_reneg,self.cls_fields.NB_DAYS_AN].values)
            self.mat_date = cnp.asarray(data_hab.loc[filter_reneg,self.cls_fields.NC_LDP_MATUR_DATE].values)
            self.prof_infine = cnp.array((data_hab.loc[filter_reneg,self.cls_fields.NC_PROFIL_AMOR] == "INFINE").values)
            self.prof_amor = cnp.array((data_hab.loc[filter_reneg,self.cls_fields.NC_PROFIL_AMOR] == "ECHCONST").values)
            mat_date_real = data_hab.loc[filter_reneg,self.cls_fields.NC_LDP_MATUR_DATE + "_REAL"]
            self.day_mat_date = cnp.array(pd.to_datetime(mat_date_real).dt.day - 1).astype(int)
            self.delta_days = cnp.array(self.cls_cal.delta_days)
            self.ftp_rate = cnp.asarray(data_hab.loc[filter_reneg, self.cls_fields.NC_LDP_FTP_RATE].values)

            c = 250
            for is_infine_prof in [True, False]:
                for k in range(0, n_r // c + 1):
                    self.perf_c = ~self.douteux[c * k:c * (k + 1)].copy()
                    self.infine_c = self.is_infine[c * k:c * (k + 1)].copy() if is_infine_prof else ~self.is_infine[c * k:c * (k + 1)].copy()
                    self.amount_reneg_c = self.amount_renegociated[c * k:c * (k + 1)][self.perf_c & self.infine_c].copy()
                    self.current_month_c = self.current_month[c * k:c * (k + 1)][self.perf_c & self.infine_c].copy()
                    if self.amount_reneg_c.sum() == 0:
                        t_l_c = 0
                    else:
                        t_l_c = int(cnp.where(self.amount_reneg_c > 0, self.current_month_c, 0).max())
                    if len(self.amount_reneg_c) > 0 and t_l_c > 0:
                        self.rate_renego_c = self.rate_renego[c * k:c * (k + 1)][self.perf_c & self.infine_c].copy()
                        self.mois_depart_amor_c = self.mois_depart_amor[c * k:c * (k + 1)][self.perf_c & self.infine_c].copy()
                        self.mois_fin_amor_c = self.mois_fin_amor[c * k:c * (k + 1)][self.perf_c & self.infine_c].copy()
                        self.drac_amor_c = self.drac_amor[c * k:c * (k + 1)][self.perf_c & self.infine_c].copy()
                        self.is_base_calc_30_c = self.is_base_calc_30[c * k:c * (k + 1)][self.perf_c & self.infine_c].copy()
                        self.nb_days_c = self.nb_days[c * k:c * (k + 1)][self.perf_c & self.infine_c].copy()
                        self.mat_date_c = self.mat_date[c * k:c * (k + 1)][self.perf_c & self.infine_c].copy()
                        self.prof_amor_c = self.prof_amor[c * k:c * (k + 1)][self.perf_c & self.infine_c].copy()
                        self.prof_infine_c = self.prof_infine[c * k:c * (k + 1)][self.perf_c & self.infine_c].copy()
                        self.day_mat_date_c = self.day_mat_date[c * k:c * (k + 1)][self.perf_c & self.infine_c].copy()
                        self.ftp_rate_c = self.ftp_rate[c * k:c * (k + 1)][self.perf_c & self.infine_c].copy()
                        n_k = self.amount_reneg_c.shape[0]

                        self.reneg_leg_capital_tmp, rate_with_spread = \
                            self.get_renegociated_capital_simplified(self.prof_amor_c, self.prof_infine_c, self.mat_date_c,
                                                                     self.amount_reneg_c, self.rate_renego_c, self.current_month_c,
                                                                   self.mois_depart_amor_c, self.mois_fin_amor_c, self.drac_amor_c,
                                                                   dar_mois, n_k, t, t_l_c, is_infine=is_infine_prof)

                        self.avg_renego_leg_capital_tmp = \
                            self.generate_avg_renego_capital_simplified(self.delta_days, self.day_mat_date_c,
                                                                        self.reneg_leg_capital_tmp, n_k, t) #INDICATORS_RENGEG

                        self.reneg_leg_mni_tmp, self.reneg_leg_ftp_mni_tmp = \
                            self.calculate_renego_mni_simplified(self.delta_days, self.nb_days_c, self.is_base_calc_30_c,
                                                                 self.avg_renego_leg_capital_tmp, rate_with_spread,
                                                                 self.ftp_rate_c, n_k, t, t_l_c)#INDICATORS_RENGEG

                        self.reneg_leg_capital_ag, self.avg_reneg_leg_capital_ag, self.reneg_leg_ir_ag, self.reneg_leg_ftp_ir_ag,\
                            self.rate_reneg_ag = \
                            self.calculate_simplified_agregated_indics(self.reneg_leg_capital_tmp, self.avg_renego_leg_capital_tmp,
                                                                     self.reneg_leg_mni_tmp, self.reneg_leg_ftp_mni_tmp,
                                                                       rate_with_spread, self.amount_reneg_c, n_k, t_l_c, t)#INDICATORS_RENGEG

                        self.reneg_leg_capital_nr[c * k:c * (k + 1)][self.perf_c & self.infine_c] = self.reneg_leg_capital_ag
                        self.avg_reneg_leg_capital_nr[c * k:c * (k + 1)][self.perf_c & self.infine_c] = self.avg_reneg_leg_capital_ag
                        self.reneg_leg_ir_nr[c * k:c * (k + 1)][self.perf_c & self.infine_c] = self.reneg_leg_ir_ag
                        self.reneg_leg_ftp_ir_nr[c * k:c * (k + 1)][self.perf_c & self.infine_c] = self.reneg_leg_ftp_ir_ag
                        self.rate_reneg_ag_nr[c * k:c * (k + 1)][self.perf_c & self.infine_c] = self.rate_reneg_ag

            #print(mempool.used_bytes())
            #print(mempool.total_bytes())

            self.clean_reneg_vars()

            reneg_leg_capital[filter_reneg] = self.reneg_leg_capital_nr
            avg_reneg_leg_capital[filter_reneg] = self.avg_reneg_leg_capital_nr
            reneg_leg_ir[filter_reneg]  = self.reneg_leg_ir_nr
            reneg_leg_ftp_ir[filter_reneg]  = self.reneg_leg_ftp_ir_nr
            reneg_rate[filter_reneg] = self.rate_reneg_ag_nr

        return (reneg_leg_capital.copy(), avg_reneg_leg_capital.copy(), reneg_leg_ir.copy(), reneg_leg_ftp_ir.copy(),
                reneg_rate.copy())



    def load_cupy(self):
        self.is_cupy = is_cupy
        if self.is_cupy:
            self.mempool = cnp.get_default_memory_pool()
            self.pinned_mempool = cnp.get_default_pinned_memory_pool()


    def clean_reneg_vars(self):
        if self.is_cupy:
            self.rate_with_spread = None
            self.reneg_leg_ir_ag = None
            self.reneg_leg_ftp_ir_ag = None
            self.avg_reneg_leg_capital_ag = None
            self.reneg_leg_capital_ag = None
            self.reneg_leg_mni_tmp = None
            self.reneg_leg_ftp_mni_tmp = None

            self.reneg_leg_capital_nr = cnp.asnumpy(self.reneg_leg_capital_nr)
            self.avg_reneg_leg_capital_nr = cnp.asnumpy(self.avg_reneg_leg_capital_nr)
            self.reneg_leg_ir_nr = cnp.asnumpy(self.reneg_leg_ir_nr)
            self.reneg_leg_ftp_ir_nr = cnp.asnumpy(self.reneg_leg_ftp_ir_nr)
            self.rate_reneg_ag_nr = cnp.asnumpy(self.rate_reneg_ag_nr)

            self.rate_renego_c = None
            self.mois_depart_amor_c = None
            self.mois_fin_amor_c = None
            self.current_month_c = None
            self.drac_amor_c = None
            self.is_base_calc_30_c = None
            self.nb_days_c = None
            self.mat_date_c = None
            self.prof_amor_c = None
            self.prof_infine_c = None
            self.day_mat_date_c = None

            self.avg_renego_leg_capital_tmp = None
            self.reneg_leg_capital_tmp = None
            self.amount_renegociated = None
            self.mois_depart_amor = None
            self.drac_amor = None
            self.is_infine = None
            self.rate_renego = None
            self.douteux = None
            self.mois_fin_amor = None
            self.current_month = None
            self.is_base_calc_30 = None
            self.nb_days = None
            self.mat_date = None
            self.prof_infine = None
            self.prof_amor = None
            self.day_mat_date = None
            self.delta_days = None

            self.mempool.free_all_blocks()
            self.pinned_mempool.free_all_blocks()

    def calculate_renegociated_amount(self, remaining_capital, rarn_effect, tx_rn, current_month, mois_depart):
        rarn_effect_lagged = ut.roll_and_null(rarn_effect, val=1)
        rc = remaining_capital[:, 1:]
        base_capital_rn = ne.evaluate('rc * rarn_effect_lagged')
        base_capital_rn = ne.evaluate('where(current_month < mois_depart, 0, base_capital_rn)')
        amount_renegociated = ne.evaluate('base_capital_rn * tx_rn')
        return amount_renegociated


    def force_to_infine_if_capitalization_without_palier_or_linear_ech(self, data_hab, is_palier):
        cond = (np.array((~is_palier)) \
               & (data_hab[self.cls_fields.NC_LDP_CAPITALIZATION_RATE] != 0).values
                & (data_hab[self.cls_fields.NC_LDP_FIRST_AMORT_DATE] == 0).values)

        prof_linear_ech = np.array(data_hab[self.cls_fields.NC_PROFIL_AMOR] == "LINEAIRE_ECH")
        cond2 = (np.isnan(np.array(data_hab[self.cls_fields.NC_LDP_ECHEANCE_VAL]))) & prof_linear_ech

        data_hab[self.cls_fields.NC_PROFIL_AMOR] = np.where(cond | cond2, "INFINE", data_hab[self.cls_fields.NC_PROFIL_AMOR].values)

        return data_hab


    def get_amor_deb_for_palier_with_capi(self, data_hab, is_palier, dic_palier, dar_mois, mois_depart_amor, drac_amor, n, t):
        max_palier = dic_palier["max_palier"]
        if max_palier > 1:
            is_palier = np.array(is_palier).reshape(n)
            cle_contrat = data_hab[is_palier][self.cls_fields.NC_LDP_CLE].values
            _n= cle_contrat.shape[0]
            palier_schedule = dic_palier["palier_schedule"].loc[cle_contrat]
            mois_fin_amor = data_hab.set_index([self.cls_fields.NC_LDP_CLE]).loc[palier_schedule.index][
                                self.cls_fields.NC_DATE_FIN_AMOR].values - dar_mois
            palier_schedule = self.cls_palier.format_date_palier(palier_schedule, max_palier, self.cls_fields.NC_MOIS_PALIER_AMOR, dar_mois, _n, mois_fin_amor)
            col = self.cls_fields.NC_CAPITALIZE
            range_pal = [i for i in range(1, dic_palier["max_palier"] + 1)]
            suspend_or_capitalize = (palier_schedule[[col + str(i) for i in range_pal]].copy()).astype(bool)
            col = self.cls_fields.NC_MOIS_PALIER_AMOR
            date_pal = np.array(palier_schedule[[col + str(i) for i in range_pal]]).reshape(_n, max_palier)
            if suspend_or_capitalize.any(axis=None):
                index = ut.first_nonzero(np.array(suspend_or_capitalize.fillna(False)), axis=1, val=True)
                new_deb = np.array(date_pal)[np.arange(_n), np.minimum(index, max_palier - 1)]
                mois_depart_amor_old = mois_depart_amor.copy()
                mois_depart_amor[is_palier]\
                    = np.where(((index != -1) & (index != 0)).reshape(_n, 1), new_deb.reshape(_n, 1), mois_depart_amor[is_palier])

                drac_amor = drac_amor + mois_depart_amor_old - mois_depart_amor

        return mois_depart_amor, drac_amor

    def roll_and_null_axis0_cnp(self, data_in, shift=1, val=0):
        data = cnp.roll(data_in, shift, axis=0)
        if shift >= 0:
            data[:shift] = val
        else:
            data[shift:] = val
        return data

    def get_renegociated_capital_simplified(self, prof_amor, prof_infine, mat_date, amount_renegociated, rate_renego, current_month,
                                        mois_depart_amor, mois_fin_amor, drac_amor, dar_mois, n, t, t_l,
                                        is_infine=False):

        if t_l > 0:
            amount_renegociated_u = cnp.nan_to_num(amount_renegociated)[:, :t_l].reshape(n, t_l, 1) #AMORTIZATION_RENEGO_AVT_AMOR

            _current_month = current_month.reshape(n, 1, t)

            _rate = ((rate_renego.reshape(n, t, 1)) / 12)[:, :t_l]
            _mat_date = (mat_date.reshape(n, 1, 1))
            _mois_fin_amor = mois_fin_amor.reshape(n, 1, 1)
            prof_ech_const = prof_amor.reshape(n, 1, 1)

            if not is_infine:
                mois_reneg = cnp.arange(1, t + 1).reshape(1, t, 1)
                _drac_amor = (cnp.maximum(0, drac_amor.reshape(n, 1, 1)
                                         - cnp.maximum(0, mois_reneg - mois_depart_amor.reshape(n, 1, 1) + 1)))[:, :t_l]
                _mois_depart_amor = cnp.maximum(mois_reneg + 1, mois_depart_amor.reshape(n, 1, 1))[:, :t_l]

                if not self.is_cupy:

                    remaining_cap_reneg_u\
                        = self.cls_amor_reneg.calculate_amortization_dev2_no_infine(amount_renegociated_u , _rate,
                                                                   _current_month , _mois_depart_amor,
                                                                   _drac_amor, prof_ech_const, mois_reneg[:, :t_l]) #AMORTIZATION_RENEGO_AMOR
                else:
                    remaining_cap_reneg_u \
                            = self.cls_amor_reneg.calculate_amortization_dev_cupy_no_infine(amount_renegociated_u, _rate,
                                                                   _current_month , _mois_depart_amor,
                                                                   _drac_amor, prof_ech_const, mois_reneg[:, :t_l])

            else:
                _mois_depart_amor = cnp.minimum(t, (cnp.arange(2, t + 2)).reshape(1, t, 1))
                _drac_amor = (cnp.maximum(0, drac_amor.reshape(n, 1, 1) - (cnp.arange(1, t + 1)).reshape(1, t, 1)) \
                              + mois_depart_amor.reshape(n, 1, 1) - 1)[:, :t_l]
                _mois_depart_amor = _mois_depart_amor[:, :t_l]

                if self.is_cupy:
                    remaining_cap_reneg_u \
                        = self.cls_amor_reneg.calculate_amortization_infine_cupy(amount_renegociated_u, _current_month,
                                                           _mois_depart_amor, _mat_date, dar_mois)

                else:
                    remaining_cap_reneg_u \
                        = self.cls_amor_reneg.calculate_amortization_infine(amount_renegociated_u, _current_month,
                                                           _mois_depart_amor, _mat_date, dar_mois)

            return remaining_cap_reneg_u, _rate
        else:
            return cnp.zeros((n, t_l, t)), cnp.zeros((n, t_l, 1))


    def calculate_renego_mni_simplified(self, delta_days, nb_days_an, base_calc_30, avg_renego_leg_capital,
                                        rate_with_spread, ftp_rate, n, t, t_l):
        spread = self.cls_ra_rn_params.spread_renego / 10000
        nb_days_an = nb_days_an.reshape(n, 1, 1)
        new_mni = cnp.zeros((n, t_l, t))
        new_mni_ftp = cnp.zeros((n, t_l, t))
        new_mni_ftp_rco = cnp.zeros((n, t_l, t))
        nb_days = delta_days[:, 2:t + 1].reshape(1, 1, t - 1)
        ftp_rate = ftp_rate.reshape(n, 1, 1)
        if self.is_cupy:
            nb_days = cnp.where(base_calc_30.reshape(n, 1, 1), 30, nb_days)
        else:
            nb_days = np.where(base_calc_30.reshape(n, 1, 1), 30, nb_days)

        avg_capital_proj = avg_renego_leg_capital[:, :, 1:]
        if self.is_cupy:
            new_mni[:, :, 1:] = avg_capital_proj  * (12 * rate_with_spread * nb_days / nb_days_an)
            new_mni_ftp[:, :, 1:] = avg_capital_proj  * ((12 * rate_with_spread - spread) * nb_days / nb_days_an)
            new_mni_ftp_rco[:, :, 1:] = avg_capital_proj  * (12 * ftp_rate * nb_days / nb_days_an)
        else:
            new_mni[:, :, 1:] = ne.evaluate('avg_capital_proj  * (12 * rate_with_spread * nb_days / nb_days_an)')
            new_mni_ftp[:, :, 1:] = ne.evaluate('avg_capital_proj  * ((12 * rate_with_spread - spread) * nb_days / nb_days_an)')
            new_mni_ftp_rco[:, :, 1:] = ne.evaluate('avg_capital_proj  * (12 * ftp_rate * nb_days / nb_days_an)')

        return new_mni, new_mni_ftp.copy()

    def calculate_ind_moyen3D_cnp(self, ind2, ind1, tombee, delta_days):
        tombee = cnp.minimum(delta_days, tombee)
        ind_new = (ind1 * tombee + ind2 * (delta_days - tombee)) / (delta_days)
        return ind_new

    def calculate_ind_moyen3D(self, ind2, ind1, tombee, delta_days):
        tombee = np.minimum(delta_days, tombee)
        ind_new = ne.evaluate('(ind1 * tombee + ind2 * (delta_days - tombee)) / (delta_days)')
        return ind_new

    def roll_and_null_axis2_cnp(self, data_in, shift=1, val=0):
        data = cnp.roll(data_in, shift, axis=2)
        if shift >= 0:
            data[:, :, :shift] = val
        else:
            data[:, :, shift:] = val
        return data

    def generate_avg_renego_capital_simplified(self, delta_days, day_mat_date, reneg_leg_capital, n, t):
        day_mat_date = day_mat_date.reshape(n, 1, 1)
        delta_days = delta_days[:, 1:t + 1].reshape(1, 1, t)
        reneg_leg_capital_db = self.roll_and_null_axis2_cnp(reneg_leg_capital)
        if self.is_cupy:
            avg_static_leg_capital = self.calculate_ind_moyen3D_cnp(reneg_leg_capital, reneg_leg_capital_db, day_mat_date, delta_days)
        else:
            avg_static_leg_capital = self.calculate_ind_moyen3D(reneg_leg_capital, reneg_leg_capital_db, day_mat_date, delta_days)
        return avg_static_leg_capital


    def calculate_simplified_agregated_indics(self, reneg_leg_capital, avg_reneg_leg_capital, mni_renego, mni_ftp_renego,
                                              rate_renego, amount_reneg_c, n, t_l, t):
        mni_ag = cnp.sum(mni_renego, axis=1)
        mni_ftp_ag = cnp.sum(mni_ftp_renego, axis=1)
        capital_ag = cnp.sum(reneg_leg_capital, axis=1)
        avg_capital_ag = cnp.sum(avg_reneg_leg_capital, axis=1)
        #amount_renegociated_u = cnp.nan_to_num(amount_reneg_c)[:, :t_l].reshape(n, t_l)
        rate_renego_ag = cnp.concatenate([rate_renego.reshape(n, t_l) * 12, cnp.full((n, t - t_l), np.nan)], axis=1)

        return capital_ag, avg_capital_ag, mni_ag, mni_ftp_ag, rate_renego_ag
