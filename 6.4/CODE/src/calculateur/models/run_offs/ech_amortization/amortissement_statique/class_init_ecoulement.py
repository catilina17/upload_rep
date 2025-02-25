import numpy as np
import numexpr as ne
from calculateur.models.utils import utils as ut
from calculateur.models.mappings.products_perimeters import Perimeters

nan = np.nan


class Init_Ecoulement():
    def __init__(self, cls_proj, cls_model_params):
        self.cls_proj = cls_proj
        self.cls_rate = cls_proj.cls_rate
        self.cls_data_rate = cls_proj.cls_data_rate
        self.cls_cal = cls_proj.cls_cal
        self.cls_fix_cal = cls_proj.cls_fixing_cal
        self.cls_hz_params = cls_proj.cls_hz_params
        self.cls_debloc_params = cls_model_params.cls_debloc_params
        self.cls_fields = cls_proj.cls_fields
        self.cls_ra_rn_params = cls_model_params.cls_ra_rn_params
        self.cls_douteux_params = cls_model_params.cls_douteux_params

    def get_ec_before_amortization(self):
        data_ldp = self.cls_proj.data_ldp.copy()
        current_month= np.repeat(np.arange(0, self.cls_proj.t + 1)
                                 .reshape(1, self.cls_proj.t + 1), self.cls_proj.n, axis=0)

        outstd_crd = (np.array(data_ldp[self.cls_fields.NC_LDP_OUTSTANDING])
                      .reshape(self.cls_proj.n, 1).astype(np.float64))

        nominal = (np.array(data_ldp[self.cls_fields.NC_LDP_NOMINAL])
                   .reshape(self.cls_proj.n, 1).astype(np.float64))

        rates = self.cls_rate.sc_rates[:, :self.cls_proj.t]
        self.generate_eclt_before_amortization(data_ldp, outstd_crd, nominal, rates, current_month,
                                               self.cls_hz_params.dar_mois, self.cls_cal.mois_depart_amor,
                                               self.cls_proj.n, self.cls_proj.t)

    ###########@profile
    def generate_eclt_before_amortization(self, data_ldp, outstd_crd, nominal, rate, current_month, dar_mois,
                                          point_depart, n, t):
        is_forward = np.array(data_ldp[self.cls_fields.NC_LDP_VALUE_DATE] > dar_mois).reshape(n, 1)
        releasing_date = np.array(np.maximum(0, data_ldp[self.cls_fields.NC_LDP_RELEASING_DATE] - dar_mois)).reshape(n, 1)
        releasing_date_r = data_ldp[self.cls_fields.NC_LDP_RELEASING_DATE].values
        value_date = np.array(np.maximum(0, data_ldp[self.cls_fields.NC_LDP_VALUE_DATE] - dar_mois)).reshape(n, 1)
        num_rule = (np.nan_to_num(np.array(data_ldp[self.cls_fields.NC_LDP_RELEASING_RULE]), nan=-1).astype(int)).reshape(
            n, 1)
        is_release_rule = (num_rule != -1).reshape(n, 1)
        is_capitalized = np.array(data_ldp[self.cls_fields.NC_IS_CAPI_BEFORE_AMOR]).reshape(n, 1)
        nb_days_an = np.array(data_ldp[self.cls_fields.NB_DAYS_AN]).reshape(n, 1)
        accrued_interests = np.array(data_ldp[self.cls_fields.NC_LDP_INTERESTS_ACCRUALS]).reshape(n, 1)
        nb_day_m0 = np.array(data_ldp[self.cls_fields.NB_DAYS_M0]).reshape(n, 1)
        base_calc = np.array(data_ldp[self.cls_fields.NC_LDP_ACCRUAL_BASIS]).reshape(n, 1)
        is_produit_marche = np.array(data_ldp[self.cls_fields.NC_LDP_CONTRACT_TYPE].isin(Perimeters.produits_marche)).reshape(n, 1)
        trade_date_month = np.array(data_ldp[self.cls_fields.NC_LDP_TRADE_DATE])

        """ ECOULEMENTS FORWARD DE DEPART AVEC REGLE DE DEBLOCAGE"""
        ec_num_rule = self.ecoulements_num_rule(data_ldp, is_forward, is_release_rule, outstd_crd, nominal,
                                                num_rule, current_month, t, n, rate, is_capitalized, nb_days_an,
                                                value_date, accrued_interests, nb_day_m0, base_calc, trade_date_month,
                                                dar_mois)

        """ ECOULMENTS FORWARD DE DEPART SANS REGLE DE DEBLOCAGE"""
        ec_date_rule = self.ecoulements_forward_def(is_forward, is_release_rule, current_month, nominal,
                                                    point_depart, value_date, releasing_date, releasing_date_r,
                                                    outstd_crd, is_capitalized, rate, nb_days_an, accrued_interests,
                                                    nb_day_m0, base_calc, is_produit_marche, n, t)

        """ ECOULEMENT NON FWD MAIS DECALE"""
        cond_value_date = (value_date == 0) & (point_depart >= 2)
        ec_first_amor = np.zeros((n, t + 1))
        ec_first_amor = ne.evaluate(
            "where((current_month < point_depart) & cond_value_date, outstd_crd, ec_first_amor)")

        """ ECOULEMENT DEPART DEFAUT"""
        ec_def = np.zeros((n, t + 1))
        ec_def[:, 0] = outstd_crd.reshape(n)

        """ ECOULEMENT DEPART DEFAULT"""
        ec_depart = ne.evaluate("where(~is_forward & cond_value_date, ec_first_amor, ec_def)")
        ec_depart_prolonge = ne.evaluate("where(current_month >= point_depart, nan, ec_depart)")
        ec_depart_prolonge = ut.np_ffill(ec_depart_prolonge)

        ec_depart_prolonge[(is_forward & is_release_rule).reshape(n)] = ec_num_rule
        ec_depart_prolonge[(is_forward & ~is_release_rule).reshape(n)] = ec_date_rule

        self.ec_depart = ec_depart_prolonge.copy()

    def ecoulements_num_rule(self, data_ldp, is_forward, is_release_rule, outstd_crd, nominal, num_rule, current_month,
                             t, n, rate, is_capitalized, nb_days_an, value_date,
                             accrued_interests, nb_day_m0, base_calc, trade_date_month, dar_mois):

        cle = [self.cls_fields.NC_LDP_CONTRACT_TYPE, self.cls_fields.NC_LDP_MARCHE,self.cls_fields.NC_LDP_RELEASING_RULE]
        data_num_rule = data_ldp.loc[is_forward & is_release_rule,cle].copy()

        if len(data_num_rule) > 0:
            self.cls_debloc_params.get_deblocage_mtx(data_num_rule)
            mtx_deblocage_all = self.cls_debloc_params.mtx_deblocage_all
            cancel_rate_all = self.cls_debloc_params.cancel_rates_all

            p = data_num_rule.shape[0]
            _outstd_crd = outstd_crd[is_forward & is_release_rule].reshape(p, 1)
            _nominal = nominal[is_forward & is_release_rule].reshape(p, 1)
            _num_rule = num_rule[(is_forward & is_release_rule)].reshape(p, 1)
            _current_month = current_month[(is_forward & is_release_rule).reshape(n)]
            # _point_depart = point_depart[(is_forward & is_release_rule)].reshape(p, 1)
            indices_lignes = np.where(is_forward & is_release_rule)[0]
            _rate = rate[(is_forward & is_release_rule).reshape(n)].reshape(p, t)
            _is_capitalized = is_capitalized[is_forward & is_release_rule].reshape(p)
            _nb_days_an = nb_days_an[is_forward & is_release_rule].reshape(p)
            _value_date = value_date[is_forward & is_release_rule].reshape(p, 1)
            _accrued_interests = accrued_interests[is_forward & is_release_rule].reshape(p, 1)
            _nb_day_m0 = nb_day_m0[is_forward & is_release_rule].reshape(p, 1)
            _base_calc = base_calc[is_forward & is_release_rule]
            _trade_date_month = trade_date_month[(is_forward & is_release_rule).reshape(n)]

            cancel_rate_all = np.round(cancel_rate_all, 5)
            mtx_deblocage_all[:, 0] = _outstd_crd.reshape(p)
            mtx_deblocage_all1 = mtx_deblocage_all[:, 1:]
            mtx_deblocage_all[:, 1:] = np.minimum(ne.evaluate("_outstd_crd + mtx_deblocage_all1 * _nominal"
                                                              " * (1 - cancel_rate_all)"), _nominal)
            s = mtx_deblocage_all.shape[1]
            ec_num_rule = np.hstack([mtx_deblocage_all, np.full([p, t - s + 1], 0)])
            shift_begin = np.maximum(0, _trade_date_month - dar_mois)
            ec_num_rule = ut.strided_indexing_roll(ec_num_rule, shift_begin)
            ec_num_rule[(_current_month > shift_begin.reshape(p, 1) + s - 1 - _num_rule) & (_current_month <= _value_date)] = np.nan
            ec_num_rule = ut.np_ffill(ec_num_rule)

            ec_num_rule = self.add_capitalized_interests(ec_num_rule, _is_capitalized, _rate, \
                                                         _current_month, _value_date, _nb_days_an, _accrued_interests,
                                                         _nb_day_m0, _base_calc)

            ec_num_rule = ne.evaluate("where(_current_month > _value_date, nan, ec_num_rule)")
            ec_num_rule = ut.np_ffill(ec_num_rule)
        else:
            ec_num_rule = np.zeros((0, t + 1))

        return ec_num_rule

    def ecoulements_forward_def(self, is_forward, is_release_rule, current_month, nominal, point_depart, value_date,
                                releasing_date, releasing_date_r, outstd_crd, is_capitalized, rate,
                                nb_days_an, accrued_interests, nb_day_m0, base_calc, is_produit_marche, n, t):
        f = is_forward & ~is_release_rule
        cond_nz_rls_date = (releasing_date_r != 0)
        cond_nz_rls_date = cond_nz_rls_date[f.reshape(n)]
        p = cond_nz_rls_date.shape[0]
        cond_nz_rls_date = cond_nz_rls_date.reshape(p, 1)
        _current_month = current_month[f.reshape(n)]
        _releasing_date = releasing_date[f.reshape(n, 1)].reshape(p, 1)
        _value_date = value_date[f.reshape(n, 1)].reshape(p, 1)
        _nominal = nominal[f.reshape(n, 1)].reshape(p, 1)
        _point_depart = point_depart[f.reshape(n, 1)].reshape(p, 1)
        _outstd_crd = outstd_crd[f.reshape(n, 1)].reshape(p, 1)
        _is_capitalized = is_capitalized[f.reshape(n, 1)].reshape(p)
        _rate = rate[f.reshape(n)].reshape(p, t)
        _nb_days_an = nb_days_an[f.reshape(n, 1)].reshape(p)
        _accrued_interests = accrued_interests[f.reshape(n, 1)].reshape(p, 1)
        _nb_day_m0 = nb_day_m0[f.reshape(n, 1)].reshape(p, 1)
        _base_calc = base_calc[f.reshape(n, 1)]
        _is_produit_marche = is_produit_marche[f.reshape(n, 1)].reshape(p, 1)

        ec_date_rule = np.zeros((p, t + 1))

        if p > 0:
            cond_rd = ne.evaluate(
                "cond_nz_rls_date & (_current_month >= _releasing_date) & (_current_month <= _value_date)")
            ec_date_rule = ne.evaluate("where(cond_rd, _nominal, ec_date_rule)")
            cond_rd_prior = ne.evaluate("cond_nz_rls_date & (_current_month < _releasing_date)")

            # PRODUITS MARCHE
            ec_date_rule = ne.evaluate("where(cond_rd_prior & ~_is_produit_marche, _outstd_crd, ec_date_rule)")
            ec_date_rule = ne.evaluate("where(cond_rd_prior & _is_produit_marche, 0, ec_date_rule)")

            cond_vd = ~cond_nz_rls_date & (_current_month == _value_date)
            ec_date_rule = ne.evaluate("where(cond_vd, _nominal, ec_date_rule)")
            cond_vd_prior = ~cond_nz_rls_date & (_current_month < _value_date)

            # PRODUITS MARCHE
            ec_date_rule = ne.evaluate("where(cond_vd_prior & ~_is_produit_marche, _outstd_crd, ec_date_rule)")
            ec_date_rule = ne.evaluate("where(cond_vd_prior & _is_produit_marche, 0, ec_date_rule)")

            ec_date_rule = self.add_capitalized_interests(ec_date_rule, _is_capitalized, _rate,
                                                          _current_month, _value_date, _nb_days_an, _accrued_interests,
                                                          _nb_day_m0, _base_calc)

            ec_date_rule = ne.evaluate("where(_current_month > _value_date, nan, ec_date_rule)")
            ec_date_rule = ut.np_ffill(ec_date_rule)

        return ec_date_rule

    def add_capitalized_interests(self, ec_num_rule, is_capitalized, rate, _current_month, _value_date,
                                  _nb_days_an, _accrued_interests, _nb_day_m0, _base_calc):
        ecoul_dep = ec_num_rule[is_capitalized].copy()
        n = ecoul_dep.shape[0]
        t = ecoul_dep.shape[1]
        _rate = rate[is_capitalized.reshape(is_capitalized.shape[0])].reshape(n, t - 1)
        _current_month = _current_month[is_capitalized]
        _value_date = _value_date[is_capitalized].reshape(n, 1)
        _nb_days_an = _nb_days_an[is_capitalized].reshape(n, 1)
        _accrued_interests = _accrued_interests[is_capitalized].reshape(n)
        _nb_day_m0 = _nb_day_m0[is_capitalized].reshape(n)
        _base_calc = _base_calc[is_capitalized]

        if n > 0:
            rate_mtx = np.hstack([_rate[:, 0:1], _rate]).reshape(n, t)
            delta_days = self.cls_cal.delta_days[:, :t]
            base_calc_cond = np.isin(_base_calc, ["30/360", "30E/360", "30A/360"]).reshape(n, 1)
            delta_days = np.where(base_calc_cond, 30, delta_days)
            rate_mtx_m0 = (_accrued_interests + ecoul_dep[:, 0] * _rate[:, 0].reshape(
                n) * 12 * _nb_day_m0 / _nb_days_an.reshape(n)) \
                          / (delta_days[:, 0] / _nb_days_an.reshape(n) * ecoul_dep[:, 0])
            rate_mtx[:, 0] = np.where((_accrued_interests != 0), rate_mtx_m0, rate_mtx[:, 0])
            rate_mtx = np.nan_to_num(rate_mtx)
            capitalized_interests = ne.evaluate(
                "ecoul_dep * (rate_mtx * delta_days / _nb_days_an)")  # attention dépend de base calc
            capitalized_interests = ne.evaluate("where(_current_month >= _value_date, 0, capitalized_interests)")
            capitalized_interests = capitalized_interests.sum(axis=1).reshape(n, 1)
            ec_num_rule[is_capitalized] = ne.evaluate("ecoul_dep + where(_current_month == _value_date,\
                                                      capitalized_interests, 0)")

        return ec_num_rule
