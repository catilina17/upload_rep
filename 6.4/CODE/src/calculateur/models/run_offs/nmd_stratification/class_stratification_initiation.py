import numpy as np
import numexpr as ne

nan = np.nan


class Stratification_Initiation():
    def __init__(self, cls_proj, cls_flow_params):
        self.cls_proj = cls_proj
        self.nb_rm_groups = cls_proj.nb_rm_groups
        self.cls_rate = cls_proj.cls_rate
        self.cls_data_rate = cls_proj.cls_data_rate
        self.cls_cal = cls_proj.cls_cal
        self.cls_hz_params = cls_proj.cls_hz_params
        self.cls_fields = cls_proj.cls_fields
        self.cls_flow_params = cls_flow_params
    #########@profile
    def get_stratification(self):
        n = self.cls_proj.n
        t = self.cls_proj.t
        data_ldp = self.cls_proj.data_ldp.copy()
        current_month = np.repeat(np.arange(0, self.cls_proj.t + 1)
                                  .reshape(1, self.cls_proj.t + 1), self.cls_proj.n, axis=0)
        outstd_crd = (np.array(data_ldp[self.cls_fields.NC_LDP_OUTSTANDING]).reshape(n, 1, 1).astype(np.float64))
        nominal = (np.array(data_ldp[self.cls_fields.NC_LDP_NOMINAL]).reshape(n, 1, 1).astype(np.float64))
        dar_mois = self.cls_hz_params.dar_mois
        point_depart = self.cls_cal.mois_depart
        self.ec_depart = self.generate_init_ec(data_ldp, outstd_crd, nominal, current_month, dar_mois,
                                                        point_depart, n, t)

        self.coeffs_strates = self.cls_flow_params.monthly_flow
        self.coeffs_strates_all = self.cls_flow_params.monthly_flow_all
        self.coeffs_gptx = self.cls_flow_params.monthly_flow_gptx
        self.coeffs_gptx_all = self.cls_flow_params.monthly_flow_gptx_all

    ##########@profile
    def generate_init_ec(self, data_ldp, outstd_crd, nominal, current_month, dar_mois,
                         point_depart, n, t):
        is_forward = np.array(data_ldp[self.cls_fields.NC_LDP_VALUE_DATE] > dar_mois).reshape(n, 1)
        value_date = np.array(np.maximum(0, data_ldp[self.cls_fields.NC_LDP_VALUE_DATE] - dar_mois)).reshape(n, 1)

        """ ECOULMENTS FORWARD DE DEPART SANS REGLE DE DEBLOCAGE"""
        ec_date_rule = self.ecoulements_forward_def(is_forward, current_month, nominal, point_depart, value_date, n, t)

        """ ECOULEMENT DEPART DEFAUT"""
        ec_def = np.zeros((n, t + 1))
        ec_def[:, 0] = outstd_crd.reshape(n)

        """ ECOULEMENT DEPART DEFAULT"""
        ec_depart = ne.evaluate("where(current_month == point_depart, 0, ec_def)")
        #ec_depart_prolonge = ut.np_ffill(ec_depart_prolonge)

        ec_depart[(is_forward).reshape(n)] = ec_date_rule

        return ec_depart.copy()

    def ecoulements_forward_def(self, is_forward, current_month, nominal, point_depart, value_date, n, t):
        f = is_forward
        p = f[f].shape[0]
        _current_month = current_month[f.reshape(n)]
        _value_date = value_date[f.reshape(n, 1)].reshape(p, 1)
        _nominal = nominal[f.reshape(n, 1)].reshape(p, 1)
        _point_depart = point_depart[f.reshape(n, 1)].reshape(p, 1)
        ec_date_rule = np.zeros((p, t + 1))

        if p > 0:
            cond_vd = (_current_month == _value_date)
            ec_date_rule = ne.evaluate("where(cond_vd, _nominal, ec_date_rule)")
            #cond_vd_prior = (_current_month < _value_date)
            #ec_date_rule = ne.evaluate("where(_current_month > _value_date, nan, ec_date_rule)")
            #ec_date_rule = ut.np_ffill(ec_date_rule)

        return ec_date_rule
