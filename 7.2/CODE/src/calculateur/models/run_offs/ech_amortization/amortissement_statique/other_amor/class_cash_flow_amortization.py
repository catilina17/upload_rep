import numpy as np

class Cash_Flow_Amortization():
    """
    Formate les données
    """
    def __init__(self, class_init_ecoulement):
        self.class_init_ecoulement = class_init_ecoulement
        self.class_projection_manager = class_init_ecoulement.cls_proj
        self.cls_fields = class_init_ecoulement.cls_fields


    def get_amortized_capital_with_cash_flows(self, data_ldp, cap_before_amor, data_cf, current_month, dar_mois, t):
        cash_flows_data = np.zeros((0, t + 1))
        if len(data_cf) > 0 :
            cap_before_amor_cf = cap_before_amor[data_cf.index]
            if len(cap_before_amor_cf) > 0:
                cash_flows_data = np.array(data_cf[[x for x in data_cf.columns if x != self.cls_fields.NC_LDP_CONTRAT]])
                cash_flows_data = cash_flows_data[:, :t]
                _n = data_cf.shape[0]
                value_date = np.array(data_ldp.iloc[data_cf.index][self.cls_fields.NC_LDP_VALUE_DATE]).reshape(_n,1) - dar_mois
                is_forward = (value_date > 0).reshape(_n)
                cash_flows_data[is_forward] = np.where(current_month[data_cf.index][is_forward] <= value_date[is_forward],
                                                       0, cash_flows_data[is_forward])

                cash_flows_data = cap_before_amor_cf[:, 1:] - cash_flows_data.cumsum(axis=1)
                cash_flows_data = np.maximum(0, cash_flows_data)
                cash_flows_data = np.hstack([cap_before_amor_cf[:, 0:1], cash_flows_data])
        return cash_flows_data

