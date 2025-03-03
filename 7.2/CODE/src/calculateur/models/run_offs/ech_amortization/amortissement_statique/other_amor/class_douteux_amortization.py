import numpy as np
import numexpr as ne
from calculateur.models.utils import utils as ut


class Douteux_Amortization():
    """
    Formate les données
    """
    def __init__(self, class_init_ecoulement):
        self.class_init_ecoulement = class_init_ecoulement
        self.class_projection_manager = class_init_ecoulement.cls_proj
        self.cls_hz_params = class_init_ecoulement.cls_proj.cls_hz_params
        self.cls_douteux_params = class_init_ecoulement.cls_douteux_params
        self.fields = class_init_ecoulement.cls_fields

    def calculate_non_performing_amortization(self, data, capital_ec, current_month, t):
        n = capital_ec.shape[0]
        if n > 0:
            echeances_capital = np.maximum(0, ut.roll_and_null(capital_ec, shift=1) - capital_ec).reshape(n, t + 1, 1)

            ec_douteux_all = np.zeros(capital_ec.shape)

            for contrat in data[self.fields.NC_LDP_CONTRACT_TYPE].unique().tolist():
                cas = (data[self.fields.NC_LDP_CONTRACT_TYPE].values == contrat)
                if cas.any():
                    if contrat in self.cls_douteux_params.profil_douteux:
                        profil_douteux = self.cls_douteux_params.profil_douteux[contrat].copy()
                        profil_douteux = profil_douteux[:, :t + 1]
                        profil_douteux = np.vstack([profil_douteux] * (t + 1))
                        profil_douteux = np.column_stack([profil_douteux, np.zeros((t + 1, max(0, t + 1 - profil_douteux.shape[1])))])
                        profil_douteux = ut.strided_indexing_roll(profil_douteux, np.arange(0, t + 1).reshape(t + 1))
                        profil_douteux = profil_douteux.reshape(1, t + 1, t + 1)

                        echeances_cap = echeances_capital[cas]
                        echeances_douteux = ne.evaluate("echeances_cap * profil_douteux")
                        echeances_douteux = echeances_douteux.sum(axis=1)

                        capital_ec_tmp = capital_ec[cas].copy()
                        ec_douteux = capital_ec_tmp.copy()
                        depart_amor = ut.first_nonzero(echeances_cap, 1, 0, invalid_val=-1)
                        ec_douteux[:, 1:] = np.where(current_month[cas] >= depart_amor, np.nan, ec_douteux[:, 1:])
                        ec_douteux = ut.np_ffill(ec_douteux)
                        ec_douteux = ec_douteux - echeances_douteux.cumsum(axis=1)
                        ec_douteux_all[cas] = ec_douteux.copy()
                    else:
                        ec_douteux_all[cas] = capital_ec[cas].copy()

            return ec_douteux_all
        else:
            return 0
