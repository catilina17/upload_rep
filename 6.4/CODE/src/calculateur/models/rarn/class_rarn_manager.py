import numpy as np
import pandas as pd
import numexpr as ne
from utils import general_utils as ut
from .class_rarn_immo_manager import RARN_IMMO_Manager
from .class_rarn_pcat_manager import RARN_PCAT_Manager
import logging

logger = logging.getLogger(__name__)

nan = np.nan


class RARN_Manager():
    """
    Formate les données
    """
    def __init__(self, cls_proj, cls_model_params):
        self.cls_proj = cls_proj
        self.cls_rate = cls_proj.cls_rate
        self.cls_data_rate = cls_proj.cls_data_rate
        self.cls_cal = cls_proj.cls_cal
        self.cls_model_params = cls_model_params
        self.cls_ra_rn_params = cls_model_params.cls_ra_rn_params
        self.cls_hz_params = cls_proj.cls_hz_params
        self.cls_fields = cls_proj.cls_fields

    def get_rarn_values(self, sc_rates, sc_rates_lag, data_ldp, mois_depart_rarn, current_month,
                        rarn_periods, drac_rarn, drac_init, n, t):

        tx_ra_annual_cst, ra_model = self.get_ra_rn_params(data_ldp, n, is_ra=True)
        tx_rb_annual_cst, rn_model, renego_model = self.get_ra_rn_params(data_ldp, n, is_ra=False)

        tx_ra, tx_rn\
            = self.get_rarn_constant_rate_model(data_ldp, rarn_periods, mois_depart_rarn, current_month,
                                                tx_ra_annual_cst, tx_rb_annual_cst, n, t)

        rate_renego_immo = np.full((n, t), np.nan)
        rate_renego_cat = np.full((n, t), np.nan)

        rarn_immo_cls = RARN_IMMO_Manager(self.cls_proj, self.cls_model_params)
        rarn_pcat_cls = RARN_PCAT_Manager(self.cls_proj, self.cls_model_params)

        tx_ra_immo_model, tx_rn_immo_model, is_ra_immo_model, is_rn_immo_model \
            = rarn_immo_cls.get_ra_rn_model_immo(sc_rates, data_ldp, current_month, mois_depart_rarn, rarn_periods,
                                        ra_model, ra_model, drac_rarn, drac_init, n, t)

        tx_ra_cat_model, tx_rn_cat_model, is_ra_cat_model, is_rn_cat_model  \
            = rarn_pcat_cls.get_ra_rn_model_cat(sc_rates_lag, data_ldp, current_month, mois_depart_rarn,
                                        ra_model, rn_model, drac_rarn, rarn_periods, n, t)

        rate_renego_immo_mod, is_renego_immo = rarn_immo_cls.get_rate_renego_from_cms(data_ldp, drac_rarn,
                                                         current_month, mois_depart_rarn, renego_model, t)

        rate_renego_cat_mod, is_renego_cat = rarn_pcat_cls.get_rate_renego(sc_rates_lag, data_ldp, drac_rarn, current_month,
                                                        mois_depart_rarn, renego_model, t)

        tx_rn[is_rn_immo_model] = tx_rn_immo_model
        tx_ra[is_ra_immo_model] = tx_ra_immo_model

        tx_rn[is_rn_cat_model] = tx_rn_cat_model
        tx_ra[is_ra_cat_model] = tx_ra_cat_model
        rate_renego_cat[is_renego_cat] = rate_renego_cat_mod

        if self.cls_proj.calc_mode == "quick":
            rate_renego_immo[is_renego_immo] = rate_renego_immo_mod
        else:
            rate_renego_cat[is_renego_immo] = rate_renego_immo_mod

        tx_rarn = ne.evaluate('tx_ra + tx_rn')

        if np.isnan(tx_ra[:, 0]).any() or np.isnan(tx_rn[:, 0]).any():
            logger.error("There are some contracts that don't have an RA/RN")

        self.tx_rarn, self.tx_rn, self.tx_ra = tx_rarn, tx_rn, tx_ra
        self.rate_renego_immo = rate_renego_immo
        self.rate_renego_cat = rate_renego_cat


    def get_ra_rn_params(self, data_ldp, n, is_ra=True):
        cles_a_combiner = [self.cls_fields.NC_LDP_CONTRACT_TYPE, self.cls_fields.NC_LDP_ETAB,
                           self.cls_fields.NC_LDP_RATE_TYPE,
                           self.cls_fields.NC_LDP_MATUR,
                           self.cls_fields.NC_LDP_BASSIN,
                           self.cls_fields.NC_LDP_MARCHE]

        data_spec = data_ldp[cles_a_combiner].copy()
        data_spec[self.cls_fields.NC_LDP_MARCHE] = data_spec[self.cls_fields.NC_LDP_MARCHE].fillna("")

        params = self.cls_ra_rn_params.ra_constants.copy() if is_ra else self.cls_ra_rn_params.rn_constants.copy()
        for col, i in zip(cles_a_combiner, range(len(cles_a_combiner))):
            list_vals = data_ldp[col].unique().tolist() + ["*"]
            params = params[params.index.str.split("$").str[i].isin(list_vals)].copy()

        ra_or_rn_params = ut.map_with_combined_key(data_spec, params, cles_a_combiner,
                                             symbol_any="*", no_map_value=np.nan, filter_comb=True,
                                             necessary_cols=1, error=False, sep="$", upper_strip=False)

        if is_ra:
            ra_rn_model = ra_or_rn_params.iloc[:, -1].fillna("").astype(str)
            tx_ra_or_rn_cst = np.array(ra_or_rn_params.iloc[:, -2]).reshape(n, 1)
            return tx_ra_or_rn_cst, ra_rn_model
        else:
            ra_rn_model = ra_or_rn_params.iloc[:, -2].fillna("").astype(str)
            tx_ra_or_rn_cst = np.array(ra_or_rn_params.iloc[:, -3]).reshape(n, 1)
            renego_model = ra_or_rn_params.iloc[:, -1].fillna("").astype(str)
            return tx_ra_or_rn_cst, ra_rn_model, renego_model




    def get_rarn_constant_rate_model(self, data_ldp, rarn_periods, mois_dep_rarn, current_month, tx_ra_cst, tx_rn_cst, n, t):
        tx_ra = self.apply_tx_ra_rn_constant_rate_model(data_ldp, tx_ra_cst, rarn_periods, mois_dep_rarn, current_month, n, t)
        tx_rn = self.apply_tx_ra_rn_constant_rate_model(data_ldp, tx_rn_cst, rarn_periods, mois_dep_rarn, current_month, n, t)
        return tx_ra, tx_rn


    def apply_tx_ra_rn_constant_rate_model(self, data_ldp, tx_ra_cst, rarn_periods, mois_dep_rarn, current_month, n, t):
        no_tx_ra_cst = np.isnan(tx_ra_cst)
        tx_ra_cst = ne.evaluate("where(no_tx_ra_cst, 0, tx_ra_cst)")

        ones = np.ones((n, t))
        tx_ra_an = ne.evaluate("ones * tx_ra_cst")

        # VERIFIER SI LES FLUX DE PNs négatifs subissent des RA
        is_adjst = data_ldp[[self.cls_fields.NC_NOM_MULTIPLIER]].values.reshape(n, 1) == -1
        is_adjst = is_adjst & (~data_ldp[[self.cls_fields.NC_LDP_IS_FUTURE_PRODUCT]].values).reshape(n, 1)
        tx_ra_an = np.where(is_adjst, 0, tx_ra_an)

        tx_ra = ne.evaluate("1 - (1 - tx_ra_an) ** (30 / 360)")
        tx_ra_2 = ne.evaluate("1 - (1 - tx_ra_an) ** (rarn_periods / 360)")
        tx_ra = np.where(rarn_periods != self.cls_cal.delta_days[:, :t], tx_ra_2, tx_ra)
        tx_ra = ne.evaluate("where(current_month < mois_dep_rarn, 0, tx_ra)")

        # tf-tv
        is_tv = (data_ldp[self.cls_fields.NC_LDP_RATE_TYPE] == "FLOATING").values
        _n = is_tv[is_tv].shape[0]
        if _n > 0:
            mat_date_month = np.array(data_ldp.loc[is_tv, self.cls_fields.NC_LDP_MATUR_DATE]).reshape(_n)
            mat_date_real = pd.to_datetime(data_ldp.loc[is_tv, self.cls_fields.NC_LDP_MATUR_DATE + "_REAL"])
            mat_date_day = mat_date_real.dt.day
            dar_mois = self.cls_hz_params.dar_mois
            tx_ra[is_tv, 0] = np.where(mat_date_month - dar_mois == 1,
                                       1 - (1 - tx_ra_an[is_tv, 0]) ** ((rarn_periods[is_tv, 0] - mat_date_day + 1) / 360),
                                       tx_ra[is_tv, 0])

        return tx_ra

