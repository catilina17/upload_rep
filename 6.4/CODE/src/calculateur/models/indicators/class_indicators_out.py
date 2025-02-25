import numpy as np
import numexpr as ne
from calculateur.models.utils import utils as ut


class Indicators_Output():

    def __init__(self, cls_stat_runoff):
        self.cls_hz_params = cls_stat_runoff.cls_hz_params
        self.cls_fields = cls_stat_runoff.cls_fields
        self.cls_proj = cls_stat_runoff.cls_proj

    ######@profile
    def calculate_output_indics(self, data_ldp, capitals, cls_static_ind, cls_reneg_ind_immo, cls_reneg_ind_cat,
                                cls_rarn, gap_tx_params):
        dic_inds = {}
        if "lef" in self.cls_fields.exit_indicators:
            dic_inds["lef"] = cls_static_ind.static_leg_capital["liq"]["all"].copy()
            lef_proj = dic_inds["lef"][:, 1:]
            reneg_leg_capital = cls_reneg_ind_immo.reneg_leg_capital + cls_reneg_ind_cat.reneg_leg_capital
            dic_inds["lef"][:, 1:] = ne.evaluate("lef_proj + reneg_leg_capital")

        if "lem" in self.cls_fields.exit_indicators:
            dic_inds["lem"] = cls_static_ind.avg_static_leg_capital["liq"]["all"].copy()
            lem_proj = dic_inds["lem"][:, 1:]
            avg_reneg_leg_capital = cls_reneg_ind_immo.avg_reneg_leg_capital + cls_reneg_ind_cat.avg_reneg_leg_capital
            dic_inds["lem"][:, 1:] = ne.evaluate("lem_proj + avg_reneg_leg_capital")
            dic_inds["lem"][:, 0] = cls_static_ind.static_leg_capital["liq"]["all"].copy()[:, 0].copy()

        if "lem_stable" in self.cls_fields.exit_indicators:
            dic_inds["lem_stable"] = cls_static_ind.avg_static_leg_capital["liq"]["stable"].copy()
            dic_inds["lem_stable"][:, 0] = cls_static_ind.static_leg_capital["liq"]["stable"].copy()[:, 0].copy()

        if "lem_volatile" in self.cls_fields.exit_indicators:
            dic_inds["lem_volatile"] = cls_static_ind.avg_static_leg_capital["liq"]["vol"].copy()
            dic_inds["lem_volatile"][:, 0] = cls_static_ind.static_leg_capital["liq"]["vol"].copy()[:, 0].copy()

        if "lef_stable" in self.cls_fields.exit_indicators:
            dic_inds["lef_stable"] = cls_static_ind.static_leg_capital["liq"]["stable"].copy()

        if "lef_volatile" in self.cls_fields.exit_indicators:
            dic_inds["lef_volatile"] = cls_static_ind.static_leg_capital["liq"]["vol"].copy()

        if "tef" in self.cls_fields.exit_indicators:
            dic_inds["tef"] = cls_static_ind.static_leg_capital["taux"]["all"].copy()
            dic_inds["tef"][:, 0] = cls_static_ind.static_leg_capital["liq"]["all"].copy()[:, 0].copy()
            # self.adapt_gp_tx_to_gestion_rules(data_ldp, gap_tx_params, dic_inds, "tef", "lef")

        if "tem" in self.cls_fields.exit_indicators:
            dic_inds["tem"] = cls_static_ind.avg_static_leg_capital["taux"]["all"].copy()
            dic_inds["tem"][:, 0] = cls_static_ind.static_leg_capital["liq"]["all"].copy()[:, 0].copy()
            # self.adapt_gp_tx_to_gestion_rules(data_ldp, gap_tx_params, dic_inds, "tem", "lem")

        if "tem_stable" in self.cls_fields.exit_indicators:
            dic_inds["tem_stable"] = cls_static_ind.avg_static_leg_capital["taux"]["stable"].copy()
            dic_inds["tem_stable"][:, 0] = cls_static_ind.static_leg_capital["liq"]["stable"].copy()[:, 0].copy()

        if "tem_volatile" in self.cls_fields.exit_indicators:
            dic_inds["tem_volatile"] = cls_static_ind.avg_static_leg_capital["taux"]["vol"].copy()
            dic_inds["tem_volatile"][:, 0] = cls_static_ind.static_leg_capital["liq"]["vol"].copy()[:, 0].copy()

        if "mni" in self.cls_fields.exit_indicators:
            dic_inds["mni"] = cls_static_ind.static_mni["liq"]["all"].copy()
            mni_proj = dic_inds["mni"][:, 1:]
            reneg_leg_mni = cls_reneg_ind_immo.reneg_leg_mni + cls_reneg_ind_cat.reneg_leg_mni
            dic_inds["mni"][:, 1:] = ne.evaluate("mni_proj + reneg_leg_mni")

        if "mni_stable" in self.cls_fields.exit_indicators:
            dic_inds["mni_stable"] = cls_static_ind.static_mni["liq"]["stable"].copy()

        if "mni_volatile" in self.cls_fields.exit_indicators:
            dic_inds["mni_volatile"] = cls_static_ind.static_mni["liq"]["vol"].copy()

        if "mni_tci" in self.cls_fields.exit_indicators:
            dic_inds["mni_tci"] = cls_static_ind.static_ftp_mni["liq"]["all"].copy()
            mni_ftp_proj = dic_inds["mni_tci"][:, 1:]
            reneg_leg_ftp_mni = cls_reneg_ind_immo.reneg_leg_ftp_mni + cls_reneg_ind_cat.reneg_leg_ftp_mni
            dic_inds["mni_tci"][:, 1:] = ne.evaluate("mni_ftp_proj + reneg_leg_ftp_mni")

        if "mni_tci_stable" in self.cls_fields.exit_indicators:
            dic_inds["mni_tci_stable"] = cls_static_ind.static_ftp_mni["liq"]["stable"].copy()

        if "mni_tci_volatile" in self.cls_fields.exit_indicators:
            dic_inds["mni_tci_volatile"] = cls_static_ind.static_ftp_mni["liq"]["vol"].copy()

        if "mni_gptx" in self.cls_fields.exit_indicators:
            dic_inds["mni_gptx"] = cls_static_ind.static_mni["taux"]["all"].copy()

        if "mni_gptx_stable" in self.cls_fields.exit_indicators:
            dic_inds["mni_gptx_stable"] = cls_static_ind.static_mni["taux"]["stable"].copy()

        if "mni_gptx_volatile" in self.cls_fields.exit_indicators:
            dic_inds["mni_gptx_volatile"] = cls_static_ind.static_mni["taux"]["vol"].copy()

        if "mni_tci_gptx" in self.cls_fields.exit_indicators:
            dic_inds["mni_tci_gptx"] = cls_static_ind.static_ftp_mni["taux"]["all"].copy()

        if "mni_tci_gptx_stable" in self.cls_fields.exit_indicators:
            dic_inds["mni_tci_gptx_stable"] = cls_static_ind.static_ftp_mni["taux"]["stable"].copy()

        if "mni_tci_gptx_volatile" in self.cls_fields.exit_indicators:
            dic_inds["mni_tci_gptx_volatile"] = cls_static_ind.static_ftp_mni["taux"]["vol"].copy()

        if "effet_RA" in self.cls_fields.exit_indicators or "effet_RN" in self.cls_fields.exit_indicators:
            effet_ra_rn_cum_lag = ut.roll_and_null(cls_static_ind.effet_rarn_cum, val=1)

            tx_ra = cls_rarn.tx_ra
            tx_rn = cls_rarn.tx_rn
            remaining_capital_proj = capitals["all"][:, 1:].copy()

            if "effet_RA" in self.cls_fields.exit_indicators:
                dic_inds["effet_RA"] = np.zeros(cls_static_ind.static_leg_capital["liq"]["all"].shape)
                dic_inds["effet_RA"][:, 1:] = ne.evaluate("remaining_capital_proj * effet_ra_rn_cum_lag * tx_ra")

                dic_inds["tx_RA"] = np.zeros(cls_static_ind.static_leg_capital["liq"]["all"].shape)
                dic_inds["tx_RA"][:, 1:] = ne.evaluate("- (tx_ra - 1) ** 12 + 1")

            if "effet_RN" in self.cls_fields.exit_indicators:
                dic_inds["effet_RN"] = np.zeros(cls_static_ind.static_leg_capital["liq"]["all"].shape)
                dic_inds["effet_RN"][:, 1:] = ne.evaluate("remaining_capital_proj * effet_ra_rn_cum_lag * tx_rn")

                dic_inds["tx_RN"] = np.zeros(cls_static_ind.static_leg_capital["liq"]["all"].shape)
                dic_inds["tx_RN"][:, 1:] = ne.evaluate("- (tx_rn - 1) ** 12 + 1")

        if "lef_statique" in self.cls_fields.exit_indicators:
            dic_inds["lef_statique"] = cls_static_ind.static_leg_capital["liq"]["all"].copy()

        if "lef_renego" in self.cls_fields.exit_indicators:
            reneg_leg_capital = cls_reneg_ind_immo.reneg_leg_capital + cls_reneg_ind_cat.reneg_leg_capital
            dic_inds["lef_renego"] = np.zeros(cls_static_ind.static_leg_capital["liq"]["all"].shape)
            dic_inds["lef_renego"][:, 1:] = dic_inds["lef_renego"][:, 1:] + reneg_leg_capital.copy()

        if "lem_statique" in self.cls_fields.exit_indicators:
            dic_inds["lem_statique"] = cls_static_ind.avg_static_leg_capital["liq"]["all"].copy()
            dic_inds["lem_statique"][:, 0] = cls_static_ind.static_leg_capital["liq"]["all"][:, 0].copy()

        if "lem_renego" in self.cls_fields.exit_indicators:
            avg_reneg_leg_capital = cls_reneg_ind_immo.avg_reneg_leg_capital + cls_reneg_ind_cat.avg_reneg_leg_capital
            dic_inds["lem_renego"] = np.zeros(cls_static_ind.static_leg_capital["liq"]["all"].shape)
            dic_inds["lem_renego"][:, 1:] = dic_inds["lem_renego"][:, 1:] + avg_reneg_leg_capital.copy()

        if "mni_statique" in self.cls_fields.exit_indicators:
            dic_inds["mni_statique"] = cls_static_ind.static_mni["liq"]["all"].copy()

        if "mni_renego" in self.cls_fields.exit_indicators:
            reneg_leg_mni = cls_reneg_ind_immo.reneg_leg_mni + cls_reneg_ind_cat.reneg_leg_mni
            dic_inds["mni_renego"] = np.zeros(cls_static_ind.static_leg_capital["liq"]["all"].shape)
            dic_inds["mni_renego"][:, 1:] = dic_inds["mni_renego"][:, 1:] + reneg_leg_mni.copy()

        if "sc_rates" in self.cls_fields.exit_indicators:
            dic_inds["sc_rates"] = np.zeros((self.cls_proj.n, self.cls_proj.t + 1))
            dic_inds["sc_rates"][:, 1:] = np.nan_to_num(dic_inds["mni"][:, 1:] / dic_inds["lem"][:, 1:]) * 12

        if "sc_rates_statique" in self.cls_fields.exit_indicators:
            dic_inds["sc_rates_statique"] = np.zeros((self.cls_proj.n, self.cls_proj.t + 1))
            dic_inds["sc_rates_statique"][:, 1:] = self.cls_proj.cls_rate.sc_rates[:, :self.cls_proj.t]

        if "sc_rates_reneg" in self.cls_fields.exit_indicators:
            dic_inds["sc_rates_reneg"] = np.zeros((self.cls_proj.n, self.cls_proj.t + 1))
            dic_inds["sc_rates_reneg"][:, 1:] = cls_reneg_ind_immo.reneg_rate + cls_reneg_ind_cat.reneg_rate

        if "sc_rates_tci" in self.cls_fields.exit_indicators:
            dic_inds["sc_rates_tci"] = np.zeros((self.cls_proj.n, self.cls_proj.t + 1))
            dic_inds["sc_rates_tci"][:, 1:] = self.cls_proj.cls_rate.sc_rates_ftp[:, :self.cls_proj.t]

        for ind in dic_inds.keys():
            dic_inds[ind] = dic_inds[ind][:, : self.cls_hz_params.nb_months_proj + 1]

        return dic_inds


def adapt_gp_tx_to_gestion_rules(self, data_ldp, gap_tx_params, dic_inds, ind, ind_liq):
    filter = data_ldp[self.cls_fields.NC_LDP_RATE_CODE].isin(gap_tx_params["INDEXS"].index.unique().tolist())
    if filter.any():
        _n = filter[filter].shape[0]
        col_index_ag = gap_tx_params["COL_INDEX_AG"]
        col_index_val = gap_tx_params["COL_VAL"]
        coeff_gp_tx_tf = gap_tx_params["GAP_TF_COEFF_INDEX"]
        coeff_gp_tx_inf = gap_tx_params["GAP_INF_COEFF_INDEX"]
        data_index = data_ldp[[self.cls_fields.NC_LDP_RATE_CODE]][filter]
        data_index = data_index.join(gap_tx_params["INDEXS"], on=self.cls_fields.NC_LDP_RATE_CODE)

        coeff_index_gp_tf = data_index.join(coeff_gp_tx_tf, on=gap_tx_params["COL_INDEX_AG"])[
            col_index_val].values.reshape(_n, 1)
        dic_inds[ind][filter] = coeff_index_gp_tf * (dic_inds[ind_liq][filter] - dic_inds[ind][filter]) + dic_inds[ind][
            filter]
