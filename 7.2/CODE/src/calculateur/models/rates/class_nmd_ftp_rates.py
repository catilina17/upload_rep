import numpy as np
import logging
from utils import general_utils as gu
from calculateur.models.utils import utils as ut

nan = np.nan
logger = logging.getLogger(__name__)


class NMD_FTP_Rate_Calculator():

    def __init__(self, cls_model_params, tx_params, cls_cash_flow, cls_proj):
        self.cls_model_params = cls_model_params
        self.tx_params = tx_params
        self.cls_cash_flow = cls_cash_flow
        self.cls_fields = cls_proj.cls_fields
        self.cls_proj = cls_proj
        self.dic_tx_swap = cls_proj.cls_data_rate.dic_tx_swap
        self.cls_hz_params = cls_proj.cls_hz_params
        self.cls_flow_params = cls_model_params.cls_flow_params

    def calculate_ftp_rate(self, tci_contract_perimeter):
        tx_tci_override = np.zeros((self.cls_proj.n, self.cls_proj.t_max))
        tx_tci_mat = np.zeros((self.cls_proj.n, self.cls_proj.t_max))
        tx_tci_index = np.zeros((self.cls_proj.n, self.cls_proj.t_max))
        tx_tci_var = np.zeros((self.cls_proj.n, self.cls_proj.t_max))
        if len(tci_contract_perimeter) > 0:
            filter_tci = self.cls_proj.data_ldp[self.cls_fields.NC_LDP_CONTRACT_TYPE].isin(tci_contract_perimeter).values
        else:
            filter_tci = np.full((self.cls_proj.data_ldp.shape[0]), True)

        tx_tci_override[filter_tci] = self.get_override_ftp_rate(filter_tci)
        tx_tci_mat[filter_tci] = self.get_ftp_rate_using_maturity_method(tx_tci_override, filter_tci)
        tx_tci_index[filter_tci] = self.get_ftp_rate_using_index_method(tx_tci_override, filter_tci)
        tx_tci_var[filter_tci] = self.get_variable_ftp_rate(filter_tci)

        nmd_tci_rate_lag = np.nan_to_num(tx_tci_override) + tx_tci_mat + tx_tci_index + tx_tci_var
        nmd_tci_rate = ut.roll_and_null(nmd_tci_rate_lag, shift=-1)
        self.cls_proj.cls_rate.sc_rates_ftp = nmd_tci_rate
        self.cls_proj.cls_rate.sc_rates_ftp_lag = nmd_tci_rate_lag
        self.cls_proj.cls_rate.day_fixing_ftp = np.nan_to_num(self.cls_model_params.cls_flow_params.day_tombee.astype(int))

    def get_override_ftp_rate(self, filter_tci):
        tci_vals = self.tx_params["tci_vals"]["data"].copy()
        data_tci_scope =  self.cls_proj.data_ldp[filter_tci].copy()
        n_tci = data_tci_scope.shape[0]
        cles_a_combiner = ["NETWORK", self.cls_fields.NC_LDP_ETAB, self.cls_fields.NC_LDP_CURRENCY, self.cls_fields.NC_LDP_CONTRACT_TYPE,
                self.cls_fields.NC_LDP_MARCHE, self.cls_fields.NC_LDP_RATE_TYPE]
        cases = [data_tci_scope[self.cls_fields.NC_LDP_BASSIN] == x for x in ["CONSO_BC_BP", "CONSO_BC_CEP", "CONSO_CFF", "SOCFIM", "BP", "CEP"]]
        vals = ["BP", "CEP", "CFF", "SOCFIM", "BP", "CEP"]
        data_tci_scope["NETWORK"] = np.select(cases, vals, default="")

        data_spec = data_tci_scope[cles_a_combiner].copy()
        data_spec[self.cls_fields.NC_LDP_MARCHE] = data_spec[self.cls_fields.NC_LDP_MARCHE].fillna("")

        for col, i in zip(cles_a_combiner, range(len(cles_a_combiner))):
            list_vals = data_spec[col].unique().tolist() + ["*"]
            tci_vals = tci_vals[tci_vals.index.str.split("$").str[i].isin(list_vals)].copy()

        tci_override_val = gu.map_with_combined_key(data_spec, tci_vals, cles_a_combiner,
                                             symbol_any="*", no_map_value=np.nan, filter_comb=False,
                                             necessary_cols=1, error=False, sep="$", upper_strip=False).iloc[:, -1]
        tci_override_val = np.repeat(tci_override_val.values.reshape(n_tci, 1), self.cls_proj.t_max, axis=1)

        value_date = data_tci_scope[self.cls_fields.NC_LDP_VALUE_DATE].values
        #SEULEMENT POUR LE STOCK
        index_fwd = np.maximum(0, value_date - self.cls_hz_params.dar_mois).reshape(n_tci, 1)
        tci_override_val = np.where(index_fwd == 0, tci_override_val, np.nan)
        return tci_override_val

    def get_ftp_rate_using_index_method(self, tx_tci_override, filter_tci):
        data_tci_scope =  self.cls_proj.data_ldp[filter_tci].copy()
        n_tci = data_tci_scope.shape[0]
        no_override = np.isnan(tx_tci_override[filter_tci, 0])
        tx_tci_index_all = np.zeros((n_tci, self.cls_proj.t_max))
        is_index_method = (data_tci_scope[self.cls_fields.NC_LDP_TCI_METHOD] == "INDEX").values & no_override

        keys_index = [self.cls_fields.NC_LDP_TCI_FIXED_RATE_CODE, self.cls_fields.NC_LDP_TCI_FIXED_TENOR_CODE]
        s = len(keys_index)

        data_tci = data_tci_scope[is_index_method][keys_index].copy()

        tx_tci_index_all[is_index_method] \
            = (data_tci.join(self.cls_model_params.cls_flow_params.tx_tci_index, on=keys_index)).values[:,s:self.cls_proj.t_max + s]

        return np.nan_to_num(tx_tci_index_all)

    def get_variable_ftp_rate(self, filter_tci):
        data_tci_scope =  self.cls_proj.data_ldp[filter_tci].copy()
        n_tci = data_tci_scope.shape[0]
        tx_tci_var_all = np.zeros((n_tci, self.cls_proj.t_max))
        is_var_tci = (data_tci_scope[self.cls_fields.NC_LDP_TCI_VARIABLE_CURVE_CODE].fillna("") != "").values

        keys_index = [self.cls_fields.NC_LDP_TCI_VARIABLE_CURVE_CODE, self.cls_fields.NC_LDP_TCI_VARIABLE_TENOR_CODE]
        s = len(keys_index)
        data_tci = data_tci_scope[is_var_tci][keys_index].copy()

        tx_tci_var_all[is_var_tci] \
            = (data_tci.join(self.cls_model_params.cls_flow_params.tx_tci_var, on=keys_index)).values[:,s:self.cls_proj.t_max + s]

        return np.nan_to_num(tx_tci_var_all)

    def get_ftp_rate_using_maturity_method(self, tx_tci_override, filter_tci):
        no_override = np.isnan(tx_tci_override[:, 0])
        is_mat_method = (self.cls_proj.data_ldp[self.cls_fields.NC_LDP_TCI_METHOD] == "MATURITY").values & no_override
        is_recalc = is_mat_method & filter_tci
        n = is_recalc[is_recalc].shape[0]
        tx_tci_maturity_all = np.zeros((self.cls_proj.n, self.cls_proj.t_max))
        if n > 0:
            keys_index  = [self.cls_fields.NC_LDP_FLOW_MODEL_NMD_TCI, self.cls_fields.NC_LDP_TCI_FIXED_RATE_CODE,
                           self.cls_fields.NC_LDP_VALUE_DATE, self.cls_fields.NC_LDP_VALUE_DATE_REAL]
            s = len(keys_index)
            shapetci = self.cls_model_params.cls_flow_params.tx_tci_maturity.shape[1]

            data_tci = self.cls_proj.data_ldp[is_recalc][keys_index].copy()
            data_tci[self.cls_fields.NC_LDP_VALUE_DATE]\
                = np.maximum(0, data_tci[self.cls_fields.NC_LDP_VALUE_DATE] - self.cls_hz_params.dar_mois)

            tx_tci_maturity_all[is_recalc, :shapetci] \
                = (data_tci.join(self.cls_model_params.cls_flow_params.tx_tci_maturity, on=keys_index)).values[:, s:self.cls_proj.t_max + s]

            if np.isnan(tx_tci_maturity_all[is_recalc][:, 0]).any():
                courbes = data_tci[np.isnan(tx_tci_maturity_all[is_recalc][:, 0])].drop_duplicates().values.tolist()
                logger.error("Erreurs dans les modèles de TCI, contactez l'administrateur : %s" % courbes)


        return np.nan_to_num(tx_tci_maturity_all[filter_tci])

    @staticmethod
    def get_batched_tci_maturity_coeff(coeff_by_strate, nb_days_since_emission, tx_curves, index_fwd, dic_tx_swap,
                                       current_month, max_proj, n, t):
        if (~tx_curves.isin(list(dic_tx_swap.keys()))).any():
            liste = tx_curves[(~tx_curves.isin(list(dic_tx_swap.keys())))].unique().tolist()
            msg = "     Certaines courbes du TCI FIXE/LIQ en méthode MATURITY sont manquantes : %s" % liste
            logger.error(msg)
            tx_curves[(~tx_curves.isin(list(dic_tx_swap.keys())))] = "EURIBOR"

        tx_tci_strate = np.zeros((n, t))
        tx_tci_maturity = np.zeros((n, t))

        batch_size = 100
        for k in range(0, n // batch_size + 1):
            strat_coeff_batch = coeff_by_strate[batch_size * k:batch_size * (k + 1)].copy()
            tx_curves_batch = tx_curves.values[batch_size * k:batch_size * (k + 1)].copy()
            index_fwd_batch = index_fwd[batch_size * k:batch_size * (k + 1)].copy()
            nb_days_batch = nb_days_since_emission[batch_size * k:batch_size * (k + 1)].copy()
            n_batch = strat_coeff_batch.shape[0]

            tx_tci_strate_batch\
                = NMD_FTP_Rate_Calculator.calculate_tci_by_strate(index_fwd_batch, nb_days_batch,
                                                                  tx_curves_batch, t, n_batch, max_proj, dic_tx_swap)

            strat_coeff_mtx = NMD_FTP_Rate_Calculator.get_strat_flow_mtx(strat_coeff_batch, index_fwd_batch,
                                                                         current_month, n_batch, t)

            tx_tci_maturity_batch = NMD_FTP_Rate_Calculator.calculate_capital_weighted_tci(tx_tci_strate_batch,
                                                                                     strat_coeff_mtx, t)

            tx_tci_strate[batch_size * k:batch_size * (k + 1)] = tx_tci_strate_batch
            tx_tci_maturity[batch_size * k:batch_size * (k + 1)] = tx_tci_maturity_batch

        return tx_tci_maturity, tx_tci_strate

    @staticmethod
    def get_strat_flow_mtx(strat_coeff_batch, index_fwd_batch, current_month, n_batch, t):
        strat_coeff_mtx = np.repeat(strat_coeff_batch[:, :t], t, axis=0).reshape(n_batch, t, t)
        fwd_coeff = np.where(current_month[:, :t] > index_fwd_batch.reshape(n_batch, 1), 1, 0).reshape(n_batch, t, 1)
        triu_mtx = (np.repeat(np.ones((1, t, t)), n_batch, axis=0)
                    * np.where((np.arange(0, t).reshape(1, t) - np.arange(0, t).reshape(t, 1))
                               .reshape(1, t, t) + 1 > 0, 1, 0))
        strat_coeff_mtx = strat_coeff_mtx * triu_mtx * fwd_coeff

        return strat_coeff_mtx

    @staticmethod
    def calculate_tci_by_strate(index_fwd_batch, nb_days_batch, courbe_vect_batch, t_l, n_batch,
                               max_proj, dic_swap):
        tx_tci_by_strate = np.zeros((n_batch, t_l))
        indice_tenor = np.round(nb_days_batch[:, :t_l].astype(np.float64) / 30  * 360 / 365, 0).astype(int)
        indice_tenor = np.minimum(indice_tenor, max_proj - 1).reshape(n_batch, t_l)
        index_fwd_batch = index_fwd_batch.reshape(n_batch, 1)
        for courbe in np.unique(courbe_vect_batch):
            is_courbe = np.array(courbe_vect_batch == courbe)
            tx_cms = np.transpose(dic_swap[courbe])[
                index_fwd_batch[is_courbe], indice_tenor[is_courbe]]
            tx_tci_by_strate[is_courbe] = tx_cms

        return tx_tci_by_strate

    @staticmethod
    def calculate_capital_weighted_tci(tx_tci_by_strate, capital_by_strate, t_l):
        tx_tci_rsh = tx_tci_by_strate.reshape(tx_tci_by_strate.shape[0], 1, t_l)
        capital_sum = capital_by_strate.sum(axis=2)
        capital_sum_new = np.where(capital_sum == 0, 1, capital_sum)
        capital_weighted_tci = (tx_tci_rsh * capital_by_strate).sum(axis=2) / capital_sum_new

        capital_weighted_tci[capital_sum == 0] = tx_tci_by_strate[capital_sum == 0]

        return capital_weighted_tci
