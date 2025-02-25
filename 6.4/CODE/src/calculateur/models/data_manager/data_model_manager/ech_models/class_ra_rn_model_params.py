import pandas as pd
from utils import excel_utils as ex
import numpy as np
from calculateur.models.utils import utils as ut


class Data_RA_RN_Model_Params():
    """
    Formate les données
    """
    def __init__(self, model_wb, nb_mois_proj):
        self.model_wb = model_wb
        self.nb_mois_proj = nb_mois_proj
        self.load_col_model_file()
        self.MODEL_NAME_RA_IMMO = "RA_IMMO"
        self.MODEL_NAME_RN_IMMO = "RA_IMMO"
        self.MODEL_NAME_RENEGO_IMMO = "RENEGO_IMMO"

        self.MODEL_NAME_RA_CAT_LIQ = "RA_CAT_LIQ"
        self.MODEL_NAME_RA_CAT_TX = "RA_CAT_TX"
        self.MODEL_NAME_RN_CAT_LIQ = "RN_CAT_LIQ"
        self.MODEL_NAME_RN_CAT_TX = "RN_CAT_TX"
        self.MODEL_NAME_RENEGO_CAT = "RENEGO_CAT_LIQ"

    def load_col_model_file(self):
        self.NC_SPREAD_RENEGO = "_SpreadRN"
        self.NC_TX_RARN_USER = "_USER_RARN"

        self.NC_MARGE_MODELE = "_MARGES_MODELE"
        self.NC_TX_FLOOR = "_FloorNvTxProd"
        self.NC_PARAM_RARN = "_PARAM_RARN"
        self.NC_COEFF_RARN = "_COEF_RARN"

        self.NC_RA_STRUCT = "RA Struct"
        self.NC_NIVEAU = "Niveau"
        self.NC_SENSI = "Sensi"
        self.NC_BURNOUT = "Burnout"
        self.NC_MIN_RA = "Min RA"
        self.NC_MAX_RA = "Max RA"

        self.NC_TYPE_MODEL = "_ModeCalcRA"

        self.NC_ALPHA = "α"
        self.NC_GAMMA = "ɤ"

        self.NC_ACTIVER_BACKTEST = "_activer_backtest"
        self.NC_BACKTEST_RA_DRAC50 = "_backtest_ra_drac50"
        self.NC_BACKTEST_RA_DRAC80 = "_backtest_ra_drac80"
        self.NC_BACKTEST_RA_DRAC100 = "_backtest_ra_drac100"
        self.NC_BACKTEST_RN_DRAC50 = "_backtest_rn_drac50"
        self.NC_BACKTEST_RN_DRAC80 = "_backtest_rn_drac80"
        self.NC_BACKTEST_RN_DRAC100 = "_backtest_rn_drac100"

        self.NC_MOD_RA_CONST = "_CONST_RA"
        self.NC_MOD_RN_CONST = "_CONST_RN"
        self.NC_MOD_ETAB = "ETAB"
        self.NC_MOD_CTRT = "CONTRACT_TYPE"
        self.NC_MOD_MATUR = "MATUR"
        self.NC_MOD_BASSIN = "BASSIN"
        self.NC_MOD_MARCHE = "MARCHE"
        self.NC_MOD_RATE_TYPE = "RATE_TYPE"
        self.ra_cst_val = "RA_CONSTANT"
        self.rn_cst_val = "RN_CONSTANT"
        self.ra_model = "RA_MODEL"
        self.rn_model = "RN_MODEL"

        self.NC_RARN_PCAT_LIQ = "_CAT_RA_RN_LIQ"
        self.NC_RARN_PCAT_TX = "_CAT_RA_RN_TX"

        self.NC_TYPE_RA_PCAT_LIQ = "_TYPE_RA_CAT_LIQ"
        self.NC_TYPE_RN_PCAT_LIQ = "_TYPE_RN_CAT_LIQ"
        self.NC_TYPE_RA_PCAT_TX = "_TYPE_RA_CAT_TX"
        self.NC_TYPE_RN_PCAT_TX = "_TYPE_RN_CAT_TX"

        self.NC_RA_WITHOUT_ARBI = "RA_SS_ARBITRAGE"
        self.NC_RA_ARBI = "RA_ARBITRAGE"
        self.NC_RN_WITHOUT_ARBI = "RN_SS_ARBITRAGE"
        self.NC_RN_ARBI = "RN_ARBITRAGE"

    def load_rarn_model_params(self):
        self.load_crif_ra_rn_model()
        self.load_ra_rn_p_cat_model()

        """ Modèles RA/RN constants """
        cle = [self.NC_MOD_CTRT, self.NC_MOD_ETAB, self.NC_MOD_RATE_TYPE, self.NC_MOD_MATUR,
               self.NC_MOD_BASSIN, self.NC_MOD_MARCHE]

        self.ra_constants = ex.get_dataframe_from_range(self.model_wb, self.NC_MOD_RA_CONST)
        self.ra_constants = ut.explode_dataframe(self.ra_constants , [self.NC_MOD_ETAB, self.NC_MOD_CTRT,
                                                                      self.NC_MOD_RATE_TYPE, self.NC_MOD_MARCHE, self.NC_MOD_MATUR])
        self.ra_constants['new_key'] = self.ra_constants[cle].apply(lambda row: '$'.join(row.values.astype(str)), axis=1)
        self.ra_constants = self.ra_constants.drop_duplicates(subset=cle).set_index('new_key').drop(columns=cle, axis=1).copy()

        self.rn_constants = ex.get_dataframe_from_range(self.model_wb, self.NC_MOD_RN_CONST)
        self.rn_constants = ut.explode_dataframe(self.rn_constants , [self.NC_MOD_ETAB, self.NC_MOD_CTRT,
                                                                      self.NC_MOD_RATE_TYPE, self.NC_MOD_MARCHE, self.NC_MOD_MATUR])
        self.rn_constants['new_key'] = self.rn_constants[cle].apply(lambda row: '$'.join(row.values.astype(str)), axis=1)
        self.rn_constants = self.rn_constants.drop_duplicates(subset=cle).set_index('new_key').drop(columns=cle, axis=1).copy()


    def load_backtest_params(self):
        try:
            self.activer_backtest = ex.get_value_from_named_ranged(self.model_wb, self.NC_ACTIVER_BACKTEST)
            self.activer_backtest = True if self.activer_backtest == "OUI" else False
            if self.activer_backtest:
                self.drac50_ra = float(ex.get_value_from_named_ranged(self.model_wb, self.NC_BACKTEST_RA_DRAC50))
                self.drac80_ra = float(ex.get_value_from_named_ranged(self.model_wb, self.NC_BACKTEST_RA_DRAC80))
                self.drac100_ra = float(ex.get_value_from_named_ranged(self.model_wb, self.NC_BACKTEST_RA_DRAC100))
                self.drac50_rn = float(ex.get_value_from_named_ranged(self.model_wb, self.NC_BACKTEST_RN_DRAC50))
                self.drac80_rn = float(ex.get_value_from_named_ranged(self.model_wb, self.NC_BACKTEST_RN_DRAC80))
                self.drac100_rn = float(ex.get_value_from_named_ranged(self.model_wb, self.NC_BACKTEST_RN_DRAC100))
        except:
            self.activer_backtest = False

    def parse_coeff_rarn(self, coeff_rarn_md):
        self.gamma = {};
        self.alpha = {}
        for i, name in zip(range(0, coeff_rarn_md.shape[0]), ["D50", "D80", "D100"]):
            self.gamma[name] = coeff_rarn_md.loc[i, self.NC_GAMMA]
            self.alpha[name] = coeff_rarn_md.loc[i, self.NC_ALPHA]


    def parse_rarn_params(self, param_rarn_md):
        self.tx_ra_struct = {};
        self.niveau = {};
        self.burnout = {};
        self.sensi = {};
        self.min_ra = {};
        self.max_ra = {}
        for i, name in zip(range(0, param_rarn_md.shape[0]), ["D50", "D80", "D100"]):
            self.tx_ra_struct[name] = param_rarn_md.loc[i, self.NC_RA_STRUCT]
            self.niveau[name] = param_rarn_md.loc[i, self.NC_NIVEAU]
            self.burnout[name] = param_rarn_md.loc[i, self.NC_BURNOUT]
            self.sensi[name] = param_rarn_md.loc[i, self.NC_SENSI]
            self.min_ra[name] = param_rarn_md.loc[i, self.NC_MIN_RA]
            self.max_ra[name] = param_rarn_md.loc[i, self.NC_MAX_RA]


    def parse_marge_model(self, marge_md):
        marge_md.columns = ["M" + str(int(col)) for col in marge_md.columns]
        init_cols = marge_md.columns.tolist()
        list_new_cols = []
        name_cols = []
        for i in range(1, 168):
            if i % 12 != 0:
                name_cols.append("M" + str(i))
                new_col = 1 / 12 * ((int(i / 12) * 12 + 12 - i) * marge_md["M" + str((int(i / 12)) * 12)] \
                                    + (i - int(i / 12) * 12) * marge_md["M" + str((int(i / 12) + 1) * 12)])
                list_new_cols.append(new_col)

        for i in range(169, 360):
            if i % 60 != 0:
                name_cols.append("M" + str(i))
                new_col = 1 / 60 * ((int(i / 60) * 60 + 60 - i) * marge_md["M" + str(int(i / 60) * 60)] \
                                    + (i - int(i / 60) * 60) * marge_md["M" + str((int(i / 60) + 1) * 60)])
                list_new_cols.append(new_col)

        marge_md = pd.concat([marge_md] + list_new_cols, axis=1)
        marge_md.columns = init_cols + name_cols
        marge_md = marge_md[["M" + str(i) for i in range(1, 361)]].copy()

        return np.array(marge_md).reshape(360)

    def load_pcat_ra_rn_model(self):
        """ MODELE RARN"""
        self.ra_cat = ex.get_dataframe_from_range(self.model_wb, self.NC_MARGE_MODELE)
        self.rn_cat = ex.get_dataframe_from_range(self.model_wb, self.NC_MARGE_MODELE)



    def load_crif_ra_rn_model(self):
        """ SPREAD RENEGO"""
        self.spread_renego = ex.get_value_from_named_ranged(self.model_wb, self.NC_SPREAD_RENEGO)

        """ MODELE RARN"""
        self.marge_md = ex.get_dataframe_from_range(self.model_wb, self.NC_MARGE_MODELE)
        self.marge_md = self.parse_marge_model(self.marge_md)

        self.param_rarn_md = ex.get_dataframe_from_range(self.model_wb, self.NC_PARAM_RARN)
        self.parse_rarn_params(self.param_rarn_md)

        self.tx_floor = ex.get_value_from_named_ranged(self.model_wb, self.NC_TX_FLOOR)

        coeff_rarn_md = ex.get_dataframe_from_range(self.model_wb, self.NC_COEFF_RARN)
        self.parse_coeff_rarn(coeff_rarn_md)

        """ TYPE MODELE """
        self.type_modele = ex.get_value_from_named_ranged(self.model_wb, self.NC_TYPE_MODEL)
        self.is_user_mod = ("UTILISATEUR" in str(self.type_modele).upper().strip())

        """ TX RA/RN USER"""
        if self.is_user_mod:
            tx_rarn_user = ex.get_dataframe_from_range(self.model_wb, self.NC_TX_RARN_USER)
            tx_ra_usr = np.array(tx_rarn_user.iloc[0, 1:]).reshape(1, tx_rarn_user.shape[1] - 1)
            tx_rn_usr = np.array(tx_rarn_user.iloc[1, 1:]).reshape(1, tx_rarn_user.shape[1] - 1)
            tx_ra_usr = np.insert(tx_ra_usr, 0, 0, axis=1)
            tx_rn_usr = np.insert(tx_rn_usr, 0, 0, axis=1)
            if self.nb_mois_proj > tx_ra_usr.shape[1]:
                tx_ra_usr = np.hstack(
                    (tx_ra_usr, np.tile(tx_ra_usr[:, [-1]], self.nb_mois_proj - tx_ra_usr.shape[1])))
                tx_rn_usr = np.hstack(
                    (tx_rn_usr, np.tile(tx_rn_usr[:, [-1]], self.nb_mois_proj - tx_ra_usr.shape[1])))

            self.tx_ra_usr = tx_ra_usr.astype(np.float)
            self.tx_rn_usr = tx_rn_usr.astype(np.float)

        """ LOAD BACKTEST PARAMS"""
        self.load_backtest_params()

    def load_ra_rn_p_cat_model(self):
        """ Modèles RA/RN constants """
        cle = [self.NC_MOD_CTRT, self.NC_MOD_ETAB, self.NC_MOD_RATE_TYPE, self.NC_MOD_MATUR,
               self.NC_MOD_BASSIN, self.NC_MOD_MARCHE]

        ra_rn_pcat_liq_model = ex.get_dataframe_from_range(self.model_wb, self.NC_RARN_PCAT_LIQ)
        ra_rn_pcat_liq_model = ut.explode_dataframe(ra_rn_pcat_liq_model, [self.NC_MOD_ETAB, self.NC_MOD_CTRT,
                                                                      self.NC_MOD_RATE_TYPE, self.NC_MOD_MARCHE, self.NC_MOD_MATUR])
        ra_rn_pcat_liq_model['new_key'] = ra_rn_pcat_liq_model[cle].apply(lambda row: '$'.join(row.values.astype(str)), axis=1)
        self.ra_rn_pcat_liq_model = ra_rn_pcat_liq_model.drop_duplicates(subset=cle).set_index('new_key').drop(columns=cle, axis=1).copy()
        self.type_ra_pcat_liq = ex.get_value_from_named_ranged(self.model_wb, self.NC_TYPE_RA_PCAT_LIQ)
        self.type_rn_pcat_liq = ex.get_value_from_named_ranged(self.model_wb, self.NC_TYPE_RN_PCAT_LIQ)

        ra_rn_pcat_tx_model = ex.get_dataframe_from_range(self.model_wb, self.NC_RARN_PCAT_TX)
        ra_rn_pcat_tx_model = ut.explode_dataframe(ra_rn_pcat_tx_model, [self.NC_MOD_ETAB, self.NC_MOD_CTRT,
                                                                      self.NC_MOD_RATE_TYPE, self.NC_MOD_MARCHE, self.NC_MOD_MATUR])
        ra_rn_pcat_tx_model['new_key'] = ra_rn_pcat_tx_model[cle].apply(lambda row: '$'.join(row.values.astype(str)), axis=1)
        self.ra_rn_pcat_tx_model = ra_rn_pcat_tx_model.drop_duplicates(subset=cle).set_index('new_key').drop(columns=cle, axis=1).copy()
        self.type_ra_pcat_tx = ex.get_value_from_named_ranged(self.model_wb, self.NC_TYPE_RA_PCAT_TX)
        self.type_rn_pcat_tx = ex.get_value_from_named_ranged(self.model_wb, self.NC_TYPE_RN_PCAT_TX)
