import numpy as np
import pandas as pd
import tempfile
import os
import pickle
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import dask.dataframe as dd


class Agregation():
    list_indics = ["lef", "lem", "tef", "tem", "mni", "mni_tci", "mni_gptx", "mni_tci_gptx",
                   "effet_RA", "effet_RN", "sc_rates", "sc_rates_tci", "lem_stable", "lem_volatile",
                   "mni_stable", "mni_volatile", "tem_stable", "tem_volatile",
                   "mni_gptx_stable", "mni_gptx_volatile", "mni_tci_stable", "mni_tci_volatile",
                   "mni_tci_gptx_stable", "mni_tci_gptx_volatile", "lef_stable", "lef_volatile",
                   "lem_statique", "lem_renego", "mni_statique", "mni_renego", "sc_rates_reneg",
                   "sc_rates_statique", "tx_RA", "tx_RN"]

    list_names = ["A$LEF", "B$LEM", "C$TEF", "D$TEM", "E$LMN", "F$LMN_FTP", "G$LMN_GPTX", "H$LMN_FTP_GPTX",
                  "K$EFFET_RA", "L$EFFET_RN", "I$SC_RATES", "J$SC_RATES_TCI", "M$LEM_STABLE", "N$LEM_VOL",
                  "P$LMN_STABLE", "Q$LMN_VOL", "R$TEM_STABLE", "S$TEM_VOL",
                  "T$LMN_GPTX_STABLE", "U$LMN_GPTX_VOL", "V$LMN_FTP_STABLE", "W$LMN_FTP_VOL",
                  "X$LMN_FTP_GPTX_STABLE", "Y$LMN_FTP_GPTX_VOL", "Z$LEF_STABLE", "ZA$LEF_VOL",
                  "ZB$LEM_STATIQUE", "ZC$LEM_RENEGO", "ZE$LMN_STATIQUE", "ZF$LMN_RENEGO", "ZH$SC_TX_RENEG",
                  "ZI$SC_TX_STATIQUE", "ZJ$TX_RA", "ZJK$TX_RN"]

    pass_alm_rco_map_indics = {x: y for x, y in zip(list_indics, list_names)}

    def __init__(self, agregation_level, cls_fields, nb_mois_proj, output_mode="dump"):
        self.cls_fields = cls_fields
        self.compiled_indics = []
        self.get_agregation_level(agregation_level)
        self.t = nb_mois_proj
        self.temp_files_name = []
        self.output_mode = output_mode
        if self.output_mode in ["dump", "all"]:
            self.create_temp_directory()
        self.num_cols = ["M" + str(i) for i in range(0, self.t + 1)]

    def create_temp_directory(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def dump_in_temp_dir(self, data):
        name_file = "AG_DATA_SIMUL_%s.pi" % (len(self.temp_files_name) + 1)
        name_file = os.path.join(self.temp_dir.name, name_file)
        with open(name_file, 'wb') as handle:
            pickle.dump(data, handle)
        self.temp_files_name.append(name_file)

    def store_compiled_indics(self, dic_inds, data_ldp, data_optional=[]):
        results_all_ind = self.concat_indics(dic_inds)

        if len(data_optional) > 0:
            keep_other = [x for x in self.keep_vars if x in data_optional]
            data_other = np.repeat(data_optional[keep_other].values, int(results_all_ind.shape[0] /
                                                                         data_optional.shape[0]), axis=0)
            data_other = pd.DataFrame(data_other, columns=keep_other)

        keep_main = [x for x in self.keep_vars if x in data_ldp]
        data_main = np.repeat(data_ldp[keep_main].values, int(results_all_ind.shape[0] /
                                                              data_ldp.shape[0]), axis=0)
        data_main = pd.DataFrame(data_main, columns=keep_main)

        if len(data_optional) > 0:
            results_ag = pd.concat([data_main, data_other, results_all_ind], axis=1)
        else:
            results_ag = pd.concat([data_main, results_all_ind], axis=1)

        #dask_results_ag = dd.from_pandas(results_ag, npartitions=4)
        #results_ag_dask = (dask_results_ag.groupby(self.ag_vars, dropna=False).sum(numeric_only=True).reset_index()).compute()
        results_ag_dask = results_ag.groupby(self.ag_vars, dropna=False, as_index=False).sum(numeric_only=True)
        if self.output_mode in ["dump", "all"]:
            self.dump_in_temp_dir(results_ag_dask)

        if self.output_mode in ["dataframe", "all"]:
            self.compiled_indics.append(results_ag_dask.copy())

        results_all_ind = 0
        del results_all_ind

    def final_wrap(self):
        if len(self.compiled_indics) > 0:
            data_compil = pd.concat(self.compiled_indics)
            #dask_data_compil = dd.from_pandas(data_compil, npartitions=4)
            #self.compiled_indics = (dask_data_compil.groupby(self.ag_vars, dropna=False).sum(numeric_only=True).reset_index()).compute()
            self.compiled_indics = data_compil.groupby(self.ag_vars, dropna=False, as_index=False).sum(numeric_only=True)

    def get_agregation_level(self, agregation_level):
        if agregation_level == "AG":
            self.keep_vars = [self.cls_fields.NC_LDP_BASSIN, self.cls_fields.NC_LDP_CONTRACT_TYPE,
                              self.cls_fields.NC_LDP_MATUR,
                              self.cls_fields.NC_LDP_CURRENCY,
                              self.cls_fields.NC_LDP_MARCHE, self.cls_fields.NC_LDP_GESTION,
                              self.cls_fields.NC_LDP_RATE_TYPE, self.cls_fields.NC_LDP_PALIER,
                              self.cls_fields.NC_LDP_ETAB, self.cls_fields.NC_LDP_BUY_SELL,
                              self.cls_fields.NC_LDP_RATE_CODE, self.cls_fields.NC_LDP_DIM6]

            self.keep_vars_dic = {"CONTRACT_TYPE": self.cls_fields.NC_LDP_CONTRACT_TYPE,
                                  "MATUR": self.cls_fields.NC_LDP_MATUR,
                                  "CUR": self.cls_fields.NC_LDP_CURRENCY,
                                  "MARCHE": self.cls_fields.NC_LDP_MARCHE, "GESTION": self.cls_fields.NC_LDP_GESTION,
                                  "INDEX": self.cls_fields.NC_LDP_RATE_TYPE,
                                  "PALIER": self.cls_fields.NC_LDP_PALIER, "ETAB": self.cls_fields.NC_LDP_ETAB,
                                  "BASSIN": self.cls_fields.NC_LDP_BASSIN,
                                  "BUY_SELL": self.cls_fields.NC_LDP_BUY_SELL,
                                  "RATE CODE": self.cls_fields.NC_LDP_RATE_CODE,
                                  }

        elif agregation_level == "DT":
            self.keep_vars = [self.cls_fields.NC_LDP_BASSIN, self.cls_fields.NC_LDP_CONTRACT_TYPE,
                              self.cls_fields.NC_LDP_MATUR,
                              self.cls_fields.NC_LDP_CURRENCY,
                              self.cls_fields.NC_LDP_MARCHE, self.cls_fields.NC_LDP_GESTION,
                              self.cls_fields.NC_LDP_RATE_TYPE, self.cls_fields.NC_LDP_PALIER,
                              self.cls_fields.NC_LDP_ETAB,
                              self.cls_fields.NC_LDP_BUY_SELL,
                              self.cls_fields.NC_LDP_RATE_CODE, self.cls_fields.NC_LDP_CONTRAT,
                              self.cls_fields.NC_LDP_DIM6]

            self.keep_vars_dic = {"CONTRACT_TYPE": self.cls_fields.NC_LDP_CONTRACT_TYPE,
                                  "MATUR": self.cls_fields.NC_LDP_MATUR,
                                  "CUR": self.cls_fields.NC_LDP_CURRENCY,
                                  "MARCHE": self.cls_fields.NC_LDP_MARCHE, "GESTION": self.cls_fields.NC_LDP_GESTION,
                                  "INDEX": self.cls_fields.NC_LDP_RATE_TYPE,
                                  "PALIER": self.cls_fields.NC_LDP_PALIER, "ETAB": self.cls_fields.NC_LDP_ETAB,
                                  "BASSIN": self.cls_fields.NC_LDP_BASSIN,
                                  "BUY_SELL": self.cls_fields.NC_LDP_BUY_SELL,
                                  "RATE CODE": self.cls_fields.NC_LDP_RATE_CODE,
                                  "CONTRAT": self.cls_fields.NC_LDP_CONTRAT}


        elif agregation_level == "DT_DATES":
            self.keep_vars = [self.cls_fields.NC_LDP_BASSIN, self.cls_fields.NC_LDP_CONTRACT_TYPE,
                              self.cls_fields.NC_LDP_MATUR,
                              self.cls_fields.NC_LDP_CURRENCY,
                              self.cls_fields.NC_LDP_MARCHE, self.cls_fields.NC_LDP_GESTION,
                              self.cls_fields.NC_LDP_RATE_TYPE, self.cls_fields.NC_LDP_PALIER,
                              self.cls_fields.NC_LDP_ETAB,
                              self.cls_fields.NC_LDP_BUY_SELL,
                              self.cls_fields.NC_LDP_RATE_CODE, self.cls_fields.NC_LDP_DIM6,
                              self.cls_fields.NC_LDP_MATUR_DATE, self.cls_fields.NC_LDP_VALUE_DATE,
                              self.cls_fields.NC_LDP_CONTRAT]

            self.keep_vars_dic = {"CONTRACT_TYPE": self.cls_fields.NC_LDP_CONTRACT_TYPE,
                                  "MATUR": self.cls_fields.NC_LDP_MATUR,
                                  "CUR": self.cls_fields.NC_LDP_CURRENCY,
                                  "MARCHE": self.cls_fields.NC_LDP_MARCHE, "GESTION": self.cls_fields.NC_LDP_GESTION,
                                  "INDEX": self.cls_fields.NC_LDP_RATE_TYPE,
                                  "PALIER": self.cls_fields.NC_LDP_PALIER, "ETAB": self.cls_fields.NC_LDP_ETAB,
                                  "BASSIN": self.cls_fields.NC_LDP_BASSIN,
                                  "BUY_SELL": self.cls_fields.NC_LDP_BUY_SELL,
                                  "RATE CODE": self.cls_fields.NC_LDP_RATE_CODE,
                                  "MATURITY_DATE": self.cls_fields.NC_LDP_MATUR_DATE,
                                  "VALUE_DATE": self.cls_fields.NC_LDP_VALUE_DATE
                , "CONTRAT": self.cls_fields.NC_LDP_CONTRAT}

        elif agregation_level == "NMD_DT":
            self.keep_vars = [self.cls_fields.NC_LDP_BASSIN, self.cls_fields.NC_LDP_CONTRACT_TYPE,
                              self.cls_fields.NC_LDP_MATUR,
                              self.cls_fields.NC_LDP_CURRENCY,
                              self.cls_fields.NC_LDP_MARCHE, self.cls_fields.NC_LDP_GESTION,
                              self.cls_fields.NC_LDP_RATE_TYPE, self.cls_fields.NC_LDP_PALIER,
                              self.cls_fields.NC_LDP_ETAB,
                              self.cls_fields.NC_LDP_BUY_SELL,
                              self.cls_fields.NC_LDP_RATE_CODE, self.cls_fields.NC_LDP_CONTRAT,
                              self.cls_fields.NC_LDP_RM_GROUP,
                              self.cls_fields.NC_LDP_DIM6]

            self.keep_vars_dic = {"CONTRACT_TYPE": self.cls_fields.NC_LDP_CONTRACT_TYPE,
                                  "MATUR": self.cls_fields.NC_LDP_MATUR,
                                  "CUR": self.cls_fields.NC_LDP_CURRENCY,
                                  "MARCHE": self.cls_fields.NC_LDP_MARCHE, "GESTION": self.cls_fields.NC_LDP_GESTION,
                                  "INDEX": self.cls_fields.NC_LDP_RATE_TYPE,
                                  "PALIER": self.cls_fields.NC_LDP_PALIER, "ETAB": self.cls_fields.NC_LDP_ETAB,
                                  "BASSIN": self.cls_fields.NC_LDP_BASSIN,
                                  "BUY_SELL": self.cls_fields.NC_LDP_BUY_SELL,
                                  "RATE CODE": self.cls_fields.NC_LDP_RATE_CODE,
                                  "CONTRAT": self.cls_fields.NC_LDP_CONTRAT,
                                  "RM_GROUP": self.cls_fields.NC_LDP_RM_GROUP}


        elif agregation_level == "NMD_TEMPLATE":
            self.keep_vars = [self.cls_fields.NC_LDP_BASSIN, self.cls_fields.NC_LDP_CONTRACT_TYPE,
                              self.cls_fields.NC_LDP_MATUR,
                              self.cls_fields.NC_LDP_CURRENCY,
                              self.cls_fields.NC_LDP_MARCHE, self.cls_fields.NC_LDP_GESTION,
                              self.cls_fields.NC_LDP_RATE_TYPE, self.cls_fields.NC_LDP_PALIER,
                              self.cls_fields.NC_LDP_ETAB, self.cls_fields.NC_LDP_BUY_SELL,
                              self.cls_fields.NC_LDP_RATE_CODE, self.cls_fields.NC_LDP_RM_GROUP,
                              self.cls_fields.NC_LDP_DIM6]

            self.keep_vars_dic = {"CONTRACT_TYPE": self.cls_fields.NC_LDP_CONTRACT_TYPE,
                                  "MATUR": self.cls_fields.NC_LDP_MATUR,
                                  "CUR": self.cls_fields.NC_LDP_CURRENCY,
                                  "MARCHE": self.cls_fields.NC_LDP_MARCHE, "GESTION": self.cls_fields.NC_LDP_GESTION,
                                  "INDEX": self.cls_fields.NC_LDP_RATE_TYPE,
                                  "PALIER": self.cls_fields.NC_LDP_PALIER, "ETAB": self.cls_fields.NC_LDP_ETAB,
                                  "BASSIN": self.cls_fields.NC_LDP_BASSIN,
                                  "BUY_SELL": self.cls_fields.NC_LDP_BUY_SELL,
                                  "RATE CODE": self.cls_fields.NC_LDP_RATE_CODE,
                                  "RM_GROUP": self.cls_fields.NC_LDP_RM_GROUP}

        self.ag_vars = self.keep_vars + [pa.NC_PA_IND03]

    def concat_indics(self, dic_inds):
        list_names = [self.pass_alm_rco_map_indics[x] for x in self.list_indics if x in dic_inds]
        list_num_vars = [dic_inds[ind] for ind in self.list_indics if ind in dic_inds]

        s = list_num_vars[0].shape[0]
        p = list_num_vars[0].shape[1]
        self.nb_indics = len(list_names)
        data_num = np.stack(list_num_vars, axis=1).reshape(s * self.nb_indics, p)

        if data_num.shape[1] < len(self.num_cols):
            data_num = np.column_stack(
                [data_num, np.zeros((data_num.shape[0], len(self.num_cols) - data_num.shape[1]))])

        data_num = pd.DataFrame(data_num, columns=self.num_cols)
        data_num[pa.NC_PA_IND03] = np.tile(np.array(list_names), s)

        return data_num
