import numpy as np
import numpy_groupies as npg
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)


class NMD_PN_PROJ():
    def __init__(self, compiled_indics_pn, compiled_indics_st, key_vars_dic, horizon, cls_nmd, etab,
                 get_flux_pn=False, get_sc_variables=False, custom_report=False, cls_ag=None):
        self.compiled_indics_pn = compiled_indics_pn
        self.compiled_indics_st = compiled_indics_st
        self.key_vars_dic = key_vars_dic
        self.horizon = horizon
        self.cls_nmd = cls_nmd
        self.etab = etab

        self.get_flux_pn = get_flux_pn
        self.data_pn_all = cls_nmd.data_pn_nmd.copy()
        self.nb_proj = horizon
        self.max_pn = min(cls_nmd.max_pn, self.nb_proj)
        self.cls_fields = self.cls_nmd.cls_fields
        self.NC_CONTRAT_CALC = key_vars_dic["CONTRAT"]
        self.NC_PN = "PN"
        self.NC_CONTRAT = "CONTRAT"
        self.POSITIVE_PROF = "PROFIL_POSITIF"
        self.NEGATIVE_PROF = "PROFIL_NEGATIF"
        self.cols_num = ["M%s" % i for i in range(1, self.nb_proj + 1)]
        self.cols_num2 = ["M%s" % i for i in range(1, self.nb_proj)]
        self.cols_num_flux = ["M%s_FLUX" % i for i in range(1, self.nb_proj)]
        self.cls_pa_fields = self.cls_nmd.cls_pa_fields
        self.NC_IND03 = self.cls_pa_fields.NC_PA_IND03
        self.data_ecoul = pd.DataFrame([])
        self.data_flux = pd.DataFrame([])
        self.dic_inds = {}
        self.data_index_all = []
        self.data_out = []
        self.mtx_ef_all = []
        self.mtx_em_all = []
        self.mtx_mni_all = []
        self.mtx_tem_all = []
        self.mtx_tmni_all = []
        self.mtx_flux_all = []
        self.mtx_sc_tx_all = []
        self.mtx_sc_tx_tci_all = []
        self.custom_report = custom_report
        self.get_sc_variables = get_sc_variables
        self.ALLOCATION_KEY_RCO = cls_nmd.ALLOCATION_KEY_RCO
        self.target_mode = self.cls_nmd.target_mode
        self.is_calage = self.cls_nmd.is_calage
        self.target_key_rco = [cls_nmd.ALLOCATION_KEY_RCO]
        self.target_key_ec = [self.cls_pa_fields.NC_PA_ETAB, self.cls_pa_fields.NC_PA_DEVISE,
                               self.cls_pa_fields.NC_PA_CONTRACT_TYPE, self.cls_pa_fields.NC_PA_MARCHE,
                               self.cls_pa_fields.NC_PA_RATE_CODE, self.cls_pa_fields.NC_PA_PALIER]
        self.cls_ag = cls_ag
        self.list_exemptions = ["P-RSLT-FORM1", "P-RSLT-FORM2","P-RSLT-FORM3",
                                "P-RSLT-FORM4","P-RSLT-FORM5","P-RSLT-FORM6", "P-PEL-PN",
                                "P-PEL-C-PN"]

    #@profile
    def calculate_pn_nmd(self):
        logging.getLogger().setLevel(logging.INFO)
        data_pn_all = self.data_pn_all.copy()
        self.format_pn_profiles()
        #filter data pn
        data_pn_all\
            = data_pn_all[data_pn_all[self.cls_pa_fields.NC_PA_INDEX].isin(self.all_pn_profiles[self.POSITIVE_PROF][self.NC_CONTRAT])].copy()
        dic_data_nmd = self.create_groups_by_nb_parts(data_pn_all)

        for key, data_pn in dic_data_nmd.items():
            if len(data_pn) > 0:
                self.get_group_characteritics(data_pn)

                self.get_profiles_from_compiled_indics(data_pn)

                if len(self.data_profile) > 0:
                    self.get_remaining_stock_by_key(data_pn)

                    self.generate_target_mtx(data_pn)

                    self.calc_ecoulement_mtx()

                    if self.get_flux_pn:
                        self.concat_flux_pn(data_pn)

                    if self.custom_report:
                        self.add_indics_to_dic()

                    if self.get_sc_variables:
                        self.concat_mtxs(data_pn)

        if self.get_flux_pn:
            return self.data_flux

        if self.get_sc_variables:
            return self.get_dic_sc_variables()

        if self.custom_report:
            return self.cls_ag

        return None

    def concat_mtxs(self, data_pn):
        filter_rep = self.get_filter_dim_and_market_ag_data(data_pn)
        key_ag = (data_pn.loc[filter_rep, self.cls_pa_fields.NC_PA_INDEX + "_OLD"].values)
        mtx_ef_ag = self.group_data(self.mtx_ef[filter_rep], key_ag)
        mtx_em_ag = self.group_data(self.mtx_em[filter_rep], key_ag)
        mtx_mni_ag = self.group_data(self.mtx_mni[filter_rep], key_ag)
        mtx_mni_tci_ag = self.group_data(self.mtx_mni_tci[filter_rep], key_ag)
        mtx_flux_em_ag = pd.DataFrame(self.mtx_flux_em[filter_rep], index=key_ag).groupby(key_ag).sum()
        data_pn_index = mtx_flux_em_ag.index.values
        mtx_flux_em_ag = mtx_flux_em_ag.values
        if len(self.mtx_em_all) == 0:
            self.mtx_ef_all = mtx_ef_ag
            self.mtx_em_all = mtx_em_ag
            self.mtx_mni_all = mtx_mni_ag
            self.mtx_mni_tci_all = mtx_mni_tci_ag
            self.mtx_flux_all = mtx_flux_em_ag
            self.data_index_all = data_pn_index
        else:
            self.mtx_ef_all = np.vstack([self.mtx_ef_all, mtx_ef_ag])
            self.mtx_em_all = np.vstack([self.mtx_em_all, mtx_em_ag])
            self.mtx_mni_all = np.vstack([self.mtx_mni_all, mtx_mni_ag])
            self.mtx_mni_tci_all = np.vstack([self.mtx_mni_tci_all, mtx_mni_tci_ag])
            self.mtx_flux_all = np.vstack([self.mtx_flux_all, mtx_flux_em_ag])
            self.data_index_all = np.concatenate([self.data_index_all, data_pn_index], axis=0)

    def get_filter_dim_and_market_ag_data(self, data):
        filter_map = (data[self.cls_fields.NC_LDP_DIM6].isnull()) | (data[self.cls_fields.NC_LDP_DIM6].astype(str).str.upper() != "FCT")
        filter_map = filter_map & (data[self.cls_fields.NC_LDP_MARCHE].str.upper() != "MDC")
        return filter_map.values

    def group_data(self, mtx, key_ag):
        list_mtx = []
        for i in range(0, mtx.shape[1]):
            mtx_df = pd.DataFrame(mtx[:, i], index=key_ag)
            mtx_df = mtx_df.groupby(mtx_df.index).sum().values
            list_mtx.append(mtx_df.reshape(mtx_df.shape[0], 1, mtx_df.shape[1]))
        return np.concatenate(list_mtx, axis=1)

    def get_dic_sc_variables(self):
        dic_sc = {}
        dic_sc["mtx_em"] = self.mtx_em_all
        dic_sc["mtx_ef"] = self.mtx_ef_all
        dic_sc["mtx_mni"] = self.mtx_mni_all
        dic_sc["mtx_mni_tci"] = self.mtx_mni_tci_all
        dic_sc["mtx_flux_em"] = self.mtx_flux_all
        dic_sc["data_index"] = self.data_index_all
        return dic_sc

    def concat_flux_pn(self, data_pn):
        data_pn_concat = data_pn.drop(self.cols_num, axis=1, errors='ignore')
        cols_num = ["M" + str(i) for i in range(1, self.max_pn + 1)]
        data_flux = pd.concat(
            [data_pn_concat.reset_index(drop=True)[[self.ALLOCATION_KEY_RCO]].copy(),
             pd.DataFrame(self.mtx_flux_em, columns=cols_num)], axis=1)
        data_flux = data_flux.drop_duplicates()
        if len(self.data_flux) > 0:
            self.data_flux = pd.concat([self.data_flux, data_flux])
        else:
            self.data_flux = data_flux.copy()

    def add_indics_to_dic(self):
        dic_data = {}
        dic_data["data_lem"] = self.reshape_pn_mtx(self.mtx_em)
        dic_data["data_mni"] = self.reshape_pn_mtx(self.mtx_mni)
        dic_data["data_mni_tci"] = self.reshape_pn_mtx(self.mtx_mni_tci)
        dic_data["data_lef"] = self.reshape_pn_mtx(self.mtx_ef)

        for indic in ["lem", "mni", "lef", "mni_tci"]:
            s = dic_data["data_%s" % indic].shape[0]
            dic_data["data_%s" % indic] = np.hstack([np.zeros((s, 1)), dic_data["data_%s" % indic]])
            self.dic_inds[indic] = dic_data["data_%s" % indic]

        self.data_out = self.data_profile.reset_index(drop=True).copy()

        self.cls_ag.store_compiled_indics(self.dic_inds, self.data_out)

    def reshape_pn_mtx(self, pn_mtx):
        pn_mtx = (np.swapaxes(pn_mtx.reshape(pn_mtx.shape[0] // self.nb_rm_groups, self.nb_rm_groups,
                                             self.max_pn, self.nb_proj), axis1=1, axis2=2)
                  .reshape(self.data_profile.shape[0], self.nb_proj))
        return pn_mtx

    def concat_data_qual(self, pn_mtx, ind):
        pn_mtx = pd.concat([self.data_profile.reset_index(drop=True), pd.DataFrame(pn_mtx, columns=self.cols_num2)],
                           axis=1)
        pn_mtx[self.NC_IND03] = ind
        return pn_mtx

    def get_group_characteritics(self, data_pn):
        contrat = data_pn[data_pn[self.cls_pa_fields.NC_PA_INDEX] == data_pn[self.cls_pa_fields.NC_PA_INDEX].iloc[0]].copy()
        self.nb_rm_groups = contrat.groupby([self.cls_pa_fields.NC_PA_INDEX])[
            self.cls_fields.NC_LDP_RM_GROUP].count().max()
        has_volatile_part = data_pn[self.cls_fields.NC_LDP_RM_GROUP].str.upper().str.contains("VOLATILE").any()
        self.nb_vol_parts = 1 if has_volatile_part else 0
        self.is_flow = (data_pn[self.cls_nmd.FLOW_OR_TARGET] == "FLOW").values
        if self.is_calage:
            self.tmp_weights = data_pn[self.cls_nmd.TEMPLATE_WEIGHT_RCO].values
        else:
            self.tmp_weights = np.where(self.is_flow, data_pn[self.cls_nmd.TEMPLATE_WEIGHT_RCO].values,
                                        data_pn[self.cls_nmd.TEMPLATE_WEIGHT_PASS_ALM].values)

        self.parts_weights = data_pn[self.cls_nmd.model_mapper.NC_PRCT_BREAKDOWN].values / 100

        group_ag_number_flow = data_pn[self.target_key_rco].groupby(self.target_key_rco, dropna=False).ngroup().values
        group_ag_number_ec = data_pn[self.target_key_ec].groupby(self.target_key_ec, dropna=False).ngroup().values + np.max(group_ag_number_flow) + 1

        if self.is_calage:
            self.group_ag_number = group_ag_number_flow
        else:
            self.group_ag_number = np.where(~self.is_flow, group_ag_number_ec, group_ag_number_flow)

        self.nb_stable_parts = self.nb_rm_groups - self.nb_vol_parts

        self.nb_c = data_pn.shape[0]

    ######@profile
    def create_groups_by_nb_parts(self, data_nmd_rm):
        dic_data_nmd = {}
        for nb_part in data_nmd_rm["NB_PARTS"].unique():
            for is_volatile in data_nmd_rm[data_nmd_rm["NB_PARTS"] == nb_part]["HAS_VOLATILE"].unique():
                filter_part = (data_nmd_rm["NB_PARTS"] == nb_part) & (data_nmd_rm["HAS_VOLATILE"] == is_volatile)
                dic_data_nmd[str(int(is_volatile)) + "_" + str(int(nb_part))] = data_nmd_rm[filter_part].copy()
        return dic_data_nmd

    ######@profile
    def get_remaining_stock_by_key(self, dyn_pn):
        cols_num = ["M%s" % i for i in range(1, self.nb_proj + 1)]
        lem_stock = self.compiled_indics_st[self.compiled_indics_st[self.NC_IND03] == "B$LEM"].copy()
        keys = [self.cls_fields.NC_LDP_ETAB, self.cls_fields.NC_LDP_CURRENCY,
                self.cls_fields.NC_LDP_CONTRACT_TYPE, self.cls_fields.NC_LDP_MARCHE,
                self.cls_fields.NC_LDP_RATE_CODE, self.cls_fields.NC_LDP_PALIER]

        data_template = self.cls_nmd.cls_nmd_tmp.data_template_mapped.copy()
        keep_cols = list(set(keys + [self.ALLOCATION_KEY_RCO]
                             + self.cls_nmd.ALLOCATION_KEY_PASS_ALM))
        data_template = data_template[keep_cols].drop_duplicates().copy().set_index(keys)

        lem_stock_with_key_ag = lem_stock.join(data_template, on=keys)

        if (lem_stock_with_key_ag[self.ALLOCATION_KEY_RCO] ==
            lem_stock_with_key_ag[self.cls_nmd.cls_fields.NC_LDP_ETAB]).any():
            raise ValueError("             Problem merging Template and STOCK")

        merge_key_all_flow = self.target_key_rco + [self.cls_fields.NC_LDP_RM_GROUP]
        merge_key_all_ec = self.target_key_ec + [self.cls_fields.NC_LDP_RM_GROUP]

        lem_stock_by_key_by_part_flow \
            = lem_stock_with_key_ag[merge_key_all_flow + cols_num].groupby(merge_key_all_flow, as_index=False, dropna=False).sum()

        lem_stock_by_key_by_part_ec \
            = lem_stock_with_key_ag[merge_key_all_ec + cols_num].groupby(merge_key_all_ec, as_index=False, dropna=False).sum()

        lem_stock_by_key_by_part = dyn_pn[merge_key_all_flow].merge(lem_stock_by_key_by_part_flow, on=merge_key_all_flow, how="left")
        lem_stock_by_key_by_part_ec = dyn_pn[merge_key_all_ec].merge(lem_stock_by_key_by_part_ec, on=merge_key_all_ec, how="left")
        _n = lem_stock_by_key_by_part.shape[0]

        if self.is_calage:
            lem_stock_by_key_by_part[cols_num] = lem_stock_by_key_by_part[cols_num].values
        else:
            lem_stock_by_key_by_part[cols_num] = np.where(self.is_flow.reshape(_n, 1),
                                                          lem_stock_by_key_by_part[cols_num].values,
                                                          lem_stock_by_key_by_part_ec[cols_num].values)

        if lem_stock_by_key_by_part["M1"].isnull().any():
            contrats_warning = lem_stock_by_key_by_part[lem_stock_by_key_by_part["M1"].isnull()][
                merge_key_all_flow].drop_duplicates().values.tolist()
            contrats_warning = [x for x in contrats_warning if all(y not in x[0] for y in self.list_exemptions)]
            if len(contrats_warning) > 0:
                logger.warning("             Il y a des clés de PN absentes dans le stock : %s" % contrats_warning)

        lem_stock_by_key_by_part = lem_stock_by_key_by_part[cols_num].fillna(0).copy()

        self.lem_stock_by_key_by_part = lem_stock_by_key_by_part.values

    #####@profile
    def format_pn_profiles(self):
        pn_profiles = {}
        self.compiled_indics_pn["FILTER"] = self.compiled_indics_pn[self.NC_CONTRAT_CALC].str.contains("NEG", regex=False)
        dict_ind = dict(tuple(self.compiled_indics_pn.groupby("FILTER")))
        pn_profiles[self.POSITIVE_PROF] = dict_ind[False]
        pn_profiles[self.NEGATIVE_PROF] = dict_ind[True]

        pn_profiles[self.NEGATIVE_PROF][self.NC_CONTRAT_CALC] = pn_profiles[self.NEGATIVE_PROF][self.NC_CONTRAT_CALC].str.replace("_NEG", "")
        for key in pn_profiles:
            pn_profiles[key][[self.NC_CONTRAT, self.NC_PN]] = pn_profiles[key][self.NC_CONTRAT_CALC].str.split("-").to_list()
            pn_profiles[key][self.NC_PN] = pn_profiles[key][self.NC_PN].str.replace("PN", "").astype(int)
            pn_profiles[key] = pn_profiles[key].sort_values([self.NC_CONTRAT, self.NC_PN])
        self.all_pn_profiles = pn_profiles

    #####@profile
    def get_profiles_from_compiled_indics(self, data_pn):
        index_data_pn = data_pn[self.cls_pa_fields.NC_PA_INDEX].unique().tolist()
        all_pn_profiles_filtered = {}
        for key in self.all_pn_profiles.keys():
            self.all_pn_profiles[key]["CONTRAT"] = self.all_pn_profiles[key]["CONTRAT"].astype("category")
            all_pn_profiles_filtered[key] = self.all_pn_profiles[key][self.all_pn_profiles[key]["CONTRAT"].isin(index_data_pn)].copy()

        ht = all_pn_profiles_filtered[self.POSITIVE_PROF].shape[0] // len(all_pn_profiles_filtered[self.POSITIVE_PROF][self.NC_IND03].unique())
        dic_pn_profiles = {}
        dic_pn_profiles[self.POSITIVE_PROF] = {}
        dic_pn_profiles[self.NEGATIVE_PROF] = {}

        name_profiles = ["profil_lem_stable", "profil_lem_vol", "profil_mni_stable", "profil_mni_vol"]
        name_profiles = name_profiles + ["profil_mni_tci_stable", "profil_mni_tci_vol",
                                         "profil_lef_stable", "profil_lef_vol"]

        name_inds = ["M$LEM_STABLE", "N$LEM_VOL", "P$LMN_STABLE", "Q$LMN_VOL", "V$LMN_FTP_STABLE", "W$LMN_FTP_VOL",
                     "Z$LEF_STABLE", "ZA$LEF_VOL"]

        tmp_shape = ht // self.nb_rm_groups // self.max_pn, self.max_pn, self.nb_rm_groups, self.nb_proj
        new_shape = (ht // self.max_pn, self.max_pn, self.nb_proj)

        for key in dic_pn_profiles.keys():
            dict_ind = dict(tuple(all_pn_profiles_filtered[key].groupby(self.NC_IND03)))
            for name_prof, name_ind in zip(name_profiles, name_inds):
                profile = dict_ind[name_ind][self.cols_num].values
                profile = np.swapaxes(profile.reshape(tmp_shape), axis1=1, axis2=2).reshape(new_shape)
                dic_pn_profiles[key][name_prof] = profile

        data = all_pn_profiles_filtered[self.POSITIVE_PROF][
            all_pn_profiles_filtered[self.POSITIVE_PROF][self.NC_IND03] == "M$LEM_STABLE"].copy()
        data = data[[x for x in data.columns if x not in self.cols_num and x != "M0"]]

        self.dic_pn_profiles = dic_pn_profiles
        self.data_profile = data

    ######@profile
    def generate_target_mtx(self, data_pn):

        cols_flux = ["M%s_FLUX" % i for i in range(1, self.max_pn + 1)]
        new_fl = data_pn[cols_flux].astype(float).fillna(0)
        ht = new_fl.shape[0]
        new_fl_all = np.vstack([np.triu(np.ones((self.max_pn, self.nb_proj)))] * ht).reshape(ht, self.max_pn,
                                                                                                 self.nb_proj)
        new_fl_all = new_fl_all * (np.array(new_fl)[:, :self.max_pn].reshape(ht, self.max_pn, 1))
        new_fl_all = new_fl_all.reshape(ht, self.max_pn, self.nb_proj)
        self.flux_calage_mtx = new_fl_all

        cols_target = ["M%s" % i for i in range(1, self.max_pn + 1)]
        new_ec = data_pn[cols_target].astype(float).fillna(0)
        ht = new_ec.shape[0]
        new_ec_all = np.vstack([np.triu(np.ones((self.max_pn, self.nb_proj)))] * ht).reshape(ht, self.max_pn,
                                                                                                 self.nb_proj)
        new_ec_all = new_ec_all * (np.array(new_ec)[:, :self.max_pn].reshape(ht, self.max_pn, 1))
        new_ec_all = new_ec_all.reshape(ht, self.max_pn, self.nb_proj)
        self.target_mtx = new_ec_all

        cols_mult = ["M%s" % i + "_COEFF_MULT" for i in range(1, self.max_pn + 1)]
        new_ec_mult = data_pn[cols_mult].astype(float).fillna(0)
        ht = new_ec_mult.shape[0]
        new_ec_all_mult = np.vstack([np.triu(np.ones((self.max_pn, self.nb_proj)))] * ht).reshape(ht, self.max_pn,
                                                                                                      self.nb_proj)
        new_ec_all_mult = new_ec_all_mult * (np.array(new_ec_mult)[:, :self.max_pn].reshape(ht, self.max_pn, 1))
        new_ec_all_mult = new_ec_all_mult.reshape(ht, self.max_pn, self.nb_proj)

        self.target_mtx_multiplier = new_ec_all_mult

    ######@profile
    def calc_ecoulement_mtx(self):
        ht = self.nb_c
        lg = self.nb_proj
        nb_pn = self.max_pn
        nb_rmg = self.nb_rm_groups
        nb_st = self.nb_stable_parts

        mtx_ef = np.zeros((ht, nb_pn, lg))
        mtx_em = np.zeros((ht, nb_pn, lg))
        mtx_mni = np.zeros((ht, nb_pn, lg))
        mtx_mni_tci = np.zeros((ht, nb_pn, lg ))
        #mtx_tem = np.zeros((ht, nb_pn, lg))
        #mtx_tmni = np.zeros((ht, nb_pn, lg))
        #mtx_tmni_tci = np.zeros((ht, nb_pn, lg))

        pn_to_generate = np.zeros((ht, nb_pn))
        pn_to_generate_target = np.zeros((ht, nb_pn))
        pn_to_generate_flow= np.zeros((ht, nb_pn))
        pn_to_generate_by_template = np.zeros((ht, nb_pn))

        _is = np.mod(np.arange(0, len(self.lem_stock_by_key_by_part)), nb_rmg) != nb_st
        is_stable = len(_is[_is]) > 0
        is_vol = self.nb_vol_parts > 0

        profil_pos_st = self.dic_pn_profiles[self.POSITIVE_PROF]["profil_lem_stable"]
        profil_pos_vol = self.dic_pn_profiles[self.POSITIVE_PROF]["profil_lem_vol"]
        profil_neg_st = self.dic_pn_profiles[self.NEGATIVE_PROF]["profil_lem_stable"]
        profil_neg_vol = self.dic_pn_profiles[self.NEGATIVE_PROF]["profil_lem_vol"]

        profil_ef_pos_st = self.dic_pn_profiles[self.POSITIVE_PROF]["profil_lef_stable"]
        profil_ef_pos_vol = self.dic_pn_profiles[self.POSITIVE_PROF]["profil_lef_vol"]
        profil_ef_neg_st = self.dic_pn_profiles[self.NEGATIVE_PROF]["profil_lef_stable"]
        profil_ef_neg_vol = self.dic_pn_profiles[self.NEGATIVE_PROF]["profil_lef_vol"]

        profil_mni_pos_st = self.dic_pn_profiles[self.POSITIVE_PROF]["profil_mni_stable"]
        profil_mni_pos_vol = self.dic_pn_profiles[self.POSITIVE_PROF]["profil_mni_vol"]
        profil_mni_neg_st = self.dic_pn_profiles[self.NEGATIVE_PROF]["profil_mni_stable"]
        profil_mni_neg_vol = self.dic_pn_profiles[self.NEGATIVE_PROF]["profil_mni_vol"]

        profil_mni_tci_pos_st = self.dic_pn_profiles[self.POSITIVE_PROF]["profil_mni_tci_stable"]
        profil_mni_tci_pos_vol = self.dic_pn_profiles[self.POSITIVE_PROF]["profil_mni_tci_vol"]
        profil_mni_tci_neg_st = self.dic_pn_profiles[self.NEGATIVE_PROF]["profil_mni_tci_stable"]
        profil_mni_tci_neg_vol = self.dic_pn_profiles[self.NEGATIVE_PROF]["profil_mni_tci_vol"]

        for i in range(0, nb_pn):
            remaining_pn_by_key = self.get_previous_pn_encours_by_template_key(self.group_ag_number, mtx_em, i)
            current_total \
                = (self.lem_stock_by_key_by_part[:, i].reshape(ht // nb_rmg, nb_rmg).sum(axis=1).repeat(nb_rmg, axis=0)
                   + remaining_pn_by_key)

            pn_to_generate_target[:, i] = (self.target_mtx[:, i, i] - current_total)

            pn_to_generate_flow[:, i] = self.flux_calage_mtx[:, i, i]
            new_target = pn_to_generate_flow[:, i] + current_total

            pn_to_generate[:, i] = np.where(self.is_flow, pn_to_generate_flow[:, i], pn_to_generate_target[:, i])

            pn_to_generate_by_template[:, i] = pn_to_generate[:, i] * self.tmp_weights

            if is_vol:
                target_vol_i = np.where(self.is_flow[~_is], new_target[~_is], self.target_mtx[~_is, i, i])

                pn_to_generate_vol_by_template \
                    = (((target_vol_i * self.parts_weights[~_is] - self.lem_stock_by_key_by_part[~_is, i])
                        * self.tmp_weights[~_is] - mtx_em[~_is, :i, i].sum(axis=1)))

                pn_to_generate_vol_by_template_stable \
                    = pn_to_generate_vol_by_template.repeat(nb_rmg - self.nb_vol_parts, axis=0)

                pn_to_generate_vol_by_template = pn_to_generate_vol_by_template.reshape(ht // nb_rmg, 1)

                mtx_em[~_is, i, i:] = (np.where(pn_to_generate_vol_by_template < 0,
                                                profil_neg_st[~_is, i, i:],
                                                profil_pos_st[~_is, i, i:]) * pn_to_generate_vol_by_template)

                mtx_ef[~_is, i, i:] = (np.where(pn_to_generate_vol_by_template < 0,
                                                profil_ef_neg_st[~_is, i, i:],
                                                profil_ef_pos_st[~_is, i, i:]) * pn_to_generate_vol_by_template)

                mtx_mni[~_is, i, i:] = (np.where(pn_to_generate_vol_by_template < 0,
                                                 profil_mni_neg_st[~_is, i, i:],
                                                 profil_mni_pos_st[~_is, i, i:]) * pn_to_generate_vol_by_template)

                mtx_mni_tci[~_is, i, i:] = (np.where(pn_to_generate_vol_by_template < 0,
                                                     profil_mni_tci_neg_st[~_is, i, i:],
                                                     profil_mni_tci_pos_st[~_is, i, i:])
                                            * pn_to_generate_vol_by_template)

                sum_stable_parts = (1 - self.parts_weights[~_is].repeat(nb_st, axis=0))
            else:
                pn_to_generate_vol_by_template_stable = 0
                sum_stable_parts = 1

            if is_stable:
                pn_to_generate_stable_by_template \
                    = ((pn_to_generate_by_template[_is, i] - pn_to_generate_vol_by_template_stable) *
                       self.parts_weights[_is] / sum_stable_parts)

                pn_to_generate_stable_by_template \
                    = pn_to_generate_stable_by_template.reshape((nb_st) * ht // nb_rmg, 1)

                pn_to_generate_vol_by_template_stable \
                    = (pn_to_generate_vol_by_template_stable * self.parts_weights[_is] / sum_stable_parts)

                pn_to_generate_vol_by_template_stable \
                    = pn_to_generate_vol_by_template_stable.reshape((nb_st) * ht // nb_rmg, 1)

                mtx_em[_is, i, i:] \
                    = (np.where(pn_to_generate_stable_by_template < 0, profil_neg_st[_is, i, i:],
                                profil_pos_st[_is, i, i:]) * pn_to_generate_stable_by_template)

                mtx_em[_is, i, i:] = mtx_em[_is, i, i:] + (
                        np.where(pn_to_generate_stable_by_template < 0, profil_neg_vol[_is, i, i:],
                                 profil_pos_vol[_is, i, i:]) * pn_to_generate_vol_by_template_stable)

                mtx_ef[_is, i, i:] \
                    = (np.where(pn_to_generate_stable_by_template < 0, profil_ef_neg_st[_is, i, i:],
                                profil_ef_pos_st[_is, i, i:]) * pn_to_generate_stable_by_template)

                mtx_ef[_is, i, i:] = mtx_ef[_is, i, i:] + (
                        np.where(pn_to_generate_stable_by_template < 0, profil_ef_neg_vol[_is, i, i:],
                                 profil_ef_pos_vol[_is, i, i:]) * pn_to_generate_vol_by_template_stable)

                mtx_mni[_is, i, i:] \
                    = (np.where(pn_to_generate_stable_by_template < 0, profil_mni_neg_st[_is, i, i:],
                                profil_mni_pos_st[_is, i, i:]) * pn_to_generate_stable_by_template)

                mtx_mni[_is, i, i:] = mtx_mni[_is, i, i:] + (
                        np.where(pn_to_generate_stable_by_template < 0, profil_mni_neg_vol[_is, i, i:],
                                 profil_mni_pos_vol[_is, i, i:]) * pn_to_generate_vol_by_template_stable)

                mtx_mni_tci[_is, i, i:] \
                    = (np.where(pn_to_generate_stable_by_template < 0, profil_mni_tci_neg_st[_is, i, i:],
                                profil_mni_tci_pos_st[_is, i, i:]) * pn_to_generate_stable_by_template)

                mtx_mni_tci[_is, i, i:] = mtx_mni_tci[_is, i, i:] + (
                        np.where(pn_to_generate_stable_by_template < 0, profil_mni_tci_neg_vol[_is, i, i:],
                                 profil_mni_tci_pos_vol[_is, i, i:]) * pn_to_generate_vol_by_template_stable)

            mtx_ef[:, i, i:] = mtx_ef[:, i, i:] * self.target_mtx_multiplier[:, i, i:]
            mtx_em[:, i, i:] = mtx_em[:, i, i:] * self.target_mtx_multiplier[:, i, i:]
            mtx_mni[:, i, i:] = mtx_mni[:, i, i:] * self.target_mtx_multiplier[:, i, i:]
            mtx_mni_tci[:, i, i:] = mtx_mni_tci[:, i, i:] * self.target_mtx_multiplier[:, i, i:]


        self.mtx_ef = mtx_ef
        self.mtx_em = mtx_em
        self.mtx_mni = mtx_mni
        self.mtx_mni_tci = mtx_mni_tci
        self.mtx_flux_em = pn_to_generate

    def get_previous_pn_encours_by_template_key(self, key_template, mtx_em, i):
        if len(mtx_em[:, :i, i]) > 0:
            sum_by_group = npg.aggregate(key_template, mtx_em[:, :i, i].sum(axis=1), func='sum')
            sum_dispatched = sum_by_group[key_template]
        else:
            sum_dispatched = 0
        return sum_dispatched
