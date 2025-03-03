import pandas as pd
from datetime import datetime
import modules.alim.parameters.ONEY_SOCFIM_params as os_p
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import modules.alim.parameters.general_parameters as gp
import utils.general_utils as gu
import modules.alim.formating_service.stock_formating_service.stock_common_formating as init_f_gen
from modules.alim.formating_service.stock_formating_service.NTX_SEF.initial_formating import NTX_SEF_Formater
import logging
import numpy as np
logger = logging.getLogger(__name__)

class ONEY_Formater():
    def __init__(self, cls_users_params):
        self.up = cls_users_params
        self.ntx_f = NTX_SEF_Formater(cls_users_params)

    def format_hors_bilan(self, data):
        HB = (data[pa.NC_PA_BILAN] == "HORS BILAN")

        HB_ACTIF = (HB) & ((data[gp.NC_CONTRACT_TEMP].str[-2:] == "-A"))
        data.loc[HB_ACTIF, [pa.NC_PA_BILAN]] = "HB ACTIF"

        HB_PASSIF = (HB) & ((data[gp.NC_CONTRACT_TEMP].str[-2:] == "-P"))
        data.loc[HB_PASSIF, [pa.NC_PA_BILAN]] = "HB PASSIF"

        return data


    def transform_cols(self, data):
        num_cols = [x for x in data.columns.tolist() if "/" in x]
        num_cols = [datetime.strptime(x, '%d/%m/%Y').date() for x in num_cols]
        data.columns = [x for x in data.columns.tolist() if not "/" in x] + num_cols
        return data


    def format_montants_RZO(self, data, prefix):
        data[prefix] = data[prefix].fillna(0)
        data[prefix] = data[prefix].astype(np.float64)
        qual_vars = [x for x in data.columns if x != prefix and x != gp.NC_DATE_MONTANT]
        data[qual_vars] = data[qual_vars].fillna("-")
        data = data.pivot_table(index=qual_vars, columns=[gp.NC_DATE_MONTANT], values=prefix, aggfunc="sum", fill_value=0)
        data = data.reset_index()
        data = self.transform_cols(data)
        return data.copy()


    def do_preliminary_checks(self, data, file_name):
        try:
            check = datetime.strptime(str(data[os_p.NC_OS_DATA_DAR].iloc[0]), '%d/%m/%Y').date() != self.up.dar.date()
        except:
            check = datetime.strptime(str(data[os_p.NC_OS_DATA_DAR].iloc[0]), '%Y-%m-%d').date() != self.up.dar.date()

        if check:
            raise ValueError(
                "La DAR du fichier " + file_name.split("\\")[-1] + " est différente de la DAR en paramètre")
        return data


    def rename_columns(self, data, prefix):
        data = data.rename(columns={os_p.NC_OS_DATA_DIM1: pa.NC_PA_BILAN, os_p.NC_OS_DATA_CONTRACT: gp.NC_CONTRACT_TEMP})
        data = data.rename(columns={os_p.NC_OS_DATA_DATE: gp.NC_DATE_MONTANT, \
                                    os_p.NC_OS_DATA_FAMILY: pa.NC_PA_MARCHE, os_p.NC_OS_DATA_BOOK: pa.NC_PA_BOOK, \
                                    os_p.NC_OS_DATA_LCR_TIERS: pa.NC_PA_LCR_TIERS})
        data = data.rename(
            columns={os_p.NC_OS_DATA_PALIER: gp.NC_PALIER_TEMP, os_p.NC_OS_DATA_GESTION: gp.NC_GESTION_TEMP, \
                     os_p.NC_OS_DATA_CCY: pa.NC_PA_DEVISE, os_p.NC_OS_DATA_RATEC: pa.NC_PA_RATE_CODE,
                     os_p.NC_ONEY_DATA_MATUR: gp.NC_MATUR_TEMP, os_p.NC_OS_DATA_MONTANT3: prefix})
        if prefix in ["LEM", "TEM"]:
            data = data.rename(columns={os_p.NC_OS_DATA_MONTANT2: prefix, os_p.NC_OS_DATA_MONTANT2 + "_L": prefix, \
                                        os_p.NC_OS_DATA_MONTANT2 + "_R": prefix
                                        })
        else:
            data = data.rename(columns={os_p.NC_OS_DATA_MONTANT1: prefix, os_p.NC_OS_DATA_MONTANT1 + "_L": prefix, \
                                        os_p.NC_OS_DATA_MONTANT1 + "_R": prefix
                                        })

        return data


    def format_cols(self, data):
        if self.up.current_etab == "ONEY":
            data[os_p.NC_ONEY_DATA_MATUR] = data[os_p.NC_ONEY_DATA_MATUR].str.strip()
            data[os_p.NC_ONEY_DATA_MATUR] = data[os_p.NC_ONEY_DATA_MATUR].fillna("-")
        else:
            data[os_p.NC_ONEY_DATA_MATUR] = "-"

        data = gu.force_integer_to_string(data, os_p.NC_OS_DATA_BOOK)

        return data


    def transform_num_columns_names(self, data):
        num_cols = [x for x in data.columns.tolist() if "-" in x]
        num_cols = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S').date() for x in num_cols]
        data.columns = [x for x in data.columns.tolist() if not "-" in x] + num_cols
        return data


    def filter_data(self, data):
        data = data.iloc[np.where(data.notnull().apply(any, axis=1))[0], :]
        return data


    def read_os_format_files(self, file_name, prefix):
        if self.up.current_etab == "ONEY":
            data = pd.read_csv(file_name, delimiter="\t", engine='c', encoding="utf-16", decimal=",", thousands=' ',
                               low_memory=False)
            agreg_vars = os_p.agreg_vars
            cols_to_keep = os_p.cols_to_keep
        else:
            data = pd.read_csv(file_name, delimiter="\t", engine='python', decimal=",", thousands=' ',
                               encoding="ISO-8859-1")
            agreg_vars = os_p.agreg_vars + [pa.NC_PA_LCR_TIERS]
            cols_to_keep = os_p.cols_to_keep + [pa.NC_PA_LCR_TIERS]

        data = init_f_gen.upper_columns_names(data)
        data = self.do_preliminary_checks(data, file_name)
        data = self.filter_data(data)
        data = self.format_cols(data)
        data = self.rename_columns(data, prefix)
        data = init_f_gen.format_paliers(data, gp.NC_PALIER_TEMP)
        data = init_f_gen.format_bilan_column(data)
        data = init_f_gen.select_cols_to_keep(data, cols_to_keep, num_cols=[prefix])
        data = self.format_montants_RZO(data, prefix)
        data, num_cols = init_f_gen.select_num_cols(self.up.current_etab, self.up.dar, data, prefix, force_zero=True)
        data = init_f_gen.upper_non_num_cols_vals(data, num_cols)
        if self.up.current_etab == "SOCFIM":
            data = self.ntx_f.format_hors_bilan(data, num_cols)
        else:
            data = self.format_hors_bilan(data)
            if prefix == "LEF":
                self.data_st_oney = data.copy()

        data = data.set_index(agreg_vars)
        return data


    def add_missing_vars(self, data):
        cols_tiret = [pa.NC_PA_TOP_MNI, pa.NC_PA_PERIMETRE, pa.NC_PA_CUST, pa.NC_PA_METIER, \
                      pa.NC_PA_SOUS_METIER, pa.NC_PA_ZONE_GEO, pa.NC_PA_SOUS_ZONE_GEO]
        if self.up.current_etab == "ONEY":
            cols_tiret = cols_tiret + [pa.NC_PA_LCR_TIERS]
        tirets = pd.DataFrame("-", index=data.index, columns=cols_tiret)
        data = pd.concat([data, tirets], axis=1)

        data[pa.NC_PA_ETAB] = self.up.current_etab

        if not self.up.detail_ig:
            data[gp.NC_GESTION_TEMP] =  "-"

        data = gu.add_constant_new_col_pd(data,\
                                          [pa.NC_PA_SCOPE, pa.NC_PA_LCR_TIERS_SHARE, pa.NC_PA_BASSIN],\
                                          [gp.MNI_AND_LIQ, 100, self.up.current_etab])

        return data.copy()


    def read_and_join_all_files(self):
        i = 0
        for file in self.up.main_files_name:
            logger.info("   Pour l'indicateur " + self.up.prefixes_main_files[i] + ", lecture de : " + file.split("\\")[-1])
            data_prefix = self.read_os_format_files(file, self.up.prefixes_main_files[i])
            if i == 0:
                STOCK_DATA = data_prefix.copy()
            else:
                STOCK_DATA = init_f_gen.append_data(STOCK_DATA.copy(), data_prefix.copy(), self.up.prefixes_main_files[i],
                                                    warning=False)
            i = i + 1

        STOCK_DATA = STOCK_DATA.reset_index().copy()

        STOCK_DATA = init_f_gen.finalize_formatting(STOCK_DATA)

        STOCK_DATA = self.add_missing_vars(STOCK_DATA)

        return STOCK_DATA
