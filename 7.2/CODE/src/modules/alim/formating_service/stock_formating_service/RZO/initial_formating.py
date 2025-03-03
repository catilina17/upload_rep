import pandas as pd
from datetime import datetime
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import modules.alim.parameters.RZO_params as rzo_p
import utils.general_utils as gu
import modules.alim.parameters.general_parameters as gp
import modules.alim.formating_service.stock_formating_service.stock_common_formating as init_f_gen
import logging
import numpy as np
from params import version_params as vp

logger = logging.getLogger(__name__)


class RZO_Formater():
    def __init__(self, cls_usr):
        self.up = cls_usr

    def format_bilan(self, data):
        filters = [(data[pa.NC_PA_BILAN] == "ACTIF") & (data[gp.NC_CONTRACT_TEMP].str[:2] == "HB"), \
                   (data[pa.NC_PA_BILAN] == "PASSIF") & (data[gp.NC_CONTRACT_TEMP].str[:2] == "HB"),
                   data[pa.NC_PA_BILAN] == "ACTIF", data[pa.NC_PA_BILAN] == "PASSIF"]
        data[pa.NC_PA_BILAN] = np.select(filters, ["HB ACTIF", "HB PASSIF", "B ACTIF", "B PASSIF"])

        return data


    def transform_cols(self, data):
        num_cols = [x for x in data.columns.tolist() if "/" in x]
        try:
            num_cols = [datetime.strptime(x, '%d/%m/%Y').date() for x in num_cols]
        except:
            num_cols = [datetime.strptime(x, '%d/%m/%Y %H:%M:%S').date() for x in num_cols]
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
        if datetime.strptime(str(data[rzo_p.NC_RZO_DATA_DAR].iloc[0]), '%Y%m%d').date() != self.up.dar.date():
            raise ValueError("La DAR du fichier " + file_name.split("\\")[-1] + " est différente de la DAR en paramètre")
        return data


    def rename_columns(self, data, prefix):
        data = data.rename(columns={'ï»¿etab'.upper(): pa.NC_PA_ETAB})
        data = data.rename(columns={rzo_p.NC_RZO_DATA_DIM1: pa.NC_PA_BILAN, rzo_p.NC_RZO_DATA_CONTRACT: gp.NC_CONTRACT_TEMP})
        data = data.rename(columns={rzo_p.NC_RZO_DATA_DATE: gp.NC_DATE_MONTANT, \
                                    rzo_p.NC_RZO_DATA_FAMILY: pa.NC_PA_MARCHE, rzo_p.NC_RZO_DATA_BOOK: pa.NC_PA_BOOK})
        data = data.rename(
            columns={rzo_p.NC_RZO_DATA_PALIER: gp.NC_PALIER_TEMP, rzo_p.NC_RZO_DATA_GESTION: gp.NC_GESTION_TEMP, \
                     rzo_p.NC_RZO_DATA_CCY: pa.NC_PA_DEVISE, rzo_p.NC_RZO_DATA_RATEC: pa.NC_PA_RATE_CODE,
                     rzo_p.NC_RZO_DATA_MATUR: gp.NC_MATUR_TEMP, rzo_p.NC_RZO_DATA_MONTANT: prefix, \
                     rzo_p.NC_RZO_DATA_ETAB: pa.NC_PA_ETAB})

        return data


    def format_cols(self, data):
        data.fillna({rzo_p.NC_RZO_DATA_MATUR:"-"}, inplace=True)
        data[rzo_p.NC_RZO_DATA_MATUR] = data[rzo_p.NC_RZO_DATA_MATUR].astype(str).str.strip()
        data[rzo_p.NC_RZO_DATA_MATUR] = data[rzo_p.NC_RZO_DATA_MATUR].replace("", "-")
        if self.up.current_etab != "BPCE":
            data[pa.NC_PA_BOOK] = "-"
        return data


    def transform_num_columns_names(self, data):
        num_cols = [x for x in data.columns.tolist() if "-" in x]
        num_cols = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S').date() for x in num_cols]
        data.columns = [x for x in data.columns.tolist() if not "-" in x] + num_cols
        return data


    def read_rzo_format_files(self, file_name, prefix):
        delimiter = ";" if vp.version_sources == "csv" else "\t"
        decimal = ","
        data = pd.read_csv(file_name, delimiter=delimiter, engine='python', encoding="ISO-8859-1", decimal=decimal)
        data = init_f_gen.upper_columns_names(data)
        data = self.do_preliminary_checks(data, file_name)
        data = self.format_cols(data)
        data = self.rename_columns(data, prefix)
        data = self.format_bilan(data)
        data = init_f_gen.format_paliers(data, gp.NC_PALIER_TEMP)
        data = init_f_gen.format_bilan_column(data)
        data = init_f_gen.select_cols_to_keep(data, rzo_p.cols_to_keep, num_cols=[prefix])
        data = self.format_montants_RZO(data, prefix)
        data, num_cols = init_f_gen.select_num_cols(self.up.current_etab, self.up.dar, data, prefix, force_zero=True)
        data = init_f_gen.upper_non_num_cols_vals(data, num_cols)
        data = data.set_index(rzo_p.agreg_vars)
        return data


    def add_missing_vars(self, data):
        if not self.up.detail_ig:
            data[gp.NC_GESTION_TEMP] = "-"
        cols_tirets = [pa.NC_PA_PERIMETRE, pa.NC_PA_LCR_TIERS, pa.NC_PA_CUST, pa.NC_PA_TOP_MNI, pa.NC_PA_METIER, \
                       pa.NC_PA_SOUS_METIER, pa.NC_PA_ZONE_GEO, pa.NC_PA_SOUS_ZONE_GEO]
        tirets = pd.DataFrame("-", index=data.index, columns=cols_tirets)
        data = pd.concat([data, tirets], axis=1)

        data = gu.add_constant_new_col_pd(data,\
                                          [pa.NC_PA_SCOPE, pa.NC_PA_LCR_TIERS_SHARE, pa.NC_PA_BASSIN],\
                                          [gp.MNI_AND_LIQ, 100, self.up.current_etab])
        return data.copy()


    def read_and_join_all_files(self):
        i = 0
        prefixes_done = []
        for file in self.up.main_files_name:
            logger.info("   Pour l'indicateur " + self.up.prefixes_main_files[i] + ", lecture de : " + file.split("\\")[-1])
            data_prefix = self.read_rzo_format_files(file, self.up.prefixes_main_files[i])
            if i == 0:
                STOCK_DATA = data_prefix.copy()
            else:
                if self.up.prefixes_main_files[i] not in prefixes_done:
                    STOCK_DATA = init_f_gen.append_data(STOCK_DATA.copy(), data_prefix.copy(), self.up.prefixes_main_files[i],
                                                        warning=False)
                else:
                    STOCK_DATA = pd.concat([STOCK_DATA,data_prefix.copy()])
            prefixes_done.append(self.up.prefixes_main_files[i])
            i = i + 1

        STOCK_DATA = STOCK_DATA.reset_index().copy()

        STOCK_DATA = init_f_gen.finalize_formatting(STOCK_DATA)

        STOCK_DATA = self.add_missing_vars(STOCK_DATA)

        return STOCK_DATA
