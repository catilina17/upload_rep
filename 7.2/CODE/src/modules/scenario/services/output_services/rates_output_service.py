import os
import logging
import pandas as pd
from pathlib import Path
from modules.scenario.parameters.general_parameters import *
from services.rate_services.params import tx_referential as tx_ref
from modules.scenario.services.rates_services import rate_temp_files_saver as taux_ser

logger = logging.getLogger(__name__)


class Rate_Exporter():
    def __init__(self, cls_usr, etab, output_dir, scenario_name):
        self.etab = etab
        self.output_dir = output_dir
        self.scenario_name = scenario_name
        self.up = cls_usr
        self.OUTPUT_COLS_CURVES = [tx_ref.CN_TYPE2_COURBE, tx_ref.CN_CODE_DEVISE, tx_ref.CN_CODE_COURBE,
                                   tx_ref.CN_MATURITY,
                                   tx_ref.CN_CODE_PASSALM]
        self.OUTPUT_COLS_CURVES_NUM = ["M" + str(i) for i in range(tx_ref.NB_PROJ_TAUX + 1)]

    def export_tx_data_to(self, scenario_rows, tci_liq_nmd_df):
        scenario_curves_df = taux_ser.get_sc_tx_df(scenario_rows[self.up.RN_USER_RATE_SC].iloc[0])
        bootstrap_df = taux_ser.get_zc_df(scenario_rows[self.up.RN_USER_RATE_SC].iloc[0])
        reference_tx_curves_df = taux_ser.get_stock_sc_tx_df(self.etab, self.up.st_refs)
        try:
            logger.info('    Ecriture des courbes liq et taux dans le fichier scénario ')
            sc_tx_df, sc_liq_df, fermat_df = self.format_data(scenario_curves_df, reference_tx_curves_df)
            self.print_in_output_file(sc_tx_df, sc_liq_df, fermat_df, bootstrap_df, tci_liq_nmd_df)
        except Exception as e:
            logger.error(e, exc_info=True)

    def print_in_output_file(self, sc_tx_df, sc_liq_df, fermat_df, bootstrap_df, tci_liq_nmd_df):
        output_dir_sc = os.path.join(self.output_dir, self.scenario_name)
        Path(os.path.join(output_dir_sc, "SC_TAUX")).mkdir(parents=True, exist_ok=True)
        dar = self.up.dar.strftime("%y%m%d")
        sc_tx_df.to_csv(os.path.join(output_dir_sc, self.up.om.sc_taux_output_file % (self.etab + "_" + dar)), sep=";", decimal=",",
                        header=True, index=False)
        sc_liq_df.to_csv(os.path.join(output_dir_sc, self.up.om.sc_liq_output_file % (self.etab + "_" + dar)), sep=";", decimal=",",
                         header=True, index=False)
        fermat_df.to_csv(os.path.join(output_dir_sc, self.up.om.sc_rco_ref_output_file % (self.etab + "_" + dar)), sep=";",
                         decimal=",", header=True, index=False)
        bootstrap_df.to_csv(os.path.join(output_dir_sc, self.up.om.sc_zc_output_file % (self.etab + "_" + dar)), sep=";",
                            decimal=",", header=True, index=False)
        tci_liq_nmd_df.to_csv(os.path.join(output_dir_sc, self.up.om.sc_tci_output_file % (self.etab + "_" + dar)), sep=";",
                              decimal=",", header=True, index=False)

    def format_data(self, sc_df, fermat_df):
        fermat_df = fermat_df[(fermat_df[tx_ref.CN_TYPE] == "TAUX")]
        fermat_df = fermat_df[self.OUTPUT_COLS_CURVES + self.OUTPUT_COLS_CURVES_NUM].copy()
        fermat_df = self.apply_output_format(fermat_df, ratio=100)
        fermat_df = self.rank_curves_tx(fermat_df)
        sc_tx_df = sc_df[(sc_df[tx_ref.CN_TYPE] == "TAUX")].copy()
        sc_liq_df = sc_df[(sc_df[tx_ref.CN_TYPE] == "LIQ")].copy()
        sc_tx_df = sc_tx_df[self.OUTPUT_COLS_CURVES + self.OUTPUT_COLS_CURVES_NUM].copy()
        sc_liq_df = sc_liq_df[self.OUTPUT_COLS_CURVES + self.OUTPUT_COLS_CURVES_NUM].copy()
        sc_tx_df = self.apply_output_format(sc_tx_df, ratio=100)
        sc_liq_df = self.apply_output_format(sc_liq_df, ratio=100)
        sc_tx_df = self.rank_curves_tx(sc_tx_df)
        sc_liq_df = self.rank_curves_liq(sc_liq_df)
        return sc_tx_df, sc_liq_df, fermat_df

    def get_rank_df(self, dict, cols):
        rank_df = pd.DataFrame.from_dict(dict, orient='index').reset_index()
        rank_df.columns = cols
        rank_df.set_index(cols[1], inplace=True)
        return rank_df

    def rank_curves_tx(self, tx_df):
        rank_tx = self.get_rank_df(RANK_CURVES_TX, ["RANK", "TYPE"])
        rank_dev = self.get_rank_df(RANK_CURVES_DEV, ["RANK2", "DEVISE"])
        tx_df = (tx_df.join(rank_tx, on=[tx_ref.CN_TYPE2_COURBE])).join(rank_dev, on=[tx_ref.CN_DEVISE])
        tx_df.sort_values(["RANK", "RANK2"], inplace=True)
        return tx_df.drop(["RANK", "RANK2"], axis=1)

    def rank_curves_liq(self, liq_df):
        rank_liq = self.get_rank_df(RANK_CURVES_LIQ, ["RANK", "TYPE"])
        rank_dev = self.get_rank_df(RANK_CURVES_DEV, ["RANK2", "DEVISE"])
        liq_df = (liq_df.join(rank_liq, on=[tx_ref.CN_TYPE2_COURBE])).join(rank_dev, on=[tx_ref.CN_DEVISE])
        liq_df.sort_values(["RANK", "RANK2"], inplace=True)
        return liq_df.drop(["RANK", "RANK2"], axis=1)

    def add_empty_row_index_row_index(self, data_df, empty_row, first_index_list):
        data_df.reset_index(drop=True, inplace=True)
        for code in first_index_list:
            index = data_df.index[data_df[tx_ref.CN_CODE_PASSALM] == code][0]
            data_df = pd.concat([data_df.iloc[:index, :], empty_row, data_df.iloc[index:, :]]).reset_index(drop=True)
        return data_df

    def apply_output_format(self, data_df, ratio=1):
        data_df = data_df.set_index(self.OUTPUT_COLS_CURVES)
        data_df = data_df.loc[:, 'M0':]
        data_df = data_df / ratio
        data_df = data_df.ffill(axis='columns')
        data_df.reset_index(inplace=True)
        return data_df
