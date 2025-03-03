import os
class OutputManager:
    def __init__(self):
        self.set_tags()
        self.set_output_file_paths()

    def set_tags(self):
        # Primary tags
        self.lcr_nsfr_st_tag = "ST_LCR_NSFR"
        self.stock_tag = "STOCK_AG"

        # Folder tags
        self.sc_volume_folder_tag = "SC_VOLUME"
        self.sc_taux_folder_tag = "SC_TAUX"
        self.sc_lcr_nsfr_folder_tag = "SC_LCR_NSFR"

        # Volume output tags
        self.sc_vol_nmd_output_tag = "NMD"
        self.sc_vol_ech_output_tag = "PN_ECH"
        self.sc_vol_nmd_prct_output_tag = "NMD_BC"
        self.sc_vol_pn_ech_prct_output_tag = "PN_ECH_BC"
        self.sc_vol_nmd_calage_output_tag = "NMD_CALAGE"

        # Taux output tags
        self.sc_taux_output_tag = "SC_TX"
        self.sc_liq_output_tag = "SC_LIQ"
        self.sc_zc_output_tag = "SC_ZC"
        self.sc_tci_output_tag = "SC_TCI"
        self.sc_rco_ref_output_tag = "RCO_TX"

        # Other output tags
        self.missing_map_output_tag = "MAPPINGS_MANQUANTS"
        self.nmd_template_output_tag = "STOCK_NMD_TEMPLATE"
        self.nmd_template_output_tag = "STOCK_NMD_TEMPLATE"
        self.sc_lcr_nsfr_output_tag = "SC_LCR_NSFR"

        # File extensions
        self.csv_extension = ".csv"
        self.xlsx_extension = ".xlsx"

    def set_output_file_paths(self):
        # Define base folder paths for consistency
        base_volume_path = self.sc_volume_folder_tag
        base_taux_path = self.sc_taux_folder_tag
        base_lcr_nsfr_path = self.sc_lcr_nsfr_folder_tag

        # Define output file paths
        self.lcr_nsfr_st_output_file = os.path.join("", f"{self.lcr_nsfr_st_tag}_%s{self.xlsx_extension}")
        self.stock_output_file = os.path.join("", f"{self.stock_tag}_%s{self.csv_extension}")

        # Volume-related files
        self.sc_vol_nmd_output_file = os.path.join("", base_volume_path,
                                                   f"{self.sc_vol_nmd_output_tag}_%s{self.csv_extension}")
        self.sc_vol_ech_output_file = os.path.join("", base_volume_path,
                                                   f"{self.sc_vol_ech_output_tag}_%s{self.csv_extension}")
        self.sc_vol_nmd_prct_output_file = os.path.join("", base_volume_path,
                                                        f"{self.sc_vol_nmd_prct_output_tag}_%s{self.csv_extension}")
        self.sc_vol_pn_ech_prct_output_file = os.path.join("", base_volume_path,
                                                           f"{self.sc_vol_pn_ech_prct_output_tag}_%s{self.csv_extension}")
        self.sc_vol_nmd_calage_output_file = os.path.join("", base_volume_path,
                                                          f"{self.sc_vol_nmd_calage_output_tag}_%s{self.csv_extension}")

        # Taux-related files
        self.sc_taux_output_file = os.path.join("", base_taux_path, f"{self.sc_taux_output_tag}_%s{self.csv_extension}")
        self.sc_liq_output_file = os.path.join("", base_taux_path, f"{self.sc_liq_output_tag}_%s{self.csv_extension}")
        self.sc_zc_output_file = os.path.join("", base_taux_path, f"{self.sc_zc_output_tag}_%s{self.csv_extension}")
        self.sc_tci_output_file = os.path.join("", base_taux_path, f"{self.sc_tci_output_tag}_%s{self.csv_extension}")
        self.sc_rco_ref_output_file = os.path.join("", base_taux_path,
                                                   f"{self.sc_rco_ref_output_tag}_%s{self.csv_extension}")

        # Other output files
        self.sc_lcr_nsfr_output_file = os.path.join("", base_lcr_nsfr_path,
                                                    f"{self.sc_lcr_nsfr_folder_tag}_%s{self.xlsx_extension}")
        self.missing_map_output_file = os.path.join("", f"{self.missing_map_output_tag}_%s{self.xlsx_extension}")
        self.nmd_template_output_file = os.path.join("", f"{self.nmd_template_output_tag}_%s{self.csv_extension}")