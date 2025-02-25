from modules.moteur.parameters import user_parameters as up
from modules.moteur.mappings import main_mappings as mp
import modules.moteur.low_level_services.reporting_service.mapping_module as rmp
from calculateur.rates_transformer.swap_rates_interpolator import Rate_Interpolator
from modules.moteur.parameters import general_parameters as gp
from params import version_params as vp
from modules.scenario.rate_services import maturities_referential as mr
from utils import general_utils as gu
from modules.moteur.index_curves import tx_module as tx
import pandas as pd
import re
from utils import excel_utils as ex
import logging

logger = logging.getLogger(__name__)
class Calculator_Params():

    def __init__(self):
        self.all_ech_products = ["a-creqtv", "a-creq", "a-criv", "a-crctz", "a-crctf", "a-crctv",
                                 "a-intbq-tf", "a-intbq-tv", "p-intbq-tf", "p-intbq-tv", "p-cat-tf", "p-cat-tv",
                                 "a-autres-tf", "a-autres-tv", "p-autres-tf", "p-autres-tv", "a-crif",
                                 "p-swap-tf", "p-swap-tv", "a-swap-tf", "a-swap-tv",
                                 "p-security-tf", "p-security-tv", "a-security-tf", "a-security-tv",
                                 "p-change-tf","a-change-tf"]

        self.all_ech_tf_products = ["a-creq", "a-crctz", "a-crctf", "a-intbq-tf", "p-intbq-tf", "p-cat-tf",
                                    "a-autres-tf", "p-autres-tf", "a-crif", "p-swap-tf", "a-swap-tf",
                                    "a-change-tf", "p-change-tf", "a-security-tf", "p-security-tf"]

        self.all_ech_tv_products = ["a-creqtv", "a-crctv", "a-intbq-tv", "p-intbq-tv", "p-cat-tv",
                                    "a-autres-tv", "p-autres-tv", "a-criv", "p-swap-tv","a-swap-tv",
                                    "p-security-tv", "a-security-tv"]

        self.products_name_map = {"STOCK CAP & FLOOR": ["cap_floor"], "STOCK ECH CR IMMO TF": ["a-crif"],
                                  "STOCK ECH CR IMMO TV": ["a-criv"],
                                  "STOCK ECH CREDITS EQUIPEMENTS TF": ["a-creq"],
                                  "STOCK ECH CREDITS EQUIPEMENTS TV": ["a-creqtv"],
                                  "STOCK ECH PRETS PERSO TF": ["a-crctf"], "STOCK ECH PRETS PERSO TV": ["a-crctv"],
                                  "STOCK ECH PRETS INTBQ TF": ["a-intbq-tf"],
                                  "STOCK ECH PRETS INTBQ TV": ["a-intbq-tv"],
                                  "STOCK ECH EMPRUNTS INTBQ TF": ["p-intbq-tf"],
                                  "STOCK ECH EMPRUNTS INTBQ TV": ["p-intbq-tv"],
                                  "STOCK ECH COMPTES A TERMES TF": ["p-cat-tf"],
                                  "STOCK ECH COMPTES A TERMES TV": ["p-cat-tv"],
                                  "STOCK ECH AUTRES ACTIFS TF": ["a-autres-tf"],
                                  "STOCK ECH AUTRES ACTIFS TV": ["a-autres-tv"],
                                  "STOCK ECH AUTRES PASSIFS TF": ["p-autres-tf"],
                                  "STOCK ECH AUTRES PASSIFS TV": ["p-autres-tv"],
                                  "STOCK ECH PRÊTS PTZ": ["a-crctz"], "TOUT ECH (TRES LONG)": self.all_ech_products,
                                  "STOCK TOUT ECH FIXE (TRES LONG)": self.all_ech_tf_products,
                                  "STOCK TOUT ECH VARIABLE": self.all_ech_tv_products,
                                  "STOCK NMD": ["nmd_st"],
                                  "STOCK ECH SWAP TF" : ["p-swap-tf", "a-swap-tf"],
                                  "STOCK ECH SWAP TV" : ["p-swap-tv", "a-swap-tv"],
                                  "STOCK ECH TITRES TF": ["p-security-tf", "a-security-tf"],
                                  "STOCK ECH TITRES TV": ["p-security-tv", "a-security-tv"],
                                  "STOCK ECH CHANGE TF": ["p-change-tf", "a-change-tf"]}

        self.products_map_nomenclature = {"a-crif": "CR-IMMO-TF", "a-criv": "CR-IMMO-TV",
                                          "cap_floor": "CAP-FLOOR",
                                          "a-crctz": "CR-PTZ-TF", "a-crctf": "PR-PERSO-TF", "a-creq": "CR-EQ-TF",
                                          "a-crctv": "PR-PERSO-TV",
                                          "a-creqtv": "CR-EQ-TV", "p-cat-tf": "P-CAT-TF", "p-cat-tv": "P-CAT-TV",
                                          "a-intbq-tf": "A-INTBQ-TF",
                                          "a-intbq-tv": "A-INTBQ-TV", "p-intbq-tf": "P-INTBQ-TF",
                                          "p-intbq-tv": "P-INTBQ-TV",
                                          "a-autres-tf": "A-AUTRES-TF", "a-autres-tv": "A-AUTRES-TV",
                                          "p-autres-tf": "P-AUTRES-TF", "p-autres-tv": "P-AUTRES-TV",
                                          "nmd_st": "ST-NMD", "p-swap-tv": "P-SWAP-TV",
                                          "p-swap-tf": "P-SWAP-TF", "a-swap-tv": "A-SWAP-TV", "a-swap-tf": "A-SWAP-TF",
                                          "p-security-tf": "P-SECURITY-TF", "a-security-tv": "A-SECURITY-TV",
                                          "a-security-tf": "A-SECURITY-TF", "p-security-tv": "P-SECURITY-TV",
                                          "p-change-tf": "P-CHANGE-TF", "a-change-tf": "A-CHANGE-TF", }

        self.out_of_recalc_st_etabs = ["NTX", "SEF", "CFF", "NPS"]

    def load_calculateur_params(self, data_template_mapped, data_stock, cls_loader_pn):
        self.dar = up.dar_usr
        self.horizon = up.nb_mois_proj_usr
        self.max_pn = gp.pn_max
        self.etab = up.nom_etab

        self.curve_accruals_map = mp.curve_accruals_map
        self.mapping_bassins_modele_rarn = mp.bassins_map

        logger.info("          CHARGEMENT et INTERPOLATION DES COURBES DE TAUX")
        self.tx_params = self.get_rate_params()

        self.cls_nmd_tmp = type('MyClass', (object,), {'content':{}})()
        self.cls_nmd_tmp.data_template_mapped = data_template_mapped

        self.get_pass_alm_general_mappings()

        if "EVE" in up.type_simul.keys() or "EVE_LIQ" in up.type_simul.keys() :
            self.nmd_tci_perimeter = self.get_nmd_tci_perimeter(data_stock, cls_loader_pn)
        else:
            self.nmd_tci_perimeter = ["X-XX"]


    def get_nmd_tci_perimeter(self, data_stock, cls_pn_loader):
        if "nmd" in cls_pn_loader.dic_pn_nmd:
            data_pn = cls_pn_loader.dic_pn_nmd['nmd']
            list_contracts_pn = data_pn[data_pn[gp.nc_output_dim2].isin(mp.cc_tci)][gp.nc_output_contrat_cle].unique().tolist()
            list_excl_contracts_pn = data_pn[data_pn[gp.nc_output_contrat_cle].isin(mp.cc_tci_excl)][gp.nc_output_contrat_cle].unique().tolist()
            list_contracts_pn = [x for x in list_contracts_pn if x not in list_excl_contracts_pn]
        else:
            list_contracts_pn = []

        list_contracts_stock = data_stock[data_stock[gp.nc_output_dim2].isin(mp.cc_tci)][gp.nc_output_contrat_cle].unique().tolist()
        list_excl_contracts_stock = data_stock[data_stock[gp.nc_output_contrat_cle].isin(mp.cc_tci_excl)][
            gp.nc_output_contrat_cle].unique().tolist()
        list_contracts_stock = [x for x in list_contracts_stock if x not in list_excl_contracts_stock]

        return list(set(list_contracts_stock + list_contracts_pn))


    def get_source_data_pn(self, cls_pn_loader, type, prct = ""):
        source_data_pn = {}
        source_data_pn["LDP"] = {}
        try:
            source_data_pn["LDP"]["DATA"] = eval("cls_pn_loader.dic_pn_%s['%s']" % (type, type + prct)).copy()
        except:
            try:
                source_data_pn["LDP"]["DATA"] = eval("cls_pn_loader.dic_pn_%s['%s']" % (type, type + "%")).copy()
            except:
                source_data_pn["LDP"]["DATA"] = []

        if type == "nmd" and 'calage' in cls_pn_loader.dic_pn_nmd:
            source_data_pn["CALAGE"] = {}
            source_data_pn["CALAGE"]["DATA"] = cls_pn_loader.dic_pn_nmd['calage']

        return source_data_pn

    def get_source_models(self, scenario_name, type):
        modele_path = up.scenarios_files[self.etab][scenario_name]["MODELE_%s" % type]
        gu.check_version_templates(modele_path, path=modele_path,
                                   open=True, version=eval("vp.version_modele_%s" % type.lower()))
        return self.get_model_wb(modele_path)

    def get_model_wb(self, model_file_path):
        model_wb = None
        model_wb = ex.try_close_open(model_file_path, read_only=True)
        ex.unfilter_all_sheets(model_wb)
        return model_wb

    def get_pass_alm_general_mappings(self):
        rmp.get_main_mappings(mp.map_wb)

    def get_rate_params(self):
        tx_curves = tx.tx_curves_sc.copy()
        liq_curves = tx.liq_curves_sc.copy()
        tx_cols_num = tx_curves.loc[:, "M00":].columns.tolist()
        tx_curves["MATURITE"] = [x[re.search(r"\d", x).start():] if re.search(r"\d", x) is not None else 0 for x in
                                 tx_curves["CODE"].fillna("").values.tolist()]

        tx_params = {"curves_df": {"data" : pd.concat([tx_curves, liq_curves]), "cols": tx_cols_num, "max_proj": len(tx_cols_num) - 1,
                     "nom_index": "CODE", "curve_code" : "CODE COURBE", "tenor" : "TENOR",
                     "maturity_to_days": mr.maturity_to_days_360, "curve_name_taux_pel":"TAUX_PEL",
                     "tenor_taux_pel": "12M1D"},
                     "accrual_map": {"data" : mp.curve_accruals_map, "accrual_conv_col": gp.nc_accrual_conversion_curve_accruals,
                     "type_courbe_col": gp.nc_type_courbe_curve_accruals,
                     "accrual_method_col": gp.nc_accrual_method_curve_accruals, "alias":gp.nc_type_courbe_alias,
                     "standalone_const": "Standalone index" ,"curve_name" : "CURVE_NAME"}, "ZC_DATA": {"data" :tx.zc_curves_df},
                     "map_pricing_curves": {"data" : mp.param_pn_ech_pricing,
                     "col_pricing_curve": "COURBE PRICING"}, "map_index_curve_tenor": {"data" : mp.index_curve_tenor_map,
                     "col_curve": "CURVE_NAME", "col_tenor": "TENOR"}, "tci_vals" : {"data" : tx.tci_values.copy()}}


        rate_int_cls = Rate_Interpolator()
        tx_params["dic_tx_swap"] = rate_int_cls.interpolate_curves(tx_params)

        return tx_params
