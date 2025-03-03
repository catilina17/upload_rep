from calculateur.rates_transformer.swap_rates_interpolator import Rate_Interpolator
from modules.moteur.parameters import general_parameters as gp
from services.rate_services.params import maturities_referential as mr
from utils import general_utils as gu
import pandas as pd
from mappings import mapping_products as mpr
from params import version_params as vp
from calculateur.rates_transformer.rate_calc_params import Calculator_RateParams_Manager
from utils import excel_openpyxl as ex
import logging

logger = logging.getLogger(__name__)


class Calculator_Params():

    def __init__(self, cl_usr, cls_mp, cls_tx, cls_sp):
        self.tx = cls_tx
        self.up = cl_usr
        self.mp = cls_mp
        self.batch_size_nmd = cls_sp.batch_size_nmd
        self.batch_size_ech = cls_sp.batch_size_ech
        self.all_ech_products = ["a-creqtv", "a-creq", "a-criv", "a-crctz", "a-crctf", "a-crctv",
                                 "a-intbq-tf", "a-intbq-tv", "p-intbq-tf", "p-intbq-tv", "p-cat-tf", "p-cat-tv",
                                 "a-autres-tf", "a-autres-tv", "p-autres-tf", "p-autres-tv", "a-crif",
                                 "p-swap-tf", "p-swap-tv", "a-swap-tf", "a-swap-tv",
                                 "p-security-tf", "p-security-tv", "a-security-tf", "a-security-tv",
                                 "p-change-tf", "a-change-tf"]

        self.all_ech_tf_products = ["a-creq", "a-crctz", "a-crctf", "a-intbq-tf", "p-intbq-tf", "p-cat-tf",
                                    "a-autres-tf", "p-autres-tf", "a-crif", "p-swap-tf", "a-swap-tf",
                                    "a-change-tf", "p-change-tf", "a-security-tf", "p-security-tf"]

        self.all_ech_tv_products = ["a-creqtv", "a-crctv", "a-intbq-tv", "p-intbq-tv", "p-cat-tv",
                                    "a-autres-tv", "p-autres-tv", "a-criv", "p-swap-tv", "a-swap-tv",
                                    "p-security-tv", "a-security-tv", "a-repo-tv", "p-repo-tv"]

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
                                  "STOCK ECH SWAP TF": ["p-swap-tf", "a-swap-tf"],
                                  "STOCK ECH SWAP TV": ["p-swap-tv", "a-swap-tv"],
                                  "STOCK ECH TITRES TF": ["p-security-tf", "a-security-tf"],
                                  "STOCK ECH TITRES TV": ["p-security-tv", "a-security-tv"],
                                  "STOCK ECH CHANGE TF": ["p-change-tf", "a-change-tf"],
                                  "STOCK ECH REPOS TF": ["p-repos-tf", "a-repos-tf"],
                                  "STOCK ECH REPOS TV": ["p-repos-tv", "a-repos-tv"]}

        self.products_map_nomenclature = mpr.products_map

        self.out_of_recalc_st_etabs = ["NTX", "SEF", "CFF", "NPS"]

    def load_calculateur_params(self, data_template_mapped, data_stock, cls_loader_pn):
        self.dar = self.up.dar
        self.horizon = self.up.nb_mois_proj_usr
        self.max_pn = gp.pn_max
        self.etab = self.up.nom_etab

        self.curve_accruals_map = self.mp.curve_accruals_map
        self.mapping_bassins_modele_rarn = self.mp.bassins_map

        logger.info("          CHARGEMENT et INTERPOLATION DES COURBES DE TAUX")
        self.tx_params = self.get_rate_params()

        self.cls_nmd_tmp = type('MyClass', (object,), {'content': {}})()
        self.cls_nmd_tmp.data_template_mapped = data_template_mapped

        if "EVE" in self.up.type_simul.keys() or "EVE_LIQ" in self.up.type_simul.keys():
            self.nmd_tci_perimeter = self.get_nmd_tci_perimeter(data_stock, cls_loader_pn)
        else:
            self.nmd_tci_perimeter = ["X-XX"]

    def get_nmd_tci_perimeter(self, data_stock, cls_pn_loader):
        if "nmd" in cls_pn_loader.dic_pn_nmd:
            data_pn = cls_pn_loader.dic_pn_nmd['nmd']
            list_contracts_pn = data_pn[data_pn[gp.nc_output_dim2].isin(self.mp.mapping_eve["cc_tci"])][
                gp.nc_output_contrat_cle].unique().tolist()
            list_excl_contracts_pn = data_pn[data_pn[gp.nc_output_contrat_cle].isin(self.mp.mapping_eve["cc_tci_excl"])][
                gp.nc_output_contrat_cle].unique().tolist()
            list_contracts_pn = [x for x in list_contracts_pn if x not in list_excl_contracts_pn]
        else:
            list_contracts_pn = []

        list_contracts_stock = data_stock[data_stock[gp.nc_output_dim2].isin(self.mp.mapping_eve["cc_tci"])][
            gp.nc_output_contrat_cle].unique().tolist()
        list_excl_contracts_stock = data_stock[data_stock[gp.nc_output_contrat_cle].isin(self.mp.mapping_eve["cc_tci_excl"])][
            gp.nc_output_contrat_cle].unique().tolist()
        list_contracts_stock = [x for x in list_contracts_stock if x not in list_excl_contracts_stock]

        return list(set(list_contracts_stock + list_contracts_pn))

    def get_source_data_pn(self, cls_pn_loader, type, prct=""):
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
        modele_path = self.up.scenarios_files[self.etab][scenario_name]["MODELE_%s" % type]
        gu.check_version_templates(modele_path, path=modele_path,
                                   open=True, version=eval("vp.version_modele_%s" % type.lower()))
        return self.get_model_wb(modele_path)

    def get_model_wb(self, model_file_path):
        model_wb = None
        model_wb = ex.load_workbook_openpyxl(model_file_path, read_only=True)
        return model_wb


    def get_rate_params(self):
        tx_curves = self.tx.tx_curves_sc.copy()
        liq_curves = self.tx.liq_curves_sc.copy()

        rcm = Calculator_RateParams_Manager(is_formated = True, interpolate_curves=True)
        tx_params = rcm.load_rate_params(pd.concat([tx_curves, liq_curves]),
                                         self.tx.tci_data, self.tx.zc_curves_df)

        return tx_params
