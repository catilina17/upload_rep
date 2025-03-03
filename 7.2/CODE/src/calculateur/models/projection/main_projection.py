import numpy as np
from calculateur.calc_params import model_params as mod
from calculateur.models.run_offs.ech_amortization.amortissement_statique.class_init_ecoulement import Init_Ecoulement
from calculateur.models.run_offs.ech_amortization.amortissement_statique.class_static_amortization import Static_Amortization
from calculateur.models.run_offs.nmd_stratification.class_stratification_initiation import Stratification_Initiation
from calculateur.models.run_offs.nmd_stratification.class_nmd_runoff import NMD_RUNOFF_CALCULATION
from calculateur.models.rarn.class_rarn_manager import RARN_Manager
from calculateur.models.rarn.class_nmd_rarn_manager import RARN_NMD_Manager
from calculateur.models.rates.class_nmd_ftp_rates import NMD_FTP_Rate_Calculator
from calculateur.models.rarn.pel_tx_cloture import TAUX_CLOTURE_Manager
from calculateur.models.indicators.class_indicators_out import Indicators_Output
from calculateur.models.indicators.class_indicators_calculations import Indicators_Calculations
from calculateur.models.renego_immo.class_indicators_renegociated import Immo_Renegotiation
from calculateur.models.run_offs.ech_amortization.amortissement_renego.main_renego import RENGOCIATED_AMORTIZATION
from .class_projection_manager import Projection_Manager
from calculateur.models.rates.ech_pricing.fixed_rate_products_pricing import FIXED_RATE_PRODUCT_PRICING
import logging
from calculateur.calc_params.model_params import *

logger = logging.getLogger(__name__)

np.seterr(divide='ignore', invalid='ignore')
##@profile
def project_batch(cls_data_ldp, cls_data_palier, cls_fields, cls_format, cls_data_rate, cls_cal, cls_hz_params,
                  cls_data_cash_flow, cls_model_params, cls_zc_curves, tx_params, gap_tx_params, name_product,
                  calc_mode, tci_contract_perimeter):
    cls_proj = Projection_Manager(cls_fields, cls_hz_params, cls_data_rate, cls_cal)
    cls_proj.init_projection_classes(cls_data_ldp, cls_format, name_product, cls_model_params, calc_mode)

    init_projection_data(cls_proj, cls_data_ldp, cls_data_palier, cls_data_cash_flow, cls_format,
                         cls_model_params, tx_params)

    if len(cls_proj.data_ldp) > 0:
        if cls_proj.name_product == "all_ech_pn":
            cls_pricing_tf = price_future_products(cls_proj, cls_format, cls_data_cash_flow, cls_model_params,
                                                   cls_zc_curves, tx_params)
        else:
            cls_pricing_tf = None

        generate_rates_chronicles(cls_proj, cls_pricing_tf, cls_model_params, tx_params)

        cls_stat_runoff = generate_runoffs(cls_proj, cls_data_cash_flow, cls_model_params, cls_format, tx_params,
                                           tci_contract_perimeter)

        cls_rarn = get_ra_rn_params(cls_proj, cls_model_params, cls_format, tx_params)

        adapt_runoff_to_rarn_scenarios(cls_proj, cls_rarn, cls_stat_runoff, gap_tx_params, cls_model_params,
                                       cls_data_cash_flow, tx_params, cls_format, cls_data_palier)

        return cls_proj
    else:
        return None

####@profile
def get_ra_rn_params(cls_proj, cls_model_params, cls_format, tx_params):
    data_ldp = cls_proj.data_ldp
    mois_depart_rarn = cls_proj.cls_cal.mois_depart
    rarn_periods = cls_proj.cls_cal.rarn_periods
    drac_rarn = cls_proj.cls_cal.drac_rarn
    drac_init = cls_proj.cls_cal.drac_init
    current_month = cls_proj.cls_cal.current_month
    sc_rates = cls_proj.cls_rate.sc_rates.copy()
    sc_rates_lag = cls_proj.cls_rate.sc_rates_lag.copy()
    n = cls_proj.n
    t = cls_proj.t

    if cls_proj.name_product not in (models_nmd_st + models_nmd_pn):
        cls_rarn = RARN_Manager(cls_proj, cls_model_params)
        cls_rarn.get_rarn_values(sc_rates, sc_rates_lag, data_ldp, mois_depart_rarn,
                                 current_month, rarn_periods, drac_rarn, drac_init, n, t)
    elif cls_proj.name_product in (models_nmd_st + models_nmd_pn):
        cls_rarn = TAUX_CLOTURE_Manager(cls_proj, cls_format, cls_model_params)
        cls_rarn.get_taux_cloture(data_ldp, tx_params)
    else:
        cls_rarn = RARN_NMD_Manager(cls_proj)

    return cls_rarn


##@profile
def init_projection_data(cls_proj, cls_data_ldp, cls_data_palier, cls_data_cash_flow, cls_format,
                         cls_model_params, tx_params):
    cls_proj.prepare_ldp_data(cls_proj, cls_data_ldp, cls_data_cash_flow, cls_format, cls_model_params, tx_params)

    if len(cls_proj.data_ldp) > 0:

        cls_proj.set_min_proj(cls_proj.data_ldp)
        if cls_proj.name_product in mod.models_nmd_st + mod.models_nmd_pn:
            cls_model_params.cls_flow_params.get_flows(cls_proj, cls_proj.data_ldp.copy(), cls_proj.min_proj)
            cls_model_params.cls_flow_params.get_nmd_maturity(cls_proj, cls_model_params.cls_flow_params.monthly_flow)

        cls_proj.set_projection_dimensions(cls_model_params.cls_flow_params)

        if cls_proj.name_product in mod.models_nmd_st + mod.models_nmd_pn:
            if cls_proj.calculate_tci:
                cls_model_params.cls_flow_params.calculate_tci_mat_by_flow_model(cls_proj)
                cls_model_params.cls_flow_params.calculate_tci_index(cls_proj, cls_format, tx_params)
                cls_model_params.cls_flow_params.calculate_tci_var(cls_proj, cls_format, tx_params)

        cls_proj.cls_cal.prepare_calendar_parameters(cls_proj)

        cls_proj.cls_cal.get_calendar_periods(cls_proj)

        cls_proj.cls_cal.load_rarn_calendar_parameters(cls_proj)

        cls_proj.cls_fixing_cal.get_fixing_parameters(cls_proj)

        cls_proj.cls_palier.prepare_palier_data(cls_proj, cls_data_palier, cls_model_params.cls_ra_rn_params)

        if cls_proj.name_product in (models_nmd_st + models_nmd_pn):
            cls_proj.versements_model.create_versements_contracts(cls_proj)


############@profile
def price_future_products(cls_proj, cls_format, cls_cash_flow, cls_model_params, cls_zc_curves, tx_params):
    # FIRST PRICING
    cls_pricing_tf = FIXED_RATE_PRODUCT_PRICING(cls_proj, cls_zc_curves)
    cls_pricing_tf.generate_linear_amortization(cls_format, cls_cash_flow, cls_model_params, tx_params)
    cls_pricing_tf.check_pricing_curves()
    cls_pricing_tf.get_discount_factor()
    cls_pricing_tf.assign_fixed_rate(cls_proj)

    """for i in range(0,5):
        cls_pricing_tf = FIXED_RATE_PRODUCT_PRICING(cls_proj, cls_zc_curves)
        cls_pricing_tf.generate_real_amortization(cls_format, cls_cash_flow, cls_model_params, tx_params)
        cls_pricing_tf.check_pricing_curves()
        cls_pricing_tf.get_discount_factor()
        cls_pricing_tf.assign_fixed_rate(cls_proj)"""

    return cls_pricing_tf


####@profile
def generate_rates_chronicles(cls_proj, cls_pricing_tf, cls_model_params, tx_params):
    cls_proj.cls_data_rate.prepare_curve_rates(cls_proj, cls_model_params, tx_params, cls_pricing_tf=cls_pricing_tf)
    cls_proj.cls_rate.get_rates(cls_proj, cls_model_params)



def generate_runoffs(cls_proj, cls_cash_flow, cls_model_params, cls_format, tx_params, tci_contract_perimeter):
    name_product = cls_proj.name_product

    if name_product not in (models_nmd_st + models_nmd_pn):
        """CALCUL ECOULEMENT INITIAL AVT AMORTISSEMENT """
        cls_init_ec = Init_Ecoulement(cls_proj, cls_model_params)
        cls_init_ec.get_ec_before_amortization()

        """ CALCUL AMORTISSEMENT """
        cls_stat_amor = Static_Amortization(cls_format, cls_init_ec, cls_cash_flow)
        cls_stat_amor.generate_static_amortization(tx_params)

        return cls_stat_amor

    else:
        """ Initiation de la STRATIFICATION"""
        cls_init_ec = Stratification_Initiation(cls_proj, cls_model_params.cls_flow_params)
        cls_init_ec.get_stratification()

        """ CALCUL ECOULEMENT """
        cls_nmd_runoff = NMD_RUNOFF_CALCULATION(cls_init_ec, cls_cash_flow, cls_proj.versements_model)
        cls_nmd_runoff.compute_nmd_runoffs()

        """ CALCUL TCI """
        if cls_proj.calculate_tci:
            cls_nmd_tci = NMD_FTP_Rate_Calculator(cls_model_params, tx_params, cls_cash_flow, cls_proj)
            cls_nmd_tci.calculate_ftp_rate(tci_contract_perimeter)

        return cls_nmd_runoff


####@profile
def adapt_runoff_to_rarn_scenarios(cls_proj, cls_rarn, cls_stat_runoff, gap_tx_params, cls_model_params,
                                   cls_data_cash_flow, tx_params, cls_format, cls_data_palier):
    name_product = cls_stat_runoff.cls_proj.name_product
    data_ldp = cls_stat_runoff.cls_proj.data_ldp
    dic_palier = cls_stat_runoff.cls_palier.dic_palier
    current_month = cls_stat_runoff.cls_cal.current_month
    mois_depart_amor = cls_stat_runoff.cls_cal.mois_depart_amor
    mois_fin_amor = cls_stat_runoff.cls_cal.mois_fin_amor
    interests_calc_periods = cls_stat_runoff.cls_cal.interests_calc_periods
    drac_amor = cls_stat_runoff.cls_cal.drac_amor
    dar_mois = cls_stat_runoff.cls_hz_params.dar_mois
    capitals = {}
    capitals["all"] = cls_stat_runoff.capital_ec
    capitals["stable"] = cls_stat_runoff.capital_ec_stable
    capitals["vol"] = cls_stat_runoff.capital_ec_mni_volatile
    capital_gptx = {}
    capital_gptx["all"] = cls_stat_runoff.capital_ec_gptx
    capital_gptx["stable"] = cls_stat_runoff.capital_ec_gptx_stable
    capital_gptx["vol"] = cls_stat_runoff.capital_ec_gptx_mni_volatile
    capital_avt_amor = {}
    capital_avt_amor["all"] = cls_stat_runoff.cls_init_ec.ec_depart
    capital_avt_amor["stable"] = cls_stat_runoff.cls_init_ec.ec_depart
    capital_avt_amor["vol"] = np.zeros(capital_avt_amor["all"].shape)
    n = cls_stat_runoff.cls_proj.n
    t = cls_stat_runoff.cls_proj.t
    data_optional = cls_stat_runoff.cls_proj.data_optional
    mois_depart = cls_stat_runoff.cls_cal.mois_depart
    mois_fin = cls_stat_runoff.cls_cal.mois_fin
    douteux = np.array((data_ldp[cls_stat_runoff.cls_fields.NC_LDP_PERFORMING] == "T"))
    tombee_fixing = cls_stat_runoff.cls_rate.tombee_fixing[:, :t]
    period_fixing = cls_stat_runoff.cls_rate.period_fixing[:, :t]

    """ CALCUL JAMBE PERMANENTE (LEG D RCO) """
    cls_static_ind = Indicators_Calculations(cls_stat_runoff)
    cls_proj.cls_cal.prepare_data_cal_indicators(data_ldp, name_product, cls_stat_runoff, n, t)
    if "lem_stable" in cls_proj.cls_fields.exit_indicators:
        types_capital = ["all", "stable", "vol"]
    else:
        types_capital = ["all"]

    for type_ind in ["liq", "taux"]:
        for type_capital in types_capital:
            cls_static_ind.get_static_indics(cls_stat_runoff, data_ldp, capitals[type_capital].copy(), cls_rarn,
                                             capital_avt_amor[type_capital], mois_depart, mois_fin,
                                             mois_fin_amor, current_month, dar_mois, douteux, tombee_fixing, n, t,
                                             type_ind=type_ind, period_fixing=period_fixing,
                                             type_capital=type_capital, rem_cap_gptx=capital_gptx)

    """ CALCUL JAMBE RENEGOCIE IMMO (LEG N RCO)"""
    cls_reneg_ind_immo = Immo_Renegotiation(cls_stat_runoff, cls_static_ind, cls_model_params)
    cls_reneg_ind_immo.get_reneg_indics(cls_static_ind, data_ldp, capitals["all"], cls_rarn,
                                        mois_depart_amor, mois_fin_amor, drac_amor, current_month, mois_depart,
                                        interests_calc_periods, dar_mois, douteux, dic_palier, n, t)

    """ CALCUL JAMBE RENEGOCIE CAT (LEG N RCO)"""
    cls_reneg_ind_cat = RENGOCIATED_AMORTIZATION(cls_proj, cls_model_params, cls_data_cash_flow,
                                                 cls_format, tx_params, cls_data_palier)
    cls_reneg_ind_cat.calculate_renegociated_amortization(cls_static_ind, capitals["all"],
                                                          cls_static_ind.effet_rarn_cum, cls_rarn,
                                                          current_month, mois_depart, douteux, n, t)

    """ LEF, LEM, TEF, TEM etc. """
    cls_proj.dic_inds = {}
    cls_ind_out = Indicators_Output(cls_stat_runoff)
    cls_proj.dic_inds \
        = cls_ind_out.calculate_output_indics(data_ldp, capitals, cls_static_ind, cls_reneg_ind_immo, cls_reneg_ind_cat,
                                              cls_rarn, gap_tx_params)

    cls_proj.data_ldp = data_ldp
    cls_proj.data_optional = data_optional
