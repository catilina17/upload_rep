# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:26:43 2020
@author: TAHIRIH"""
from mappings.pass_alm_fields import PASS_ALM_Fields as pa


""" "PARAMETRES INTERNES """
stock_tag = "STOCK_AG"
stock_nmd_template_tag = "STOCK_NMD_TEMPLATE"
sc_tx_tag = "SC_TAUX"
sc_vol_tag = "SC_VOLUME"
sc_modele_dav_tag = "SC_MOD_DAV"
sc_modele_ech_tag = "SC_MOD_ECH"
sc_modele_nmd_tag = "SC_MOD_NMD"
sc_modele_pel_tag = "SC_MOD_PEL"
sc_lcr_nsfr_tag = "SC_LCR_NSFR"
sc_params_tag = "scenario_params"
data_sc_tag = "DATA_SC_CALC"
main_sc_eve_tag = "MAIN_SCENARIO_EVE"
absc_map = "#Mapping"
# MODIF
FIX_ind = "FIXE"


pn_max = 60
real_max_months = 300
max_months = 360
max_months2 = 600
base_actu_list = ["30/360", "ACT/360", "ACT/365", "ACT/ACT", "30E/360"]
avg_nb_days_month = 14
nc_tx_fix_f = "TAUX_FIXE_F"
nc_tx_fix_m = "TAUX_FIXE_M"
nc_tx_prod = "TAUX_PROD"

""" Feuille Param_Autres"""
ng_courbes_ajust = "_COURBES_AJUST"
ng_spread_ajust = "_spread_ajust"
nc_taux_spread = "TAUX (bps)"
nc_devise_spread = "DEVISE"
nc_nb_mois_pn = "_nb_mois_pn"
nc_type_eve = "_type_eve"

ng_tla_retraitement = "_TLA_IsRetraitement"
ng_date_refix = "_TLA_NextRefix"
ng_freq_refix = "_FREQ_REFIX_TLA"

ng_force_gp_liq = "_force_gap_liq"
ng_force_gps_nmd = "_force_gaps_nmd"
ng_ajust_only = "_only_ajust"

ng_tla_indexes = "_TLA_INDEXES"
ng_tlb_indexes = "_TLB_INDEXES"
ng_tdav_indexes = "_TDAV_INDEXES"
ng_topc_indexes = "_TOPC_INDEXES"
ng_lep_indexes = "_LEP_INDEXES"
ng_cel_indexes = "_CEL_INDEXES"
ng_inf_indexes = "_INF_INDEXES"

""" Feuille Param_EVE """
ng_eve_contracts = "_EVE_EXCL_CONT"
nc_contrat_eve = "CONTRAT"
ng_eve_reg_floor = "_REG_FLOOR_EVE"
ng_vision_eve = "_vision_eve"
ng_contrat_ss_stress = "_CC_SANS_STRESS"
ng_contrat_tci_pn = "_TCI_PN"
ng_contrat_tci_pn_excl = "_TCI_PN_EXCL"
ng_tla_retraitement_eve = "_TLA_IsRetraitement_eve"
ng_date_refix_eve = "_TLA_NextRefix_eve"
ng_freq_refix_eve = "_FREQ_REFIX_TLA_eve"
ng_act_eve = "_ACT_EVE"
nc_act_eve_bilan = "BILAN"
nc_act_eve_contrat = "CONTRAT"
nc_act_eve_dev = "DEVISE"
nc_act_eve_act_curve = "COURBE ACT"
nc_mode_calc_gptx_immo = "_MODE_CALC_GP_TX_IMMO"

ng_force_gp_liq_eve = "_force_gap_liq_eve"
ng_force_gps_nmd_eve = "_force_gaps_nmd_eve"
ng_taux_pel_eve = "_Taux_PEL_eve"
ng_taux_ra_eve = "_Taux_RA_eve"

""" Feuille Global params"""
ng_liste_etab_global = "_LISTE_ETAB"

""" Feuille Param_General """
ng_list_etab_user = "_Etab"
ng_input_path = "_INPUT_PATH"
ng_source_path = "_SOURCE_PATH"
ng_modele_ech_filename = "_FILE_PATH_MODELE_ECH"
ng_nom_scenario = "_nom_scenario"
ng_nom_etab = "_nom_etab"
ng_DAR = "_DAR_MOTEUR"
ng_output_path = "_PathSauv"
ng_tf_tla = "_tf_tla"
ng_tf_tlb = "_tf_tlb"
ng_tf_cel = "_tf_cel"
ng_inf_tla = "_inf_tla"
ng_inf_tlb = "_inf_tlb"
ng_inf_cel = "_inf_cel"
ng_reg_tf = "_reg_tf"
ng_reg_inf = "_reg_inf"
ng_table_tx = "_TABLE_TX"
ng_nb_mois_proj = "_NB_MOIS_PROJ"

""" Feuille cle_mc """
ng_ind_sortie = "_ind_sortie"
ng_ind_sortie_eve = "_ind_sortie_eve"
ng_col_sortie = "_col_sortie"
nc_axe_nom = "AXE"
nc_col_sort_restituer = "Restituer"
nc_ind_indic = "INDIC"
nc_ind_restituer = "Restituer"
nc_ind_type = "TYPE"
nc_ind_cat = "CAT"
nc_ind_pas = "PAS"
nc_ind_deb = "MOIS DEB"
nc_ind_fin = "MOIS FIN"
nb_etab_ac = 5

ef_sti = "EF";
em_sti = "EM";
mn_sti = "MN"
gp_ef_sti = "GP TF EF";
gp_em_sti = "GP TF EM";
gpi_ef_sti = "GP INF EF";
gpi_em_sti = "GP INF EM"
gpr_ef_sti = "GP RG EF";
gpr_em_sti = "GP RG EM"
ef_ra_sti = "EFFET RA";
ef_rn_sti = "EFFET RN"

em_eve_sti = "EM EVE"
ef_eve_sti = "EF EVE"
tef_eve_sti = "TEF EVE"
tem_eve_sti = "TEM EVE"
fk_ef_sti = "FLUX CAP EF";
fi_ef_sti = "FLUX INT EF";
fk_em_sti = "FLUX CAP EM";
fi_em_sti = "FLUX INT EM";
fk_em_act_sti = "FLUX CAP ACT EM";
fi_em_act_sti = "FLUX INT ACT EM";
eve_ef_sti = "EVE EF";
eve_em_sti = "EVE EM";
mn_gp_rg_sti = "MN EVE GP RG"
gp_em_eve_sti = "GP TF EVE EM";
gpi_em_eve_sti = "GP INF EVE EM"
gpr_em_eve_sti = "GP RG EVE EM"
gp_liq_em_eve_sti = "GP LQ EVE EM";

indics_eve_tx_st = [em_eve_sti , ef_eve_sti, tef_eve_sti, tem_eve_sti,
                 mn_gp_rg_sti, gp_em_eve_sti, gpi_em_eve_sti, gpr_em_eve_sti, gp_liq_em_eve_sti]

outf_sti = "OUTFLOW"
gp_liq_f_sti = "GP LQ EF";
gp_liq_m_sti = "GP LQ EM"
delta_rl_sti = "DELTA RL"
delta_em_sti = "DELTA EM"
delta_nco_sti = "DELTA NCO"
nco_sti = "NCO"
rl_sti = "RL"
delta_asf_sti = "DELTA ASF"
delta_rsf_sti = "DELTA RSF"
asf_sti = "ASF"
rsf_sti = "RSF"

ef_pni = "EF";
em_pni = "EM";
mn_tx_pni = "MN TX"
mn_lq_pni = "MN LQ";
mn_mg_pni = "MN MG"
mn_pni = "MN"
gp_ef_pni = "GP TF EF";
gp_em_pni = "GP TF EM";
gp_liq_f_pni = "GP LQ EF";
gp_liq_m_pni = "GP LQ EM"
gpi_ef_pni = "GP INF EF";
gpi_em_pni = "GP INF EM"
gpr_ef_pni = "GP RG EF";
gpr_em_pni = "GP RG EM"
tx_cli_pni = "TX CLI";
vol_pn_pni = "VOL PN"

em_eve_pni = "EM EVE"
ef_eve_pni = "EF EVE"
mn_gpr_tx_pni = "MN EVE GP RG TX"
mn_gpr_mg_pni = "MN EVE GP RG MG"
mn_gpr_pni = "MN EVE GP RG"
fk_ef_pni = "FLUX CAP EF";
fi_ef_pni = "FLUX INT EF";
fk_em_pni = "FLUX CAP EM";
fi_em_pni = "FLUX INT EM";
fk_em_act_pni = "FLUX CAP ACT EM";
fi_em_act_pni = "FLUX INT ACT EM";
eve_ef_pni = "EVE EF";
eve_em_pni = "EVE EM";
gp_em_eve_pni = "GP TF EVE EM";
gpi_em_eve_pni = "GP INF EVE EM"
gpr_em_eve_pni = "GP RG EVE EM"
gp_liq_em_eve_pni = "GP LQ EVE EM";

mn_gpliq_tx_pni = "MN EVE GP LIQ TX"
mn_gpliq_mg_pni = "MN EVE GP LIQ MG"
mn_gpliq_pni = "MN EVE GP LIQ"

indics_eve_tx_pn = [em_eve_pni , ef_eve_pni,
                    gp_em_eve_pni, gpi_em_eve_pni, gpr_em_eve_pni, gp_liq_em_eve_pni,
                    mn_gpr_tx_pni, mn_gpr_mg_pni, mn_gpr_pni, mn_gpliq_tx_pni, mn_gpliq_mg_pni, mn_gpliq_pni]

outf_pni = "OUTFLOW"
delta_rl_pni = "DELTA RL"
delta_em_pni = "DELTA EM"
delta_nco_pni = "DELTA NCO"
nco_pni = "NCO"
rl_pni = "RL"
delta_asf_pni = "DELTA ASF"
delta_rsf_pni = "DELTA RSF"
asf_pni = "ASF"
rsf_pni = "RSF"

""" INDICATEURS NECESSAIRES AU CALCUL DES AJUSTEMENTS"""
dependencies_ajust = [em_sti, ef_sti, em_eve_sti, ef_eve_sti]

""" INDICATEURS NECESSAIRES AU CALCUL DU NSFR et LCR """
list_indic_nsfr = [outf_pni + " 0M-6M", outf_pni + " 6M-12M", outf_pni + " 12M-inf"]
list_indic_lcr = [outf_pni + " 0M-1M"]

""" Variables DE SORTIES DE COMPIL"""
nc_output_sc = "SC"
nc_output_key = pa.NC_PA_CLE
nc_output_bilan = pa.NC_PA_BILAN
nc_output_poste = pa.NC_PA_POSTE
nc_output_dim2 = pa.NC_PA_DIM2
nc_output_dim3 = pa.NC_PA_DIM3
nc_output_dim4 = pa.NC_PA_DIM4
nc_output_dim5 = pa.NC_PA_DIM5
nc_output_contrat_cle = pa.NC_PA_CONTRACT_TYPE
nc_output_maturite_cle = pa.NC_PA_MATUR
nc_output_devise_cle = pa.NC_PA_DEVISE
nc_output_index_calc_cle = pa.NC_PA_RATE_CODE
nc_output_index_agreg = pa.NC_PA_INDEX_AGREG
nc_output_marche_cle = pa.NC_PA_MARCHE
nc_output_gestion_cle = pa.NC_PA_GESTION
nc_output_palier_cle = pa.NC_PA_PALIER
nc_output_book_cle = pa.NC_PA_BOOK
nc_output_cust_cle = pa.NC_PA_CUST
nc_output_perimetre_cle = pa.NC_PA_PERIMETRE
nc_output_soc1 = pa.NC_PA_Affectation_Social
nc_output_soc2 = pa.NC_PA_Affectation_Social_2
nc_output_lcr_share = pa.NC_PA_LCR_TIERS_SHARE
nc_output_nsfr1 = pa.NC_PA_DIM_NSFR_1
nc_output_nsfr2 = pa.NC_PA_DIM_NSFR_2
nc_output_r1 = pa.NC_PA_Regroupement_1
nc_output_r2 = pa.NC_PA_Regroupement_2
nc_output_r3 = pa.NC_PA_Regroupement_3
nc_output_bc = pa.NC_PA_Bilan_Cash
nc_output_bcd = pa.NC_PA_Bilan_Cash_Detail
nc_output_bcc = pa.NC_PA_Bilan_Cash_CTA
nc_output_top_mni = pa.NC_PA_TOP_MNI
nc_output_metier = pa.NC_PA_METIER
nc_output_sous_metier = pa.NC_PA_SOUS_METIER
nc_output_zone_geo = pa.NC_PA_ZONE_GEO
nc_output_sous_zone_geo = pa.NC_PA_SOUS_ZONE_GEO
nc_output_bassin_cle = pa.NC_PA_BASSIN
nc_output_etab_cle = pa.NC_PA_ETAB
nc_output_lcr_tiers_cle = pa.NC_PA_LCR_TIERS
nc_output_scope_cle = pa.NC_PA_SCOPE
nc_output_isIG = "IsIG"
nc_output_ind1 = "IND01"
nc_output_ind2 = "IND02"
nc_output_ind3 = pa.NC_PA_IND03

var_compil = [nc_output_bassin_cle, nc_output_etab_cle, nc_output_sc, nc_output_key, nc_output_bilan, nc_output_poste, nc_output_dim2,
              nc_output_dim3, nc_output_dim4, \
              nc_output_dim5, nc_output_contrat_cle, nc_output_maturite_cle, nc_output_devise_cle,
              nc_output_index_calc_cle, nc_output_index_agreg, \
              nc_output_marche_cle, nc_output_gestion_cle, nc_output_palier_cle, nc_output_book_cle, nc_output_cust_cle,
              nc_output_perimetre_cle, nc_output_soc1, \
              nc_output_soc2, nc_output_nsfr1, nc_output_nsfr2, nc_output_r1, nc_output_r2, nc_output_r3, \
              nc_output_bc, nc_output_bcd,
              nc_output_bcc, nc_output_top_mni, nc_output_metier, nc_output_sous_metier, nc_output_zone_geo,
              nc_output_sous_zone_geo, nc_output_lcr_tiers_cle, nc_output_lcr_share, nc_output_scope_cle, nc_output_isIG, nc_output_ind1,
              nc_output_ind2,
              nc_output_ind3]

nom_compil_sc = "CompilSC.csv"
nom_compil_st = "CompilST.csv"
nom_compil_pn = "CompilPN.csv"
nom_compil_st_pn = "CompilST+PN.csv"

""" Feuille STOCK """
ng_stock = "_STOCK_DATA"
stock_lef = pa.NC_PA_LEF
stock_lem = pa.NC_PA_LEM
stock_lmn = pa.NC_PA_LMN
stock_lmn_eve = pa.NC_PA_LMN_EVE
stock_tem = pa.NC_PA_TEM
stock_tef = pa.NC_PA_TEF

cle_stock = [nc_output_bassin_cle, nc_output_etab_cle, nc_output_contrat_cle, nc_output_maturite_cle, nc_output_devise_cle, nc_output_index_calc_cle,
             nc_output_marche_cle, nc_output_gestion_cle, \
             nc_output_palier_cle, nc_output_book_cle, nc_output_cust_cle, nc_output_perimetre_cle, nc_output_top_mni,
             nc_output_lcr_tiers_cle, \
             nc_output_scope_cle]

st_ind = [stock_lef, stock_lem, stock_tef, stock_tem, stock_lmn]
st_num_cols = ["M" + str(i) for i in range(0, 120 + 1)] + ["M" + str(i) for i in range(132, real_max_months + 1, 12)]

ng_pn_nmd = "_pn_nmd"
ng_pn_nmd_prct = "_pn_nmd_prct"
ng_pn_ech = "_pn_ech"
ng_pn_ech_prct = "_pn_ech_prct"

nc_pn_cle_pn = pa.NC_PA_INDEX
nc_pn_jr_pn = pa.NC_PA_JR_PN
nc_pn_profil = pa.NC_PA_AMORTIZING_TYPE
nc_pn_duree = pa.NC_PA_MATURITY_DURATION
nc_pn_periode_amor = pa.NC_PA_AMORTIZING_PERIODICITY
nc_pn_periode_interets = pa.NC_PA_PERIODICITY
nc_pn_periode_fixing = pa.NC_PA_FIXING_PERIODICITY
nc_pn_periode_capi = pa.NC_PA_COMPOUND_PERIODICITY
nc_pn_base_calc = pa.NC_PA_ACCRUAL_BASIS
nc_pn_regle_deblocage = pa.NC_PA_RELEASING_RULE
nc_ech_profil = [nc_pn_jr_pn, nc_pn_profil, nc_pn_duree, nc_pn_base_calc, nc_pn_periode_amor,
                 nc_pn_periode_interets, nc_pn_periode_fixing, nc_pn_periode_capi, nc_pn_base_calc,
                 nc_pn_regle_deblocage]

cle_pn = cle_stock

""" PRICING CURVE COL"""
nc_pricing_curve = "PRICING CURVE"

""" MAPPING Conv_GP_LIQ"""
ng_date_conv_gpliq = "_date_gap_liq_fwd_conv"
ng_conv_gpliq = "_gap_liq_fwd_conv"
ng_conv_gpliq_cle = "CLE"

""" MAPPING Conv_Gaps_NMD """
ng_date_conv_gps_nmd = "_date_gaps_fwd_nmd_conv"
ng_conv_gps_nmd = "_gap_fwd_nmd_conv"
ng_conv_gps_nmd_cle = "CLE"

""" MAPPING ParamsECHLiq """
ng_echliq = "_DATA_ECH_LIQ"
nc_echliq_contrat = "CONTRAT"
nc_echliq_dev = "DEVISE"
nc_echliq_perimetre = "PERIMETRE"
nc_echliq_cle = "CLE"
nc_echliq_unsec = "UNSEC_E3M"
nc_echliq_secur = "SECUR_E3M"
nc_echliq_of = "OFX"
nc_echliq_tci = "TCI_EUR"
nc_echliq_unsec_usd = "UNSEC_U3M"
nc_echliq_tci_usd = "TCI_USD"
echliq_cle = [nc_echliq_cle]

""" MAPPING CURVE ACCRUALS"""
ng_map_curve_accruals = "_MP_CURVE_ACCRUALS2"
cle_map_curve_accruals = ["CURVE_NAME", "TENOR_BEGIN"]
nc_accrual_method_curve_accruals= "ACCRUAL_METHOD"
nc_accrual_conversion_curve_accruals= "ACCRUAL_CONVERSION"
nc_type_courbe_curve_accruals= "TYPE DE COURBE"
nc_type_courbe_alias= "ALIAS"

""" MAPPING CURVE ACCRUALS"""
ng_map_rate_code_curve = "_MAP_RATE_CODE_CURVE"
cle_mp_rate_code_curve  = ["CCY_CODE", "RATE_CODE"]

""" MAPPING ParamsNMD """
ng_nmd = "_DATA_PN_NMD"
ng_nmd_basecalc = "_BASECALC_PN_NMD"
nc_nmd_contrat = "CONTRAT"
nc_nmd_marche = "MARCHE"
nc_nmd_cle = "CLE"
nc_nmd_index = "INDEX"
nc_nmd_flow_mode = "FLOW MOD PN"
nc_nmd_part_stable = "PART STABLE"
nc_nmd_basec_tf = "BASECALC TF"
nc_nmd_basec_tv = "BASECALC TV"
nmd_cle = [nc_nmd_cle]
nmd_cle2 = [nc_nmd_contrat]

""" Feuilles Map1 & MapLIQ_NSFR """

""" MAPPING CONTRATS """
ng_cp = "_MAP_GENERAL"
nc_cp_is_ech_col = "IsECH"
nc_cp_mirr_ig_ntx = "NTX"
nc_cp_mirr_ig_bpce = "BPCE"
nc_cp_mirr_ig_rzo = "RZO"
nc_cp_isech = "IsECH"
nc_cp_contrat_fermat_old = "CONTRAT"
nc_cp_contrat_fermat = "CONTRAT FERMAT"
nc_cp_contrat_old = "CONTRAT PASS"
nc_cp_bilan_old = "CATEGORY"
nc_cp_poste_old = "POSTE AGREG"
nc_cp_poste = nc_output_poste
nc_cp_contrat = nc_output_contrat_cle
nc_cp_bilan = nc_output_bilan
nc_cp_poste_ = nc_output_poste
nc_cp_bassin = "BASSIN"
nc_cp_dim2_old = "DIM 2"
nc_cp_dim3_old = "DIM 3"
nc_cp_dim4_old = "DIM 4"
nc_cp_dim5_old = "DIM 5"
nc_cp_dim2 = nc_output_dim2
nc_cp_dim3 = nc_output_dim3
nc_cp_dim4 = nc_output_dim4
nc_cp_dim5 = nc_output_dim5

"""MAPPING BILAN CASH"""
ng_bc = "_MAP_CONSO_CPT"
nc_bc_contrat_old = "CONTRAT CONSO"
nc_bc_contrat = nc_output_contrat_cle
nc_bc_book_old = "BOOK CODE"
nc_bc_book = nc_output_book_cle
nc_bc_lcr_tiers = nc_output_lcr_tiers_cle
nc_bc_rg3 = "Regroupement 3"
nc_bc_rg2 = "Regroupement 2"
nc_bc_rg1 = "Regroupement 1"
nc_bc_bc1 = "Bilan Cash"
nc_bc_bc2 = "Bilan Cash Detail"
nc_bc_bc3 = "Bilan Cash CTA"
nc_bc_soc1 = "Affectation Social"
bc_cle = [nc_bc_contrat, nc_bc_book, nc_bc_lcr_tiers]

""" MAPPING EMPREINTE MARCHE"""
ng_emm = "_MAP_EM"
nc_emm_bilan = nc_output_bilan
nc_emm_r1 = nc_output_r1
nc_emm_mat = nc_output_maturite_cle
nc_emm_bcd = nc_output_bcd
nc_emm_sco1 = nc_output_soc1
emm_cle = [nc_emm_bilan, nc_emm_r1, nc_emm_mat]

""" BASSINS MAP"""
ng_bassins_mp =  "_MP_BASSIN_SOUS_BASSIN"
nc_cle_bassins_mp = ["SOUS-BASSIN"]

""" Feuille PROFIL ECH FLOORS """
ng_contrat_floor = "_FLOORS_TX_SC"
nc_contrat_floor_cle = "CONTRAT"
nc_value_floor_cle = "FLOOR"

""" MAPPING LIQ IG"""
ng_liq_ig = "_MAP_LIQ_IG"
nc_liq_ig_bilan = nc_output_bilan
nc_liq_ig_bassin = nc_output_bassin_cle
nc_liq_ig_contrat = nc_output_contrat_cle
nc_liq_ig_palier = nc_output_palier_cle
nc_liq_ig_bc = nc_output_bc
nc_liq_ig_bcd = nc_output_bcd
nc_liq_ig_soc1 = nc_output_soc1
liq_ig_cle = [nc_liq_ig_bilan, nc_liq_ig_bc, nc_liq_ig_bassin, nc_liq_ig_contrat, nc_liq_ig_palier]

""" MAPPING LIQ COMPTES """
ng_liq_cmpt = "_MAP_LIQ_CT"
nc_liq_cmpt_bilan = nc_output_bilan
nc_liq_cmpt_r1 = nc_output_r1
nc_liq_cmpt_bassin = nc_output_bassin_cle
nc_liq_cmpt_palier = nc_output_palier_cle
nc_liq_cmpt_soc1 = nc_output_soc1
liq_cmpt_cle = [nc_liq_cmpt_bilan, nc_liq_cmpt_r1, nc_liq_ig_bassin, nc_liq_ig_palier]

""" MAPPING LIQ OPFI """
ng_liq_opfi = "_MAP_LIQ_FI"
nc_liq_opfi_bc = nc_output_bc
nc_liq_opfi_contrat = nc_output_contrat_cle
nc_liq_opfi_bilan = nc_output_bilan
nc_liq_opfi_r1 = nc_output_r1
nc_liq_opfi_bassin = nc_output_bassin_cle
nc_liq_opfi_palier = nc_output_palier_cle
nc_liq_opfi_soc1 = nc_output_soc1
nc_liq_opfi_ighg = "IG/HG Social"
liq_opfi_cle = [nc_liq_opfi_bilan, nc_liq_opfi_bassin, nc_liq_opfi_bc, \
                nc_liq_opfi_r1, nc_liq_opfi_contrat, nc_liq_opfi_ighg]

""" MAPPING SOCIAL AGREGE"""
ng_liq_soc = "_MAP_LIQ_SOC_AGREG"
nc_liq_soc_soc1 = nc_output_soc1
nc_liq_soc_soc2 = nc_output_soc2
liq_soc_cle = [nc_output_soc1]

"""MAPPING NSFR"""
ng_nsfr = "_MAP_NSFR"
nc_nsfr_contrat_old = "CONTRAT_NSFR"
nc_nsfr_lcr_tiers_old = "LCR_TIERS"
nc_nsfr_lcr_tiers = nc_output_lcr_tiers_cle
nc_nsfr_contrat = nc_output_contrat_cle
nsfr_cle = [nc_nsfr_contrat, "LCR TIERS"]
nc_nsfr_dim1 = "DIM NSFR 1"
nc_nsfr_dim2 = "DIM NSFR 2"

ng_itm = "_MapIndexAgreg"
nc_itm_index_calc_old = "RATE CODE"
nc_itm_index_agreg_old = "INDEX_AGREG"
nc_itm_index_calc = nc_output_index_calc_cle
nc_itm_index_agreg = nc_output_index_agreg

"""MAPPING IG"""
ng_igm = "_MapBassinIG"
nc_igm_bassin = "BASSIN"
nc_igm_bassinig = "BASSIN IG"
nc_ig_contrat_new = "CONTRAT IG"
nc_hierarchie = "GROUP"
cle_mc = [nc_cp_contrat]
cle_igc = [nc_cp_contrat, nc_igm_bassinig, nc_hierarchie]

""" MAPPING ParamsECHTx """
ng_echtx = "_DATA_ECH_TX"
nc_echtx_contrat = "CONTRAT"
nc_echtx_basec_tf = "BASECALC TF"
nc_echtx_basec_tv = "BASECALC TV"
nc_echtx_cle = [nc_echtx_contrat]

""" MAPPING PRICING PN ECH"""
ng_ppn = "_DATA_PN_PRICING"
nc_contrat_ppn = "CONTRAT"
nc_dim2_ppn = "DIM2"
nc_bilan_ppn = "BILAN"
nc_tftv_ppn = "TF/TV"
nc_marche_ppn = "MARCHE"
nc_devise_ppn = "DEVISE"
pa.NC_PA_INDEX_calc_ppn = "INDEX CALC"
nc_courbe_pricing_ppn = "COURBE PRICING"
nc_ppn_cle = [nc_contrat_ppn, nc_dim2_ppn, nc_bilan_ppn, nc_tftv_ppn, nc_marche_ppn, nc_devise_ppn, pa.NC_PA_INDEX_calc_ppn]

""" MAPPING ParamsFXSwaps"""
ng_fx_swaps = "_PARAM_FX_SWAP"
nc_contrat_fxsw = "CONTRAT"

"""FEUILLE LCR DAR """
ng_lcr_dar = "_LCR_DAR"
nc_indic_lcr = "INDIC LCR"
nr_rl_lcr_dar = "Réserve de Liquidité"
nr_nco_outflow_lcr_dar = "dont Outflows"
nr_nco_inflow_lcr_dar = "dont Inflows"

"""FEUILLE NSFR DAR """
ng_nsfr_dar = "_NSFR_DAR"
nc_asf_rsf_nsfr_dar = "ASF/RSF"
nc_dim_nsfr1_nsfr_dar = "DIM NSFR 1"
nc_val_nsfr_dar = "ASF/RSF officiel"

"""FEUILLE ParamsLCR """
ng_param_lcr = "_PARAM_LCR"
nc_bilan_param_lcr = nc_output_bilan
nc_contrat_param_lcr = nc_output_contrat_cle
nc_marche_param_lcr = nc_output_marche_cle
nc_nco_param_lcr = "NCO"
nc_rl_param_lcr = "RL"
param_lcr_cle = [nc_bilan_param_lcr, nc_contrat_param_lcr, nc_marche_param_lcr]
ng_param_lcr_meth = "_OUTFLOW_LCR"
nc_bilanc_cc_param_lcr_meth = "Bilan Cash"
nc_outlfow_param_lcr_meth = "APPLIQUER OUTFLOW"

"""FEUILLE ParamsLCR_Spec """
ng_param_lcr_spec = "_PARAM_LCR_SPEC"
nc_bilan_param_lcr_spec = nc_output_bilan
nc_contrat_param_lcr_spec = nc_output_contrat_cle
nc_marche_param_lcr_spec = nc_output_marche_cle
nc_nco_param_lcr_spec = "NCO"
nc_rl_param_lcr_spec = "RL"
param_lcr_spec_cle = [nc_bilan_param_lcr_spec, nc_contrat_param_lcr_spec, nc_marche_param_lcr_spec]

"""FEUILLE ParamsNSFR """
ng_param_nsfr = "_PARAM_NSFR"
nc_nsfr_dim1_param_nsfr = nc_output_nsfr1
nc_asf_rsf_param_nsfr = "ASF/RSF"
nc_nsfr_coeff_param_nsfr = "Coefficient"
nc_nsfr_coeff0_6_param_nsfr = "Coeff Outflow 0-6 mois"
nc_nsfr_coeff6_12_param_nsfr = "Coeff Outflow 6-12 mois"
nc_nsfr_coeff12_inf_param_nsfr = "Coeff Outflow 12-inf mois"
ng_param_nsfr_meth = "_OUTFLOW_NSFR"
nc_bilanc_cc_param_nsfr_meth = "Bilan Cash"
nc_outlfow_param_nsfr_meth = "APPLIQUER OUTFLOW"
ng_param_nsfr_desemc = "_NSFR_DESEMC"

""" Feuille Sc_TX et Sc_LIQ """
maturities_list = ["1D"] + [str(i) + "M" for i in range(1, 13)] + [str(i) + "Y" for i in range(2, 31)]
maturities_F_list = ["1D", "3M", "5Y"]
maturities_infla_list = [str(i) + "Y" for i in range(2, 31)]
label_tv_zero = "TVZERO"
nc_taux = "TAUX"
nc_DAR = "DAR"
ng_tx_header = "_TX_HEADER"
ng_compil_header = "_compil_header"
ng_tx_sc = "_tx"
ng_liq_sc = "_liq"
ng_tx_tci =  "_tci"
NC_TYPE_SC_TX = "TYPE"
NC_DEVISE_SC_TX = "DEVISE"
NC_CODE_SC_TX = "CODE"

""" Feuille Tx_Fermat """
ng_tx_rco = "_rco"
ng_txf_header = "_TXF_HEADER"
ng_txf_zero = "TXF_ZERO"

""" Feuille Sc_ZC """
ng_boot_curves = "_tx_bootsrap"

""" NOM FEUILLES"""
stock_wsh = "STOCK"
pn_nmd_wsh = "PN NMD"
pn_ech_wsh = "PN ECH"
pn_nmd_prct_wsh = "PN NMD%"
pn_ech_prct_wsh = "PN ECH%"

sc_tx_wsh = "Sc_TX"
sc_liq_wsh = "Sc_LIQ"
fermat_tx_wsh = "Fermat_TX"
boot_wsh = "Sc_ZC"

param_lcr_wsh = "ParamsLCR"
param_lcr_spec_wsh = "ParamsLCR_Spec"
param_nsfr_wsh = "ParamsNSFR"
lcr_dar_wsh = "LCR DAR"
nsfr_dar_wsh = "NSFR DAR"

param_echtx_wsh = "ParamsECHTx"
param_echliq_wsh = "ParamsECHLiq"

conv_gpliq_wsh = "Conv_GapLiq"
conv_gp_nmd_wsh = "Conv_Gaps_NMD"
param_fx_swap_wsh = "ParamsFXSwaps"

""" DICO DE L'ENSEMBLE DES FEUILLES"""
all_sheets = {name: value for name, value in locals().items() if type(name) == str and name[-4:] == "_wsh"}

""" ONGLETS A IMPORTER DU FICHIER alim"""
list_tabs_stock = [stock_wsh, lcr_dar_wsh, nsfr_dar_wsh]

""" ONGLETS A IMPORTER DU FICHIER scenario """
list_tabs_sc_tx = [sc_tx_wsh, sc_liq_wsh, fermat_tx_wsh, boot_wsh]
list_tabs_sc_vol = [pn_nmd_wsh, pn_ech_wsh, pn_ech_wsh, pn_nmd_prct_wsh, pn_ech_prct_wsh]
list_tabs_sc_lcr_nsfr = [param_lcr_wsh, param_nsfr_wsh]

""" COMPIL FORMAT """
ng_compil_sep = "_COMPIL_SEP"
ng_compil_merge = "_FUSIONNER_COMPILS"

"""  PERIMETRE IMMO RA/RN/TF"""
CONTRATS_RA_RN_IMMO = ["A-CR-HAB-LIS", "A-CR-HAB-STD", "A-CR-HAB-MOD", "A-CR-HAB-AJU", "A-CR-HAB-STD", "A-PR-STARDEN",
                       "AHB-NS-CR-HAB"]
CONTRATS_RA_RN_IMMO_CSDN = ["A-CR-HAB-BON",  "AHB-NS-CR-HBN"]


List_TX_HEADER = ["TAUX", "DAR", "M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M09", "M10", "M11", "M12",
                  "M13", "M14", "M15", "M16", "M17", "M18", "M19", "M20", "M21", "M22", "M23", "M24", "M25", "M26",
                  "M27", "M28", "M29", "M30", "M31", "M32", "M33", "M34", "M35", "M36", "M37", "M38", "M39", "M40",
                  "M41", "M42", "M43", "M44", "M45", "M46", "M47", "M48", "M49", "M50", "M51", "M52", "M53", "M54",
                  "M55", "M56", "M57", "M58", "M59", "M60", "M61", "M62", "M63", "M64", "M65", "M66", "M67", "M68",
                  "M69", "M70", "M71", "M72", "M73", "M74", "M75", "M76", "M77", "M78", "M79", "M80", "M81", "M82",
                  "M83", "M84", "M85", "M86", "M87", "M88", "M89", "M90", "M91", "M92", "M93", "M94", "M95", "M96",
                  "M97", "M98", "M99", "M100", "M101", "M102", "M103", "M104", "M105", "M106", "M107", "M108", "M109",
                  "M110", "M111", "M112", "M113", "M114", "M115", "M116", "M117", "M118", "M119", "M120", "M121",
                  "M122", "M123", "M124", "M125", "M126", "M127", "M128", "M129", "M130", "M131", "M132", "M133",
                  "M134", "M135", "M136", "M137", "M138", "M139", "M140", "M141", "M142", "M143", "M144", "M145",
                  "M146", "M147", "M148", "M149", "M150", "M151", "M152", "M153", "M154", "M155", "M156", "M157",
                  "M158", "M159", "M160", "M161", "M162", "M163", "M164", "M165", "M166", "M167", "M168", "M169",
                  "M170", "M171", "M172", "M173", "M174", "M175", "M176", "M177", "M178", "M179", "M180", "M181",
                  "M182", "M183", "M184", "M185", "M186", "M187", "M188", "M189", "M190", "M191", "M192", "M193",
                  "M194", "M195", "M196", "M197", "M198", "M199", "M200", "M201", "M202", "M203", "M204", "M205",
                  "M206", "M207", "M208", "M209", "M210", "M211", "M212", "M213", "M214", "M215", "M216", "M217",
                  "M218", "M219", "M220", "M221", "M222", "M223", "M224", "M225", "M226", "M227", "M228", "M229",
                  "M230", "M231", "M232", "M233", "M234", "M235", "M236", "M237", "M238", "M239", "M240"]


compil_header = ["BASSIN", "SC", "CLE", "BILAN", "POSTE", "DIM2", "DIM3", "DIM4", "DIM5", "CONTRAT", "MATUR", "DEVISE",
                 "RATE CODE", "INDEX AGREG", "MARCHE", "GESTION", "PALIER", "BOOKv2", "CUST", "PERIMETRE",
                 "Affectation Liq 1", "Affectation Liq 2", "Affectation Liq 3", "PILOTAGE LIQ BPCE SA",
                 "PILOTAGE LIQ BPCE SA 2", "BPCESA DIM LIQ GLOBALE 1", "BPCESA DIM LIQ GLOBALE 2",
                 "BPCESA DIM LIQ GLOBALE 3", "CT_MLT_LIQ", "DIM NSFR 1", "DIM NSFR 2", "Regroupement 1",
                 "Regroupement 2", "Regroupement 3", "Bilan Cash", "Bilan Cash Detail", "Bilan Cash CTA",
                 "ORGANISATION", "LCR TIERS", "SCOPE", "IsIG", "IND01", "IND02", "IND03", "M00", "M01", "M02", "M03",
                 "M04", "M05", "M06", "M07", "M08", "M09", "M10", "M11", "M12", "M13", "M14", "M15", "M16", "M17",
                 "M18", "M19", "M20", "M21", "M22", "M23", "M24", "M25", "M26", "M27", "M28", "M29", "M30", "M31",
                 "M32", "M33", "M34", "M35", "M36", "M37", "M38", "M39", "M40", "M41", "M42", "M43", "M44", "M45",
                 "M46", "M47", "M48", "M49", "M50", "M51", "M52", "M53", "M54", "M55", "M56", "M57", "M58", "M59",
                 "M60", "M61", "M62", "M63", "M64", "M65", "M66", "M67", "M68", "M69", "M70", "M71", "M72", "M73",
                 "M74", "M75", "M76", "M77", "M78", "M79", "M80", "M81", "M82", "M83", "M84", "M85", "M86", "M87",
                 "M88", "M89", "M90", "M91", "M92", "M93", "M94", "M95", "M96", "M97", "M98", "M99", "M100", "M101",
                 "M102", "M103", "M104", "M105", "M106", "M107", "M108", "M109", "M110", "M111", "M112", "M113", "M114",
                 "M115", "M116", "M117", "M118", "M119", "M120", "M121", "M122", "M123", "M124", "M125", "M126", "M127",
                 "M128", "M129", "M130", "M131", "M132", "M133", "M134", "M135", "M136", "M137", "M138", "M139", "M140",
                 "M141", "M142", "M143", "M144", "M145", "M146", "M147", "M148", "M149", "M150", "M151", "M152", "M153",
                 "M154", "M155", "M156", "M157", "M158", "M159", "M160", "M161", "M162", "M163", "M164", "M165", "M166",
                 "M167", "M168", "M169", "M170", "M171", "M172", "M173", "M174", "M175", "M176", "M177", "M178", "M179",
                 "M180", "M181", "M182", "M183", "M184", "M185", "M186", "M187", "M188", "M189", "M190", "M191", "M192",
                 "M193", "M194", "M195", "M196", "M197", "M198", "M199", "M200", "M201", "M202", "M203", "M204", "M205",
                 "M206", "M207", "M208", "M209", "M210", "M211", "M212", "M213", "M214", "M215", "M216", "M217", "M218",
                 "M219", "M220", "M221", "M222", "M223", "M224", "M225", "M226", "M227", "M228", "M229", "M230", "M231",
                 "M232", "M233", "M234", "M235", "M236", "M237", "M238", "M239", "M240"]