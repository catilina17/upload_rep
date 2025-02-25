# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:01:46 2020

DEPENDANCE INDICATEURS
"""
import modules.moteur.parameters.general_parameters as gp
import modules.moteur.parameters.user_parameters as up

global gap_liq_ef_pn, gap_liq_em_pn, gap_liq_ef_st, gap_liq_em_st, gap_liq_ef_aj, gap_liq_em_aj
global gap_tx_ef_st, gap_tx_em_st, gap_inf_ef_st, gap_inf_em_st, gap_reg_ef_st, gap_reg_em_st
global gap_tx_ef_pn, gap_tx_em_pn, gap_inf_ef_pn, gap_inf_em_pn, gap_reg_ef_pn, gap_reg_em_pn
global gap_tx_ef_aj, gap_tx_em_aj, gap_inf_ef_aj, gap_inf_em_aj, gap_reg_ef_aj, gap_reg_em_aj
global mni_gpr_pn, mni_gpr_tx_pn, mni_gpr_lq_pn, mni_gpr_mg_pn
global gap_liq_eve_em_pn, gap_liq_eve_em_st, gap_tx_eve_em_st, gap_tx_eve_em_pn
global gap_inf_eve_em_pn, gap_inf_eve_em_st, gap_reg_eve_em_st, gap_reg_eve_em_pn
global mni_gpliq_pn, mni_gpliq_tx_pn, mni_gpliq_mg_pn
global mni_gpliq_tx_aj, mni_gpliq_aj, mni_gpliq_mg_aj

gap_liq_ef_pn = {};gap_liq_em_pn = {};gap_liq_ef_st = {};
gap_liq_em_st = {};gap_liq_ef_aj = {};gap_liq_em_aj = {}
gap_tx_ef_st = {};gap_tx_em_st = {};gap_inf_ef_st = {};
gap_inf_em_st = {};gap_reg_ef_st = {};gap_reg_em_st = {};
gap_tx_ef_pn = {};gap_tx_em_pn = {};gap_inf_ef_pn = {};
gap_inf_em_pn = {};gap_reg_ef_pn = {};gap_reg_em_pn = {};
gap_tx_ef_aj = {};gap_tx_em_aj = {};gap_inf_ef_aj = {};
gap_inf_em_aj = {};gap_reg_ef_aj = {};gap_reg_em_aj = {};
mni_gpr_pn = {}; mni_gpr_tx_pn = {}
mni_gpr_lq_pn = {};
mni_gpr_mg_pn  = {}
mni_gpr_aj = {}; mni_gpr_tx_aj = {}
mni_gpr_lq_aj = {};
mni_gpr_mg_aj  = {}

mni_gpliq_pn = {};mni_gpliq_tx_pn={};mni_gpliq_mg_pn={}
mni_gpliq_aj = {};mni_gpliq_tx_aj={};mni_gpliq_mg_aj={}

gap_liq_eve_em_pn = {}; gap_liq_eve_em_st = {}; gap_tx_eve_em_st = {}; gap_tx_eve_em_pn = {}
gap_inf_eve_em_pn = {}; gap_inf_eve_em_st = {}; gap_reg_eve_em_st = {}; gap_reg_eve_em_pn = {}

gap_liq_eve_em_aj = {};  gap_tx_eve_em_aj = {}
gap_inf_eve_em_aj = {}; gap_reg_eve_em_aj = {}


def generate_dependencies_indic():
    global gap_liq_ef_pn, gap_liq_em_pn, gap_liq_ef_st, gap_liq_em_st, gap_liq_ef_aj, gap_liq_em_aj, gap_tx_ef_st, gap_tx_em_st
    global gap_tx_ef_st, gap_tx_em_st, gap_inf_ef_st, gap_inf_em_st, gap_reg_ef_st, gap_reg_em_st
    global gap_tx_ef_pn, gap_tx_em_pn, gap_inf_ef_pn, gap_inf_em_pn, gap_reg_ef_pn, gap_reg_em_pn
    global gap_tx_ef_aj, gap_tx_em_aj, gap_inf_ef_aj, gap_inf_em_aj, gap_reg_ef_aj, gap_reg_em_aj
    global mni_gpr_pn, mni_gpr_tx_pn, mni_gpr_mg_pn
    global mni_gpliq_pn, mni_gpliq_tx_pn, mni_gpliq_mg_pn
    global gap_liq_eve_em_pn, gap_liq_eve_em_st, gap_tx_eve_em_st, gap_tx_eve_em_pn
    global gap_inf_eve_em_pn, gap_inf_eve_em_st, gap_reg_eve_em_st, gap_reg_eve_em_pn
    global gap_liq_eve_em_aj, gap_tx_eve_em_aj
    global gap_inf_eve_em_aj, gap_reg_eve_em_aj
    global mni_gpr_aj, mni_gpr_tx_aj, mni_gpr_lq_aj, mni_gpr_mg_aj
    global mni_gpliq_tx_aj, mni_gpliq_aj, mni_gpliq_mg_aj

    for p in range(0, up.nb_mois_proj_out + 1):

        """ GAP LIQUIDITE """
        if p>=1:
            gap_liq_ef_pn[p] = gp.gp_liq_f_pni + str(p) in up.indic_sortie["PN"] \
                               or gp.gpi_ef_pni + str(p) in up.indic_sortie["PN"] \
                               or gp.gpr_ef_pni + str(p) in up.indic_sortie["PN"] \
                               or gp.gp_ef_pni + str(p) in up.indic_sortie["PN"]

            gap_liq_em_pn[p] = gp.gp_liq_m_pni + str(p) in up.indic_sortie["PN"] \
                               or gp.gpi_em_pni + str(p) in up.indic_sortie["PN"] \
                               or gp.gpr_em_pni + str(p) in up.indic_sortie["PN"] \
                               or gp.gp_em_pni + str(p) in up.indic_sortie["PN"]

            gap_liq_eve_em_pn[p] = gp.gp_liq_em_eve_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.gpi_em_eve_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.gpr_em_eve_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.gp_em_eve_pni+ str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fi_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fk_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fi_em_act_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fk_em_act_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.eve_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.mn_gpr_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.mn_gpr_tx_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.mn_gpr_mg_pni + str(p) in up.indic_sortie_eve["PN"]

        else:
            gap_liq_ef_pn[p] = False
            gap_liq_em_pn[p] = False
            gap_liq_eve_em_pn[p] = False

        gap_liq_ef_st[p] = gp.gp_liq_f_sti + str(p) in up.indic_sortie["ST"] \
                               or gp.gpi_ef_sti + str(p) in up.indic_sortie["ST"] \
                               or gp.gpr_ef_sti + str(p) in up.indic_sortie["ST"] \
                               or gp.gp_ef_sti+ str(p) in up.indic_sortie["ST"]

        gap_liq_em_st[p] = gp.gp_liq_m_sti + str(p) in up.indic_sortie["ST"] \
                               or gp.gpi_em_sti + str(p) in up.indic_sortie["ST"] \
                               or gp.gpr_em_sti + str(p) in up.indic_sortie["ST"] \
                               or gp.gp_em_sti+ str(p) in up.indic_sortie["ST"] \
                               or gp.fi_em_sti + str(p) in up.indic_sortie["ST"] \
                               or gp.fk_em_sti + str(p) in up.indic_sortie["ST"] \
                               or gp.fi_em_act_sti + str(p) in up.indic_sortie["ST"] \
                               or gp.fk_em_act_sti + str(p) in up.indic_sortie["ST"] \
                               or gp.eve_em_sti + str(p) in up.indic_sortie["ST"]

        gap_liq_eve_em_st[p] = gp.gp_liq_em_eve_sti + str(p) in up.indic_sortie_eve["ST"] \
                               or gp.gpi_em_eve_sti + str(p) in up.indic_sortie_eve["ST"] \
                               or gp.gpr_em_eve_sti + str(p) in up.indic_sortie_eve["ST"] \
                               or gp.gp_em_eve_sti+ str(p) in up.indic_sortie_eve["ST"] \
                               or gp.fi_em_sti + str(p) in up.indic_sortie_eve["ST"] \
                               or gp.fk_em_sti + str(p) in up.indic_sortie_eve["ST"] \
                               or gp.fi_em_act_sti + str(p) in up.indic_sortie_eve["ST"] \
                               or gp.fk_em_act_sti + str(p) in up.indic_sortie_eve["ST"] \
                               or gp.eve_em_sti + str(p) in up.indic_sortie_eve["ST"]

        gap_liq_ef_aj[p] = gp.gp_liq_f_pni + str(p) in up.indic_sortie["AJUST"]

        gap_liq_em_aj[p] = gp.gp_liq_m_pni + str(p) in up.indic_sortie["AJUST"]

        gap_liq_eve_em_aj[p] = gp.gp_liq_em_eve_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                               or gp.gpi_em_eve_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                               or gp.gpr_em_eve_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                               or gp.gp_em_eve_pni+ str(p) in up.indic_sortie_eve["AJUST"] \
                               or gp.fi_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                               or gp.fk_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                               or gp.fi_em_act_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                               or gp.fk_em_act_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                               or gp.eve_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                               or gp.mn_gpr_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                               or gp.mn_gpr_tx_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                               or gp.mn_gpr_mg_pni + str(p) in up.indic_sortie_eve["AJUST"]

        """ GAP TAUX """
        gap_tx_ef_st[p] = gp.gp_ef_sti + str(p) in up.indic_sortie["ST"]\
                          or gp.gpi_ef_sti + str(p) in up.indic_sortie["ST"] \
                          or gp.gpr_ef_sti + str(p) in up.indic_sortie["ST"]


        gap_tx_em_st[p] = gp.gp_em_sti + str(p) in up.indic_sortie["ST"] \
                          or gp.gpi_em_sti + str(p) in up.indic_sortie["ST"] \
                          or gp.gpr_em_sti + str(p) in up.indic_sortie["ST"]

        gap_tx_eve_em_st[p] = (gp.gp_em_eve_sti + str(p) in up.indic_sortie_eve["ST"] \
                          or gp.gpi_em_eve_sti + str(p) in up.indic_sortie_eve["ST"] \
                          or gp.gpr_em_eve_sti + str(p) in up.indic_sortie_eve["ST"] \
                          or gp.fk_em_sti + str(p) in up.indic_sortie_eve["ST"] \
                          or gp.fi_em_sti + str(p) in up.indic_sortie_eve["ST"] \
                          or gp.fk_em_act_sti + str(p) in up.indic_sortie_eve["ST"] \
                          or gp.fi_em_act_sti + str(p) in up.indic_sortie_eve["ST"] \
                          or gp.eve_em_sti + str(p) in up.indic_sortie_eve["ST"])   and (up.type_eve!="ICAAP")

        gap_inf_ef_st[p] = gp.gpi_ef_sti + str(p) in up.indic_sortie["ST"] \
                           or gp.gpr_ef_sti + str(p) in up.indic_sortie["ST"]

        gap_inf_em_st[p] = gp.gpi_em_sti + str(p) in up.indic_sortie["ST"] \
                           or gp.gpr_em_sti + str(p) in up.indic_sortie["ST"]

        gap_inf_eve_em_st[p] = (gp.gpi_em_eve_sti + str(p) in up.indic_sortie_eve["ST"] \
                           or gp.gpr_em_eve_sti + str(p) in up.indic_sortie_eve["ST"] \
                           or gp.eve_em_sti + str(p) in up.indic_sortie_eve["ST"] \
                           or gp.fi_em_sti + str(p) in up.indic_sortie_eve["ST"] \
                           or gp.fk_em_sti + str(p) in up.indic_sortie_eve["ST"] \
                           or gp.fi_em_act_sti + str(p) in up.indic_sortie_eve["ST"] \
                           or gp.fk_em_act_sti + str(p) in up.indic_sortie_eve["ST"])   and (up.type_eve!="ICAAP")

        gap_reg_ef_st[p] = gp.gpr_ef_sti + str(p) in up.indic_sortie["ST"]

        gap_reg_em_st[p] = gp.gpr_em_sti + str(p) in up.indic_sortie["ST"]

        gap_reg_eve_em_st[p] = (gp.gpr_em_eve_sti + str(p) in up.indic_sortie_eve["ST"] \
                           or gp.eve_em_sti + str(p) in up.indic_sortie_eve["ST"] \
                           or gp.fi_em_sti + str(p) in up.indic_sortie_eve["ST"] \
                           or gp.fk_em_sti + str(p) in up.indic_sortie_eve["ST"] \
                           or gp.fi_em_act_sti + str(p) in up.indic_sortie_eve["ST"] \
                           or gp.fk_em_act_sti + str(p) in up.indic_sortie_eve["ST"])   and (up.type_eve!="ICAAP")

        if p>=1:
            gap_tx_ef_pn[p] = gp.gp_ef_pni + str(p) in up.indic_sortie["PN"] \
                              or gp.gpi_ef_pni + str(p) in up.indic_sortie["PN"] \
                              or gp.gpr_ef_pni + str(p) in up.indic_sortie["PN"]

            gap_tx_em_pn[p] = gp.gp_em_pni + str(p) in up.indic_sortie["PN"] \
                              or gp.gpi_em_pni + str(p) in up.indic_sortie["PN"] \
                              or gp.gpr_em_pni + str(p) in up.indic_sortie["PN"]

            gap_tx_eve_em_pn[p] = (gp.gp_em_eve_pni + str(p) in up.indic_sortie_eve["PN"] \
                              or gp.gpi_em_eve_pni + str(p) in up.indic_sortie_eve["PN"] \
                              or gp.gpr_em_eve_pni + str(p) in up.indic_sortie_eve["PN"] \
                              or gp.fk_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                              or gp.fi_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                              or gp.fk_em_act_pni + str(p) in up.indic_sortie_eve["PN"] \
                              or gp.fi_em_act_pni + str(p) in up.indic_sortie_eve["PN"] \
                              or gp.eve_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                              or gp.mn_gpr_pni + str(p) in up.indic_sortie_eve["PN"] \
                              or gp.mn_gpr_tx_pni + str(p) in up.indic_sortie_eve["PN"] \
                              or gp.mn_gpr_mg_pni + str(p) in up.indic_sortie_eve["PN"])  and (up.type_eve != "ICAAP")

            gap_inf_ef_pn[p] = gp.gpi_ef_pni + str(p) in up.indic_sortie["PN"] \
                               or gp.gpr_ef_pni + str(p) in up.indic_sortie["PN"]


            gap_inf_em_pn[p] = gp.gpi_em_pni + str(p) in up.indic_sortie["PN"] \
                               or gp.gpr_em_pni + str(p) in up.indic_sortie["PN"]


            gap_inf_eve_em_pn[p] = (gp.gpi_em_eve_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.gpr_em_eve_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.eve_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fi_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fk_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fi_em_act_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fk_em_act_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fi_ef_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.eve_ef_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.mn_gpr_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.mn_gpr_tx_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.mn_gpr_mg_pni + str(p) in up.indic_sortie_eve["PN"])   and (up.type_eve != "ICAAP")

            gap_reg_ef_pn[p] = gp.gpr_ef_pni + str(p) in up.indic_sortie["PN"]

            gap_reg_em_pn[p] = gp.gpr_em_pni + str(p) in up.indic_sortie["PN"]

            gap_reg_eve_em_pn[p] = (gp.gpr_em_eve_pni + str(p) in up.indic_sortie_eve["PN"]\
                               or gp.eve_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fi_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fk_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fi_em_act_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fk_em_act_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.mn_gpr_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.mn_gpr_tx_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.mn_gpr_mg_pni + str(p) in up.indic_sortie_eve["PN"])  and (up.type_eve != "ICAAP")

            mni_gpr_pn[p] =    (gp.mn_gpr_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.eve_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fi_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fi_em_act_pni + str(p) in up.indic_sortie_eve["PN"])  and (up.type_eve != "ICAAP")


            mni_gpr_tx_pn[p] = (gp.mn_gpr_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.eve_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fi_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fi_em_act_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.mn_gpr_tx_pni + str(p) in up.indic_sortie_eve["PN"])   and (up.type_eve!="ICAAP")

            mni_gpr_mg_pn[p] = (gp.mn_gpr_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.eve_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fi_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fi_em_act_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.mn_gpr_mg_pni + str(p) in up.indic_sortie_eve["PN"]) and (up.type_eve!="ICAAP")

            mni_gpliq_pn[p] =  (gp.eve_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fi_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fi_em_act_pni + str(p) in up.indic_sortie_eve["PN"]) and (up.type_eve=="ICAAP")


            mni_gpliq_tx_pn[p] = (gp.eve_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fi_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fi_em_act_pni + str(p) in up.indic_sortie_eve["PN"]) and (up.type_eve=="ICAAP")

            mni_gpliq_mg_pn[p] = (gp.eve_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fi_em_pni + str(p) in up.indic_sortie_eve["PN"] \
                               or gp.fi_em_act_pni + str(p) in up.indic_sortie_eve["PN"]) and (up.type_eve=="ICAAP")

        else:
            gap_tx_ef_pn[p] = False

            gap_tx_em_pn[p] = False

            gap_tx_eve_em_pn[p] = False

            gap_inf_ef_pn[p] = False

            gap_inf_em_pn[p] = False

            gap_inf_eve_em_pn[p] = False

            gap_reg_ef_pn[p] = False

            gap_reg_em_pn[p] = False

            gap_reg_eve_em_pn[p] = False

            mni_gpr_pn[p] = False

            mni_gpr_tx_pn[p] = False

            mni_gpr_lq_pn[p] = False

            mni_gpr_mg_pn[p] = False


        gap_tx_ef_aj[p] = gp.gp_ef_pni + str(p) in up.indic_sortie["AJUST"] \
                          or gp.gpr_ef_pni + str(p) in up.indic_sortie["AJUST"]

        gap_tx_em_aj[p] = gp.gp_em_pni + str(p) in up.indic_sortie["AJUST"] \
                          or gp.gpr_em_pni + str(p) in up.indic_sortie["AJUST"]

        gap_tx_eve_em_aj[p] = (gp.gp_em_eve_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                               or gp.gpr_em_eve_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                               or gp.eve_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                               or gp.fi_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                               or gp.fk_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                               or gp.fi_em_act_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                               or gp.fk_em_act_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                               or gp.mn_gpr_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                               or gp.mn_gpr_tx_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                               or gp.mn_gpr_mg_pni + str(p) in up.indic_sortie_eve["AJUST"])  and (up.type_eve!="ICAAP")

        gap_inf_ef_aj[p] = gp.gpi_ef_pni + str(p) in up.indic_sortie["AJUST"] \
                           or gp.gpr_ef_pni + str(p) in up.indic_sortie["AJUST"]

        gap_inf_em_aj[p] = gp.gpi_em_pni + str(p) in up.indic_sortie["AJUST"]\
                           or gp.gpr_em_pni + str(p) in up.indic_sortie["AJUST"]

        gap_inf_eve_em_aj[p] = (gp.gpi_em_eve_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                                or gp.gpr_em_eve_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                                or gp.eve_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                                or gp.fi_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                                or gp.fk_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                                or gp.fi_em_act_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                                or gp.fk_em_act_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                                or gp.mn_gpr_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                                or gp.mn_gpr_tx_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                                or gp.mn_gpr_mg_pni + str(p) in up.indic_sortie_eve["AJUST"])  and (up.type_eve!="ICAAP")

        gap_reg_ef_aj[p] = gp.gpr_ef_pni + str(p) in up.indic_sortie["AJUST"]

        gap_reg_em_aj[p] = gp.gpr_em_pni + str(p) in up.indic_sortie["AJUST"]

        gap_reg_eve_em_aj[p] = (gp.gpr_em_eve_pni + str(p) in up.indic_sortie_eve["AJUST"]\
                           or gp.eve_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                           or gp.fi_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                           or gp.fk_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                           or gp.fi_em_act_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                           or gp.fk_em_act_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                           or gp.mn_gpr_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                           or gp.mn_gpr_tx_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                           or gp.mn_gpr_mg_pni + str(p) in up.indic_sortie_eve["AJUST"])  and (up.type_eve!="ICAAP") \

        mni_gpr_aj[p] = (gp.mn_gpr_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                        or gp.eve_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                        or gp.fi_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                        or gp.fi_em_act_pni + str(p) in up.indic_sortie_eve["AJUST"]) and (up.type_eve!="ICAAP")

        mni_gpr_tx_aj[p] = (gp.mn_gpr_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                           or gp.eve_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                           or gp.fi_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                           or gp.fi_em_act_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                           or gp.mn_gpr_tx_pni + str(p) in up.indic_sortie_eve["AJUST"])  and (up.type_eve!="ICAAP")

        mni_gpr_lq_aj[p] = (gp.mn_gpr_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                           or gp.eve_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                           or gp.fi_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                           or gp.fi_em_act_pni + str(p) in up.indic_sortie_eve["AJUST"])  and (up.type_eve!="ICAAP")

        mni_gpr_mg_aj[p] = (gp.mn_gpr_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                           or gp.eve_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                           or gp.fi_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                           or gp.fi_em_act_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                           or gp.mn_gpr_mg_pni + str(p) in up.indic_sortie_eve["AJUST"])  and (up.type_eve!="ICAAP")

        mni_gpliq_aj[p] = (gp.eve_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                           or gp.fi_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                           or gp.fi_em_act_pni + str(p) in up.indic_sortie_eve["AJUST"]) and (up.type_eve == "ICAAP")

        mni_gpliq_tx_aj[p] = (gp.eve_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                              or gp.fi_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                              or gp.fi_em_act_pni + str(p) in up.indic_sortie_eve["AJUST"]) and (up.type_eve == "ICAAP")

        mni_gpliq_mg_aj[p] = (gp.eve_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                              or gp.fi_em_pni + str(p) in up.indic_sortie_eve["AJUST"] \
                              or gp.fi_em_act_pni + str(p) in up.indic_sortie_eve["AJUST"]) and (up.type_eve == "ICAAP")
