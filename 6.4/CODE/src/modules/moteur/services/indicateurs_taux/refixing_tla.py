import modules.moteur.parameters.user_parameters as up
import modules.moteur.parameters.general_parameters as gp
import modules.moteur.mappings.main_mappings as mp
import numpy as np


def refix_tla_stock(data_stock, is_ech_stock, dic_stock_sci):
    """ FILTRE TLA, TLB, LEP, CEL """
    filtre_tla = (np.array(is_ech_stock) == False) & (
        np.array(data_stock[gp.nc_output_index_calc_cle].isin(mp.all_gap_gestion_index))) & np.array([up.retraitement_tla])

    """ REFIXING TLA """
    if up.retraitement_tla:
        dic_stock_sci["tef"].loc[filtre_tla, ["M" + str(x) for x in range(1, up.mois_refix_tla + 1)]] = \
            dic_stock_sci[gp.ef_sti].loc[filtre_tla, ["M" + str(x) for x in range(1, up.mois_refix_tla + 1)]]
        dic_stock_sci["tef" ].loc[
            filtre_tla, ["M" + str(x) for x in range(up.mois_refix_tla + 1, up.nb_mois_proj_usr + 1)]] = 0

        dic_stock_sci["tem"].loc[filtre_tla, ["M" + str(x) for x in range(1, up.mois_refix_tla + 1)]] = \
            dic_stock_sci[gp.em_sti].loc[filtre_tla, ["M" + str(x) for x in range(1, up.mois_refix_tla + 1)]]
        dic_stock_sci["tem"].loc[
            filtre_tla, ["M" + str(x) for x in range(up.mois_refix_tla + 1, up.nb_mois_proj_usr + 1)]] = 0