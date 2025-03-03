import modules.moteur.parameters.user_parameters as up
import numpy as np
import modules.moteur.parameters.general_parameters as gp
import pandas as pd
import modules.moteur.utils.generic_functions as gf
import logging

logger = logging.getLogger(__name__)


def calculate_outflows(dic_ind, typo, dic_data=[], mat_ef=[], cols=[]):
    """ fonction permettant de calculer les outlfows """
    if typo == "ST":
        ef_ind = gp.ef_sti
    else:
        ef_ind = gp.ef_pni

    ht = dic_ind[ef_ind].shape[0]

    ino = False

    for indic, deb_fin in up.indic_sortie[typo + "_" + "OUTFLOW"].items():
        ino = True
        deb = deb_fin[0]
        fin = deb_fin[1]
        list_outflow = []
        deb_mois = 1 if typo == "PN" else 0
        if typo == "AJUST":
            num_cols = [("M0" if j <= 9 else "M") + str(j) for j in range(deb_mois, up.nb_mois_proj_usr + 1)]
        else:
            num_cols = ["M" + str(j) for j in range(deb_mois, up.nb_mois_proj_usr + 1)]

        if fin != "inf":
            max_mois_outflow = gp.max_months - fin + 1
        else:
            max_mois_outflow = gp.max_months - deb + 1

        for j in range(deb_mois, min(up.nb_mois_proj_usr + 1, max_mois_outflow)):
            if j > up.nb_mois_proj_out:
                data_temp = np.zeros((ht, 1))
            else:
                if typo == "ST":
                    if fin != "inf":
                        data_temp = np.array(dic_ind[ef_ind]["M" + str(j + deb)] - dic_ind[ef_ind]["M" + str(j + fin)])
                    else:
                        data_temp = np.array(dic_ind[ef_ind]["M" + str(j + deb)])
                elif typo == "AJUST":
                    pref = "M" if j + deb >= 10 else "M0"
                    data_temp = np.array(dic_ind[ef_ind][pref + str(j + deb)])
                else:
                    data_temp = gf.sum_each2(mat_ef, cols, proj=True, per=j, fix=True)
                    if fin != "inf":
                        data_temp = np.array(data_temp["M" + str(j + deb)] - data_temp["M" + str(j + fin)])
                    else:
                        data_temp = np.array(data_temp["M" + str(j + deb)])
            list_outflow.append(data_temp)

        if len(list_outflow) < up.nb_mois_proj_usr - deb_mois + 1:
            list_outflow = list_outflow + [np.zeros(list_outflow[0].shape[0])] * (
                        up.nb_mois_proj_usr - len(list_outflow) + 1 - deb_mois)

        outflow = np.column_stack(list_outflow)
        if typo == "AJUST":
            dic_ind[indic] = dic_ind[ef_ind].copy()
            dic_ind[indic][gp.nc_output_ind3] = indic
            dic_ind[indic][num_cols] = outflow
        else:
            indexo = dic_data["data"].index if typo == "PN" else dic_data["stock"].index.get_level_values("new_key")
            dic_ind[indic] = pd.DataFrame(outflow, index=indexo, columns=num_cols)

    if ino:
        del data_temp;
        outflow;
        for d in list_outflow:
            del d;
        del list_outflow
