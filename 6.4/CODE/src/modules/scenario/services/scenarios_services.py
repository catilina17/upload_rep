import logging
import traceback
import os
import modules.scenario.services.referential_file_service as ref_services
from utils import excel_utils as ex
from modules.scenario.services.pn_services.pn_bpce_services import pn_bpce_services as bpce_ser
from modules.scenario.services.pn_services.pn_stress_services import pn_services as pn_ser
from modules.scenario.services.rate_shocks_services import rate_and_liq_services as rt_liq_serv
from modules.scenario.services.input_output_services import output_service as out_serv
from modules.scenario.services.rate_shocks_services import taux_files_service as taux_ser
from modules.scenario.services.excel_template_output_services import tx_output_service as tx_out_serv
from modules.scenario.parameters import user_parameters as up

logger = logging.getLogger(__name__)

mode_rarn = "dynamic"


def process_scenarios():
    logger.info('Début de création des fichiers scénarii')

    tci_liq_nmd_df = rt_liq_serv.create_tx_scenarios(up.mapping_wb,  up.scenario_list, up.tx_chocs_list, up.all_etabs)

    scenarios_names =  up.scenario_list["NOM SCENARIO ORIG"].unique()
    for etab in up.all_etabs:
        ref_services.pn_df = {}
        ref_services.stock_data = []
        for scenario_name in scenarios_names:
            scenario_rows =  up.scenario_list[ up.scenario_list["NOM SCENARIO ORIG"] == scenario_name]
            etabs2 = ref_services.get_bassin_name(scenario_rows)
            if etab in etabs2:
                etab_output_dir = os.path.join(up.output_dir, etab)
                procces_etab_scenario_creation(etab, scenario_name, scenario_rows,  up.mapping_wb,  up.pn_bpce_sc_list,
                                               up.pn_stress_list, up.pn_ajout_list, etab_output_dir, tci_liq_nmd_df)

    ex.try_close_workbook(up.mapping_wb, 'Mapping')


def procces_etab_scenario_creation(etab, scenario_name, scenario_rows, mapping_wb, pn_bpce_sc_list, pn_stress_list,
                                   pn_ajout_list, output_dir, tci_liq_nmd_df):
    try:

        tx_wb = None;
        pn_wb = None;

        logger.info('    *********** {}  ********** {} ********** '.format(etab, scenario_name))
        tx_wb, stock_file_path \
            = out_serv.get_output_sc_workbook_and_copy_stock_file(etab, output_dir, scenario_name)

        out_serv.export_scenario_parameters(output_dir, scenario_name,
                                            scenario_rows['SC CALCULATEUR'].iloc[0],
                                            scenario_rows['SURCOUCHE DAV'].iloc[0],
                                            scenario_rows['SC MODELE'].iloc[0])

        type_pn = pn_ser.get_type_pn_a_activer(etab)

        export_all_rates_curves(scenario_rows, tci_liq_nmd_df, tx_wb, etab)

        pn_wb = out_serv.get_output_sc_workbook_pn(etab, output_dir, scenario_name)

        bpce_ser.get_PN_BPCE_service(pn_bpce_sc_list, scenario_rows, pn_wb, etab, mapping_wb,
                                     scenario_name)

        pn_ser.process_scenario_pn_ajout(pn_ajout_list, scenario_rows, pn_wb, type_pn, mapping_wb, etab)

        pn_ser.process_scenario_pn_stress(pn_stress_list, scenario_rows, pn_wb, type_pn, etab)

        pn_ser.deactivate_non_requested_pns(pn_wb, type_pn)


        logger.info('    Fichier du scénario  @_{} de {} créé'.format(scenario_name, etab))
        logger.info('    **************      **************      *************'.format(scenario_name))

    except IOError as e:
        logger.error('  {}'.format(e))
        logger.error(traceback.format_exc())
    except ValueError as e:
        logger.error('  {}'.format(e))
        logger.error(traceback.format_exc())
    if tx_wb is not None:
        ex.try_close_workbook(tx_wb, scenario_name, True)
    if pn_wb is not None:
        ex.try_close_workbook(pn_wb, scenario_name, True)


def export_all_rates_curves(scenario_rows, tci_liq_nmd_df, wb, etab):
    scenario_curves_df = taux_ser.get_sc_tx_df(scenario_rows['Shocked TX'].iloc[0])
    bootstrap_df = taux_ser.get_bootstrap_df(scenario_rows['Shocked TX'].iloc[0])
    reference_tx_curves_df = taux_ser.get_stock_sc_tx_df(etab)
    tx_out_serv.export_tx_data_to(reference_tx_curves_df, scenario_curves_df, bootstrap_df, tci_liq_nmd_df, wb)
