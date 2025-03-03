import logging
import traceback
import os
import modules.scenario.utils.entities_resolver as ent
from modules.scenario.services.pn_services.pn_bpce_services.pn_bpce_services import PN_BPCE_Generator
from modules.scenario.services.pn_services.pn_stress_services.pn_services import PN_Manager
from modules.scenario.services.rates_services.rate_scenario_service import Rate_Scenario_Manager
from modules.scenario.services.output_services.rates_output_service import Rate_Exporter
from modules.scenario.services.output_services.output_services import Stock_and_PN_Files_Exporter

logger = logging.getLogger(__name__)

mode_rarn = "dynamic"


class ScenarioManager():
    def __init__(self, cls_user):
        self.up = cls_user

    def process_scenarios(self, ):
        logger.info(' *** Création des scénarios de taux ***')

        self.process_rate_scenarios()

        logger.info(' *** Création des scénarios par établissement ***')

        scenarios_names = self.up.scenario_list[self.up.SCENARIO_NAME_ORIG].unique()
        for etab in self.up.all_etabs:
            for scenario_name in scenarios_names:
                scenario_rows = self.up.scenario_list[self.up.scenario_list[self.up.SCENARIO_NAME_ORIG] == scenario_name]
                etabs2 = ent.get_available_entities(self.up.codification_etab, self.up.all_etabs, scenario_rows)
                if etab in etabs2:
                    etab_output_dir = os.path.join(self.up.output_dir, etab)
                    self.procces_etab_scenario_creation(etab, scenario_name, scenario_rows, etab_output_dir)

    def process_rate_scenarios(self):
        rate_sc_cls = Rate_Scenario_Manager(self.up)
        self.tci_liq_nmd_df = rate_sc_cls.create_rate_scenarios(self.up.scenario_list, self.up.tx_chocs_list,
                                                           self.up.all_etabs)

    def procces_etab_scenario_creation(self, etab, scenario_name, scenario_rows, output_dir):
        try:
            logger.info('  Création du scénario : %s * %s ' % (etab, scenario_name))
            outp = Stock_and_PN_Files_Exporter(self.up, etab, output_dir, scenario_name, scenario_rows)
            pn_output_path = outp.copy_alim_files()
            outp.export_scenario_parameters()

            pn_ser = PN_Manager(self.up, etab, pn_output_path, scenario_rows, output_dir, scenario_name)
            pn_ser.get_type_pn_a_activer()

            tx_out_serv = Rate_Exporter(self.up, etab, output_dir, scenario_name)
            tx_out_serv.export_tx_data_to(scenario_rows, self.tci_liq_nmd_df)

            bpce_ser = PN_BPCE_Generator(self.up, pn_output_path, etab, scenario_name)
            bpce_ser.get_PN_BPCE_service(scenario_rows)

            pn_ser.process_scenario_pn_ajout()
            pn_ser.process_scenario_pn_stress()
            pn_ser.deactivate_non_requested_pns()

            logger.info('  Fichier du scénario %s de %s créé' % (scenario_name, etab))

        except IOError as e:
            logger.error('  {}'.format(e))
            logger.error(traceback.format_exc())
        except ValueError as e:
            logger.error('  {}'.format(e))
            logger.error(traceback.format_exc())
