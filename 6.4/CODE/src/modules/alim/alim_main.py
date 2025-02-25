import modules.alim.parameters.user_parameters as up
import modules.alim.formating_service.stock_formating_service.stock_formating as st
import modules.alim.formating_service.pn_formating_service.pn_formating as pn
import mappings.general_mappings as mp
import modules.alim.pn_reprise_bilan_service.pn_reprise_bilan as pn_evol
import modules.alim.import_export_service.export_data as exp
import utils.excel_utils as ex
import traceback
import modules.alim.utils.logger_config as log_conf
import warnings
import logging



def launch_alim(args):
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    consoleHandler = logging.StreamHandler()
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s", datefmt='%m/%d/%Y %I:%M:%S %p')
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    warnings.filterwarnings('error')

    try:
        logger.info('Lecture des paramètres utilisateurs')
        up.get_user_main_params(args)

        logger.info("Lecture des mappings")
        mp.load_general_mappings(up.mapp_file, up.dar)

        up.create_alim_dir()

        up.get_input_files_name_lcr_nsfr()

        for etab in up.etabs:
            try:
                up.generate_etab_parameters(args, etab)
                logger = log_conf.load_logger(up.log_path, logger)
                logger.info("ALIM du BASSIN: " + up.current_etab)
                logger.info("DOSSIER DE SORTIE: " + up.output_folder)

                [pn_ech_final, pn_nmd_final, data_template_mapped, data_pn_nmd_calage]\
                    = pn.parse_RZO_PN()

                logger.info("LECTURE DE TOUS LES FICHIERS SOURCES DU STOCK")
                STOCK = st.read_and_format_stock_data()

                logger.info("REPRISE DU BILAN: PN%")
                [STOCK, pn_ech_pct, pn_nmd_pct] = pn_evol.calculate_PN_PRCT(STOCK, pn_ech_final, pn_nmd_final)

                logger.info("EXPORTATION DES OUTPUTS")
                exp.export_data(STOCK, data_template_mapped, data_pn_nmd_calage, pn_ech_final, pn_nmd_final,
                                pn_ech_pct, pn_nmd_pct)

                logger.info("FIN DU PROCESSUS D'ALIMENTATION de " + etab)
            except Exception as e:
                logger.error(e)
                logger.error(traceback.format_exc())
                logger.error("L'ALIMENTATION de l'étab %s n'a pas pu aboutir" % etab)


    except KeyboardInterrupt:
        ex.kill_excel()
        print("\nProcess interrupted by the user. Exiting...")
        root = logging.getLogger()
        root.handlers = []

    except:
        logger.error(traceback.format_exc())
        try:
            logger.info(traceback.format_exc())
        except:
            print(traceback.format_exc())
        try:
            ex.kill_excel()
        except:
            pass
    finally:
        try:
            ex.kill_excel()
        except:
            pass
        if not args.use_json:
            input("Type any key...")
