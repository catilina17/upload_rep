from utils import excel_utils as excel_helper
from modules.scenario.services.input_output_services.output_service import logger


def print_pn_template_file(data_df, wb, range_name):
    try:
        excel_helper.clear_range_content(wb, range_name, offset=2)
        excel_helper.write_df_to_range_name_chunk(data_df, range_name, wb, header=False, offset=1, chunk=30000)
    except Exception as e:
        logger.error('Cannot write  in {} of Excel file'.format(range_name))
        logger.error(e, exc_info=True)