import logging
import modules.scenario.parameters.user_parameters_excel as up_excel
import modules.scenario.parameters.user_parameters_json as up_json
from utils import excel_utils as ex
import ntpath

output_dir = None
dar = None
all_etabs = None
source_dir = None
curves_to_bootsrapp = None
zc_file_path = None
tx_curves_path = None
liq_curves_path = None
alim_dir_path = None
stress_dav_list = None
models_list = None
mapping_wb = None
version_sources = None
codification_etab = None
modele_dav_path = None
pn_bpce_sc_list = None
pn_stress_list = None
pn_ajout_list = None
tx_chocs_list = None
scenario_list = None
out_of_recalc_st_etabs = None
scenarii_dav_all = None
scenarii_calc_all = None
pn_a_activer_df = None
st_refs = None
holidays_list = None
stress_pn_list = None
mapping_dile_path = None
mapping_dir = None
name_config = None
stress_model_list = None
tci_curves_path = None
main_sc_eve = None
ir_shock_grp_curve = None

logger = logging.getLogger(__name__)


def get_ihm_parameters(args):
    if args.use_json:
        up_json.get_ihm_parameters_json()  # Charger les paramètres de sortie via JSON
    else:
        up_excel.get_ihm_parameters(ex.xl_interface)
