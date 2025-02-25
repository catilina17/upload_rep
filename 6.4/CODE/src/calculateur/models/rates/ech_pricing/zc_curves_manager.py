import numpy as np
import logging

logger = logging.getLogger(__name__)

class ZC_CURVES():
    """
    Formate les données
    """
    def __init__(self, zc_curves_data, cls_hz_params, cls_cal):
        self.zc_curves_data = zc_curves_data
        self.cls_hz_params = cls_hz_params
        self.dar_usr = self.cls_hz_params.dar_usr
        self.cls_cal = cls_cal
        self.load_columns_names()

    def load_columns_names(self):
        self.ZC_CURVE_NAME = 'ZC_CURVE_NAME'
    def interpolate_all_zc_curves(self):
        self.get_delta_days_from_forward_month()

        self.zc_all_curves= {}
        for curve_zc_name in self.zc_curves_data.iloc[:, 0].unique():
            zc_curve = self.zc_curves_data[self.zc_curves_data.iloc[:, 0] == curve_zc_name]
            zc_curve_interpolated = self.interpol_zc_data(zc_curve)
            self.zc_all_curves[curve_zc_name] = zc_curve_interpolated

        self.zc_all_curves["TVZERO"] = 0 * self.zc_all_curves[list(self.zc_all_curves.keys())[0]]
        self.delta_days_from_fwd_month = np.transpose(self.delta_days_from_fwd_month)

    def get_delta_days_from_forward_month(self):
        dates_fin_per = self.cls_cal.date_fin
        dates_fin_per = dates_fin_per[:, :self.max_duree]
        list_delta_days = []
        for j in range(0, self.max_zc_month):
            delta_days = (dates_fin_per - dates_fin_per[:, j]).astype('timedelta64[D]').reshape(self.max_duree)
            delta_days = np.maximum(0, delta_days / np.timedelta64(1, 'D'))
            list_delta_days.append(delta_days)

        self.delta_days_from_fwd_month = np.column_stack(list_delta_days)

    def interpol_zc_data(self, data_boot):
        data_boot = np.array(data_boot)[:, 1:]
        jr = data_boot[:, 3::2].astype(np.float64)  # On élimine le mois 0
        zc = data_boot[:, 4::2].astype(np.float64)
        listo = []
        for j in range(0, self.max_zc_month):
            delta_days = self.delta_days_from_fwd_month[:,j]
            zc_interpolated = np.interp(delta_days, jr[:, j], zc[:, j], left=0).reshape(1, self.max_duree)
            zc_interpolated = np.where(delta_days==0, 0, zc_interpolated)
            listo.append(zc_interpolated)

        data = np.vstack(listo)

        return data

    def create_zc_curves(self, max_zc, max_duree):
        logger.debug("Interpolation des courbes ZC")
        self.max_zc_month = min(self.cls_hz_params.nb_months_proj, max_zc)
        self.max_duree = max_duree + self.max_zc_month + 5
        self.interpolate_all_zc_curves()