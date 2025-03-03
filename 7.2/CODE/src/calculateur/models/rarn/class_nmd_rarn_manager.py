import numpy as np
import logging

logger = logging.getLogger(__name__)
np.warnings.filterwarnings("ignore", category=RuntimeWarning)

class RARN_NMD_Manager():
    def __init__(self, cls_proj):
        self.cls_proj = cls_proj
        self.get_rarn_params()

    def get_rarn_params(self):
        n = self.cls_proj.n
        t = self.cls_proj.t
        tx_rn = np.zeros((n, t))
        rate_renego = np.zeros((n, t))
        tx_rarn = np.zeros((n, t))
        self.tx_rn, self.tx_rarn, self.tx_ra = tx_rn, tx_rarn, tx_rarn
        self.rate_renego = rate_renego