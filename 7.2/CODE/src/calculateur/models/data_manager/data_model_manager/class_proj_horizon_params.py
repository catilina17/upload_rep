import dateutil
class Data_Projection_Horizon_Params():
    """
    Formate les données
    """
    def __init__(self, horizon, dar):
        self.horizon = horizon
        self.load_dar_params(dar)


    def load_horizon_params(self):
        self.max_pn = 60
        self.max_projection = 360
        self.max_proj_ecoul = 1200
        self.max_taux_cms = 300
        self.max_proj_stock = 300

        """ HORIZONS """
        self.nb_months_proj = self.horizon + 1
        if self.horizon > self.max_projection:
            self.nb_months_proj = self.max_projection
            #logger.warning("L'horizon choisi est supérieur au maximum autorisé de 240. L'horizon sera contraint à 240")

        if self.nb_months_proj > self.max_projection:
            self.nb_months_proj = self.max_projection

    def load_dar_params(self, dar):
        """ DAR """
        self.dar_usr = dar
        self.dar_usr = dateutil.parser.parse(str(self.dar_usr)).replace(tzinfo=None)
        self.dar_mois = self.nb_months_in_date(self.dar_usr)

    def nb_months_in_date(self, datum):
        return 12 * datum.year + datum.month
