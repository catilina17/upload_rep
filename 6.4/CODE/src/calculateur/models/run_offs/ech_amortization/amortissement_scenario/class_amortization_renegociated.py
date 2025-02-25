import numexpr as ne
#import jax as np
try:
    import cupy as cnp
except:
    import numpy as cnp

class Amortization_Renegociated():
    """
    Load les données utilisateurs
    """
    def __init__(self, class_static_amortization):
        self.class_static_amortization = class_static_amortization


    def calculate_amortization_dev2_no_infine(self, credit_depart, rate, current_month, mois_depart_amor, duree, prof_ech_const,
                                              mois_reneg):
        """
        On approxime somme(ppmt) à un dév. limité à l'ordre 2 ce qui donne:
        PPMT = PRINCIP * (1/(duree + rate * duree * (duree - 1)/2) * (1 + (m -1) * rate))
        Somme(i=1 à i=m, PPMT) =  1/2 * PRINCIP * m * A * (2 - 2*rate + rate * (m-1)) avec A = 1/(duree + rate * duree * (duree - 1)/2)

        """
        rate = ne.evaluate('where(~prof_ech_const, 0, rate)')
        relative_month = ne.evaluate('current_month - mois_depart_amor + 1')
        is_credit = ne.evaluate('where(current_month - mois_reneg < 0, 0, 1)')
        relative_month = ne.evaluate('where(relative_month < 0, 0, relative_month)')
        const = self.calc_const_dev2(rate, duree)
        adj_pmt = ne.evaluate('1/2 * credit_depart * relative_month * const * (2 - 2 * rate + rate * (relative_month + 1))')
        rem_cap = ne.evaluate('credit_depart * is_credit - adj_pmt')
        rem_cap = ne.evaluate('where(rem_cap < 0, 0, rem_cap)')
        return rem_cap


    def calc_const_dev2(self, rate, duree):
        non_zero_duree = ne.evaluate("duree != 0")
        const = ne.evaluate('where(non_zero_duree, 1 / (duree + rate * duree * (duree - 1) / 2), 0)')
        return const


    def calculate_amortization_infine(self, credit_depart, current_month, mois_depart_amor, mat_date, dar_mois):
        relative_month = ne.evaluate('current_month - mois_depart_amor + 1')
        rem_cap = ne.evaluate("credit_depart * where(relative_month < 0, 0, 1)")
        rem_cap = ne.evaluate('where(current_month >= mat_date - dar_mois, 0, rem_cap)')
        return rem_cap


    #JAX LIBRAARY

    def calculate_amortization_dev_cupy_no_infine(self, credit_depart, rate, current_month, mois_depart_amor, duree, prof_ech_const,
                                              mois_reneg):

        rate = cnp.where(~prof_ech_const, 0, rate)
        relative_month = current_month - mois_depart_amor + 1
        is_credit = cnp.where(current_month - mois_reneg < 0, 0, 1)
        relative_month = cnp.where(relative_month < 0, 0, relative_month)
        const = self.calc_const_dev_cupy(rate, duree)
        adj_pmt = 1/2 * credit_depart * relative_month * const * (2 - 2 * rate + rate * (relative_month + 1))
        rem_cap = credit_depart * is_credit - adj_pmt
        rem_cap2 = cnp.where(rem_cap < 0, 0, rem_cap)
        return rem_cap2

    def calc_const_dev_cupy(self, rate, duree):
        non_zero_duree = duree != 0
        const = cnp.where(non_zero_duree, 1 / (duree + rate * duree * (duree - 1) / 2), 0)
        return const

    def calculate_amortization_infine_cupy(self, credit_depart, current_month, mois_depart_amor, mat_date, dar_mois):
        relative_month = current_month - mois_depart_amor + 1
        rem_cap = credit_depart * cnp.where(relative_month < 0, 0, 1)
        rem_cap = cnp.where(current_month >= mat_date - dar_mois, 0, rem_cap)
        return rem_cap