import numpy as np
import scipy.special
from scipy.special import erf

class SFLaw(object):

    @property
    def scrit(self, *args):
        pass

    @property
    def SFRff(self, *args):
        pass

    @property
    def sigma_s(self):
        if np.isinf(self.Beta):
            return np.log(1 + self.b**2 * self.Mach**2)
        else:
            return np.log(1 + self.b**2 * self.Mach**2 * (self.Beta/(self.Beta+1)))


class KM2005(SFLaw):
    """
    Krumholz & McKee 2005 law as parametrized by Federrath & Klessen 2012
    """
    name = 'KM2005'

    def __init__(self, phi_x=0.12, alpha_vir=1, Mach=10, Beta=np.inf,
                 epsilon=0.3, phi_t=1./3., b=0.4):
        self.phi_x = phi_x
        self.alpha_vir = alpha_vir
        self.Mach = Mach
        self.Beta = Beta
        self.phi_t = phi_t
        self.epsilon = epsilon
        self.b = b

    @property
    def scrit(self):
        if np.isinf(self.Beta):
            return np.log((np.pi**2/5.) * self.phi_x**2 * self.alpha_vir *
                          self.Mach**2)
        else:
            return np.log((np.pi**2/5.) * self.phi_x**2 * self.alpha_vir *
                          self.Mach**2 * (1-self.Beta**-1)**-1)

    def SFRff(self):
        return (self.epsilon / self.phi_t / 2. *
                (1 + scipy.special.erf((self.sigma_s**2 -
                                        2*self.scrit) /
                                       (8*self.sigma_s**2)**0.5)))

class PN2011(SFLaw):
    """
    Padoan & Nordlund 2011 law as parametrized by Federrath & Klessen 2012
    """
    name = 'PN2011'

    def __init__(self, theta=0.65, alpha_vir=1, Mach=10, Beta=np.inf,
                 epsilon=0.3, phi_t=1./1.5, b=0.4):
        self.theta = theta
        self.alpha_vir = alpha_vir
        self.Mach = Mach
        self.Beta = Beta
        self.phi_t = phi_t
        self.epsilon = epsilon
        self.b = b

    @property
    def fbeta(self):
        return (1+0.925*self.Beta**-1.5)**(2/3.) / (1+self.Beta**-1)**2

    @property
    def scrit(self):
        return np.log(0.067 * self.theta**-2 * self.alpha_vir * self.Mach**2 *
                      self.fbeta)
                    

    def SFRff(self):
        return (self.epsilon / self.phi_t / 2. *
                (0.5 * self.scrit) *
                (1 + scipy.special.erf((self.sigma_s**2 -
                                        2*self.scrit) /
                                       (8*self.sigma_s**2)**0.5)))

class HC2011(SFLaw):
    """
    Hennebelle & Chabrier 2011 law as parametrized by Federrath & Klessen 2012
    """
    name = 'HC2011'

    def __init__(self, y_cut=1.3, alpha_vir=1, Mach=10, Beta=np.inf,
                 epsilon=0.3, phi_t=1./0.24, b=0.4):
        self.y_cut = y_cut
        self.alpha_vir = alpha_vir
        self.Mach = Mach
        self.Beta = Beta
        self.phi_t = phi_t
        self.epsilon = epsilon
        self.b = b

    @property
    def fbeta(self):
        return (1+0.925*self.Beta**-1.5)**(2/3.) / (1+self.Beta**-1)**2

    @property
    def rhocrit_thermal(self):
        return ((np.pi**2/5.) * self.y_cut**-2 * self.alpha_vir * self.Mach**-2 *
                (1+self.Beta**-1))

    @property
    def rhocrit_turbulent(self):
        return np.pi**2 / 15. * self.y_cut**-1 * self.alpha_vir

    @property
    def scrit(self):
        return np.log(self.rhocrit_turbulent + self.rhocrit_thermal)

    def SFRff(self):
        raise NotImplementedError

class HC2011_multiff(HC2011):
    """
    Hennebelle & Chabrier 2011 law as parametrized by Federrath & Klessen 2012
    """
    name = 'HC2011_multiff'

    @property
    def scrit(self):
        return np.log(self.rhocrit_thermal)

class Burkhart2018(PN2011):
    """

    """
    name = 'Burkhart2018'

    def __init__(self, alpha=2, Mach=10, epsilon=0.2, b=0.4,
                 phi_t=1/3., alpha_vir=1., Beta=20,
                ):
        self.alpha = alpha
        self.b = b
        self.epsilon = epsilon
        self.Mach = Mach

        self.alpha_vir = alpha_vir
        self.phi_t = phi_t
        self.Beta = Beta

        # eqn19
        self.s_t = 0.5 * (2*np.abs(self.alpha) - 1) * self.sigma_s**2
        self.C = (np.exp(0.5 * (self.alpha-1)
                         * self.alpha * self.sigma_s**2)
                  / (self.sigma_s * np.sqrt(2*np.pi)))
        self.N = 2 * (1 + np.erf((2*np.log(self.s_t) +
                                  self.sigma_s**2)/(2**1.5*self.sigma_s))
                      + (2*self.C * self.s_t**self.alpha)/self.alpha)**-1


    def SFRff(self):
        return (np.exp(0.5*self.scrit) * self.N * self.epsilon
                * (0.5 * erf((self.sigma_s**2 - 2*self.scrit)/np.sqrt(8*self.sigma_s**2))
                   - 0.5 * erf((self.sigma_s**2 - 2*self.s_t)/np.sqrt(8*self.sigma_s**2))
                   + self.C * np.exp(self.s_t * (1-self.alpha)) / (self.alpha-1))
               )

    @property
    def scrit(self):
        #return np.log(0.067 * self.theta**-2 * self.alpha_vir * self.Mach**2 *
        #              self.fbeta)
        return np.log(np.pi**2/15 * self.phi_t**2 * self.alpha_vir * self.Mach**2)
