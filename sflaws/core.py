import numpy as np
import scipy.special

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
    name = 'KM2005'

    def __init__(self, phi_x=1.12, alpha_vir=1, Mach=10, Beta=np.inf,
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
