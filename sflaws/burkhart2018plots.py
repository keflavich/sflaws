import numpy as np
import pylab as pl
from astropy import units as u
from astropy import constants
try:
    from .core import KM2005, PN2011, HC2011, HC2011_multiff, Burkhart2018
except (SystemError,ImportError):
    from core import KM2005, PN2011, HC2011, HC2011_multiff, Burkhart2018


fig4 = pl.figure(4)
fig4.clf()
ax = fig4.gca()

powerlaws = np.linspace(1,3)

sfr_ffs = {(Mach,b): [Burkhart2018(alpha=alpha, Mach=Mach, b=b).SFRff() for alpha in
                      powerlaws]
           for Mach in (4,25)
           for b in (1/3., 0.7)}

rho_0 = 500*u.Da/u.cm**3
tff = (3*np.pi / (32 * constants.G * rho_0))**0.5
for Mass, style in [(100, '-'), (1000, '--'), (1e4, ':')]:
    Mass = u.Quantity(Mass, u.M_sun)
    for Mach,bb in sfr_ffs:
        ax.semilogy(powerlaws,
                    sfr_ffs[(Mach,bb)], # * Mass.to(u.M_sun) / tff.to(u.yr),
                    label="$\mathcal{{M}}={0}$ $b={1}$".format(Mach, bb))

pl.legend(loc='best')
