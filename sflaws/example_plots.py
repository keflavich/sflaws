import numpy as np
import pylab as pl

try:
    from .core import KM2005, PN2011
except (SystemError,ImportError):
    from core import KM2005, PN2011

# make 9 panel plots showing the resulting sigma_s and critical density...

panels = {(1,1): {'mach': 5, 'beta': np.inf},
          (1,2): {'mach': 10, 'beta': np.inf},
          (1,3): {'mach': 20, 'beta': np.inf},
          (2,1): {'mach': 5, 'beta': 2},
          (2,2): {'mach': 10, 'beta': 2},
          (2,3): {'mach': 20, 'beta': 2},
          (3,1): {'mach': 5, 'beta': 1.25},
          (3,2): {'mach': 10, 'beta': 1.25},
          (3,3): {'mach': 20, 'beta': 1.25},
         }

def lognormal(x, sigma, mu):
    return np.exp(-(np.log(x/mu)**2/(2*sigma**2)))

if __name__ == "__main__":

    xx = np.logspace(-4, 10, 10000)
    meandens = 1e4

    for model in (KM2005, PN2011):
        pl.clf()

        pl.suptitle(model.name)

        for pp in panels:

            ax = pl.subplot2grid((3,3), np.array(pp)-1)

            thismodel = model(Mach=panels[pp]['mach'], Beta=panels[pp]['beta'])

            ax.loglog(xx, lognormal(xx, mu=meandens, sigma=thismodel.sigma_s))

            thresh = meandens + 10**thismodel.scrit
            print(pp, panels[pp], thismodel.sigma_s, thresh, np.log10(thresh), thismodel.scrit)

            ax.fill_between(xx[xx>thresh], xx[xx>thresh]*1e-100,
                            lognormal(xx[xx>thresh], mu=meandens, sigma=thismodel.sigma_s),
                            alpha=0.5)

            ax.vlines(meandens, 1e-10, 2, color='k', linestyle='--', linewidth=3,
                      alpha=0.5)

            ax.set_title("$\mathcal{{M}}={mach} \\beta={beta}$".format(**panels[pp]))

            ax.axis([1e-4,1e10,1e-6,1.5])

        pl.tight_layout()
        pl.draw()
        pl.show()
        pl.tight_layout()

        pl.savefig("{0}_machbetaplots.png".format(model.name), bbox_inches='tight')

        print()
