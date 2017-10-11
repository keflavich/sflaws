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

def lognormal(x, sigma):
    s0 = -0.5 * sigma**2
    return np.exp(-((np.log(x)-s0)**2/(2*sigma**2)))

if __name__ == "__main__":

    meandens = 1e4
    # xx is the density relative to the mean density, i.e., it is the overdensity
    xx = np.logspace(np.log10(meandens)-10, np.log10(meandens)+10, 10000) / meandens

    for model in (KM2005, PN2011):
        pl.clf()

        pl.suptitle(model.name)

        for pp in panels:

            ax = pl.subplot2grid((3,3), np.array(pp)-1)

            thismodel = model(Mach=panels[pp]['mach'], Beta=panels[pp]['beta'])

            logmeandens = np.exp(-0.5 * thismodel.sigma_s**2)

            ax.loglog(xx, lognormal(xx, sigma=thismodel.sigma_s))

            thresh = np.exp(thismodel.scrit)
            print(pp, panels[pp], thismodel.sigma_s, thresh, np.log10(thresh), thismodel.scrit)

            ax.fill_between(xx[xx>thresh], xx[xx>thresh]*1e-100,
                            lognormal(xx[xx>thresh], sigma=thismodel.sigma_s),
                            alpha=0.5)

            ax.vlines(logmeandens, 1e-10, 2, color='k', linestyle='--', linewidth=3,
                      alpha=0.5)

            #ax.vlines(meandens, 1e-10, 2, color='k', linestyle='--', linewidth=1,
            #          alpha=0.5)

            ax.set_title("$\mathcal{{M}}={mach} \\beta={beta}$".format(**panels[pp]))
            ax.set_xlabel("Overdensity")

            ax.axis([logmeandens/1e6, logmeandens*1e6,1e-6,1.5])
            ax.xaxis.set_major_locator(pl.matplotlib.ticker.LogLocator(base=100))

        pl.tight_layout()
        pl.draw()
        pl.show()
        pl.tight_layout()

        pl.savefig("{0}_machbetaplots.png".format(model.name), bbox_inches='tight')

        print()
