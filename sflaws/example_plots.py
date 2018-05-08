import numpy as np
import pylab as pl

try:
    from .core import KM2005, PN2011, HC2011, HC2011_multiff
except (SystemError,ImportError):
    from core import KM2005, PN2011, HC2011, HC2011_multiff

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
    """
    Return the lognormal evaluated at x, where x is the *overdensity*.  Sigma
    is the logarithmic width.
    """
    s0 = -0.5 * sigma**2
    return np.exp(-((np.log(x)-s0)**2/(2*sigma**2)))

# set up some colors etc...
KM2005.color = 'r'
PN2011.color = 'b'
HC2011.color = 'k'
HC2011_multiff.color = 'g'

if __name__ == "__main__":

    from cycler import cycler

    pl.rcParams['axes.prop_cycle'] = cycler('color', ("#"+x for x in ('338ADD', '9A44B6', 'A60628', '467821', 'CF4457', '188487', 'E24A33')))

    meandens = 1e4
    # xx is the density relative to the mean density, i.e., it is the overdensity
    xx = np.logspace(np.log10(meandens)-10, np.log10(meandens)+10, 10000) / meandens
    ss = np.log(xx) # this is the Gaussian random variable

    fig1 = pl.figure(1)
    fig2 = pl.figure(2)
    fig2.clf()

    for model in (KM2005, PN2011, HC2011, HC2011_multiff):
        fig1.clf()

        fig1.suptitle(model.name)

        print(model.name)

        for pp in panels:

            fig1 = pl.figure(1) # reactivate figure 1
            ax = pl.subplot2grid((3,3), np.array(pp)-1)

            thismodel = model(Mach=panels[pp]['mach'], Beta=panels[pp]['beta'])

            logmeandens = np.exp(-0.5 * thismodel.sigma_s**2)

            L, = ax.loglog(xx, lognormal(xx, sigma=thismodel.sigma_s))

            thresh = np.exp(thismodel.scrit)
            print(pp, panels[pp], thismodel.sigma_s, thresh, np.log10(thresh), thismodel.scrit)

            ax.fill_between(xx[xx>thresh], xx[xx>thresh]*1e-100,
                            lognormal(xx[xx>thresh], sigma=thismodel.sigma_s),
                            color=L.get_color(),
                            alpha=0.5)

            ax.vlines(logmeandens, 1e-10, 2, color='k', linestyle='--', linewidth=3,
                      alpha=0.5)

            #ax.vlines(meandens, 1e-10, 2, color='k', linestyle='--', linewidth=1,
            #          alpha=0.5)


            #print(pp, ax.get_subplotspec().num1, ax.get_subplotspec().num2)

            ax.text(logmeandens/5e5, 5e-2, "$\mathcal{{M}}={mach}$\n$\\beta={beta}$".format(**panels[pp]))
            ax.axis([logmeandens/1e6, logmeandens*1e6,1e-6,1.5])
            ax.xaxis.set_major_locator(pl.matplotlib.ticker.LogLocator(base=100))

            #ax.set_title("$\mathcal{{M}}={mach} \\beta={beta}$".format(**panels[pp]))
            if pp == (3,2):
                ax.set_xlabel("Overdensity")
            if pp == (2,2):
                ax.set_ylabel("$P(s)$")
            if pp[0] in (1,2):
                ax.set_xticklabels([])
            if pp[1] in (2,3):
                ax.set_yticklabels([])

            ax.axis([logmeandens/1e6, logmeandens*1e6,1e-6,1.5])

        fig1.tight_layout()
        fig1.subplots_adjust(hspace=0, wspace=0)
        pl.draw()
        pl.show()
        fig1.tight_layout()
        fig1.subplots_adjust(hspace=0, wspace=0)

        fig1.savefig("{0}_machbetaplots.png".format(model.name), bbox_inches='tight')

        ax = fig2.gca()

        Mach = np.linspace(1, 100)
        Beta = (1.25, 2, np.inf)

        #for bb,linestyle in zip(Beta, (':','--','-')):
        # only plot no-b-field instance
        for bb,linestyle in [(np.inf, '-')]:
            thismodel = model(Mach=Mach, Beta=bb)
            ax.plot(Mach, thismodel.scrit, color=thismodel.color,
                    linestyle=linestyle,
                    #label="{0} $\\beta={1}$".format(thismodel.name, bb))
                    label="{0}".format(thismodel.name,))

            if thismodel.color == 'm' and linestyle == ':':
                print("doot")

        print()

    fig2 = pl.figure(2) # activate figure 2
    ax = fig2.gca()
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("Mach number")
    ax.set_ylabel("Critical Overdensity $s_{crit}$")
    fig2.savefig("critical_density_vs_Mach.png", bbox_inches='tight')
