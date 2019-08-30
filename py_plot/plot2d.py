import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import itertools
import numpy as np

################################### preferred plot defaults ###################################
mpl.rcParams['axes.unicode_minus']=False
mpl.rc('xtick', labelsize=30)
mpl.rc('ytick', labelsize=30)
mpl.rcParams.update({'font.size': 30})


sns.set_style("white")
sns.set_style("ticks")

Greens = mpl.colors.ListedColormap(sns.color_palette("Greens",50))
Blues  = mpl.colors.ListedColormap(sns.color_palette("Blues",50))
GnBu_d = mpl.colors.ListedColormap(sns.color_palette("GnBu_d",50))
GnBu   = mpl.colors.ListedColormap(sns.color_palette("GnBu",50))

sns.set_palette("Blues_d")
palette = itertools.cycle(sns.color_palette())
bl_dark = next(palette)
bl_dark = next(palette)

sns.set_palette("Greens_d")
palette = itertools.cycle(sns.color_palette())
gr_dark = next(palette)
gr_dark = next(palette)

sns.set_palette("Paired")
palette = itertools.cycle(sns.color_palette())
bl_light = next(palette)
bl       = next(palette)
gr_light = next(palette)
gr       = next(palette)

pr = "#9b59b6"
rd = "#e74c3c"

sns.set_palette("colorblind")
palette = itertools.cycle(sns.color_palette())
##############################################################################################

def plot_contour(Xtest,pXtest,**kwargs):
    fig = kwargs.get('fig',None)
    ax  = kwargs.get('ax',None)
    # figure and ax
    if fig is None:
        fig = plt.figure()
        fig.set_size_inches(8,6,forward=True)
        rect = [.15,.15,.75,.75] # setting the axis limits in [left, bottom, width, height]
        ax   = fig.add_axes(rect)# the carthesian axis:
    # arguments for plots
    testdata  = kwargs.get('testdata',None)
    TestPoint = kwargs.get('TestPoint',None)
    colorbar  = kwargs.get('colorbar',True)
    default_shape = int(Xtest.shape[0]**.5)
    shape     = kwargs.get('shape',(default_shape,default_shape))
    axis_lim  = kwargs.get('axis_lim',(np.min(Xtest[:,0]),np.max(Xtest[:,0]),np.min(Xtest[:,1]),np.max(Xtest[:,1])))
    fontsize  = kwargs.get('fontsize',26)
    # kwargs passthrough
    kwargs_contourf = kwargs.get('kwargs_contourf',dict())
    kwargs_cbar = kwargs.get('kwargs_cbar',dict())
    # defaults
    kwargs_contourf.setdefault('cmap','Blues')
    kwargs_cbar.setdefault('label','p(x|z)\n ')
    kwargs_cbar.setdefault('fontsize',fontsize)
    kwargs_cbar.setdefault('shrink',0.65)
    kwargs_cbar.setdefault('extend','both')
    kwargs_cbar.setdefault('cmap',kwargs_contourf['cmap'])
    kwargs_cbar.setdefault('ax',ax)
    # contour plot
    C = ax.contourf(np.reshape(Xtest[:,0],shape),np.reshape(Xtest[:,1],shape),np.reshape(pXtest,shape),
                        kwargs_contourf.pop('v',25),**kwargs_contourf)
    for c in C.collections: c.set_edgecolor("face")
    # colorbar
    if colorbar:
        cbar_label = kwargs_cbar.pop('label')
        cbar_fontsize = kwargs_cbar.pop('fontsize')
        cbar  = fig.colorbar(C, **kwargs_cbar)
        tick_locator = mpl.ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        cbar.solids.set_edgecolor("face")
        cbar.ax.set_ylabel(cbar_label,fontsize=cbar_fontsize)
    else:
        cbar = None
    # labels
    ax.axis(axis_lim)
    ax.set_aspect('equal')
    ax.set_xlabel('x[m]',fontsize=26)
    ax.set_ylabel('y[m]',fontsize=26)
    plt.show()
    return fig, ax
