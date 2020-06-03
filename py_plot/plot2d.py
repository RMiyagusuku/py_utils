import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import itertools
import numpy as np
import re

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
    #fig = kwargs.get('fig',None)
    ax  = kwargs.get('ax',None)
    # figure and ax
    if ax is None:
        fig = plt.figure()
        fig.set_size_inches(8,6,forward=True)
        rect = [.15,.15,.75,.75] # setting the axis limits in [left, bottom, width, height]
        ax   = fig.add_axes(rect)# the carthesian axis:
    # arguments for plots
    #testdata  = kwargs.get('testdata',None)
    #TestPoint = kwargs.get('TestPoint',None)
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
        cbar  = plt.colorbar(C, **kwargs_cbar)
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
    #plt.show()
    return ax

class pgm_map():
    """
    Class to read and plot maps form .pgm/.yaml files
    """
    def __init__(self,file_name='/home/renato/catkin_ws/src/tests/maps/b2map'):
        fmap  = file_name+'.pgm'
        fyaml = file_name+'.yaml'
        
        self.map_yaml = self.read_yaml(fyaml)
        self.image = self.read_pgm(fmap, byteorder='<')
        self.span_y = len(self.image)*self.map_yaml['resolution']
        self.span_x = len(self.image[0])*self.map_yaml['resolution']
        self.extent = [self.map_yaml['origin'][0],self.map_yaml['origin'][0]+self.span_x,
                       self.map_yaml['origin'][1],self.span_y+self.map_yaml['origin'][1]]

    def read_yaml(self,file_yaml):
        """
        read yaml file with map information
        """
        map_yaml = dict()
        with open(file_yaml,'rb') as f:
            lines = f.readlines()
            for line in lines:   
                try:
                    key = line.split(':')[0]
                    val = line.split(':')[1][:-1]
                    if key == 'origin':
                        val = val.split('[')[1]
                        val = val.split(']')[0]
                        val = np.array(val.split(','),dtype=float)
                    else:
                        try:
                            val = float(val)
                        except:
                            val = val.replace(' ','') #eliminate white spaces
                    map_yaml[key]=val
                except:
                    pass
        return map_yaml

    def read_pgm(self, file_pgm, byteorder='<'):
        """
        Return image data from a raw PGM file as numpy array.
        Format specification: http://netpbm.sourceforge.net/doc/pgm.html
        """
        with open(file_pgm, 'rb') as f:
            buffer = f.read()
        try:
            header, width, height, maxval = re.search(
                b"(^P5\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
        except AttributeError:
            raise ValueError("Not a raw PGM file: '%s'" % file_pgm)
        image = np.frombuffer(buffer, dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                                count=int(width)*int(height), offset=len(header)
                                ).reshape((int(height), int(width)))

        #make unexplored areas gray clear
        #image.flags['WRITEABLE'] = True
        #image[image == 205] = 220
        #image.flags['WRITEABLE'] = False
        return np.flip(image,axis=0)

    def plot(self,f=None,ax=None):
        if f is None:
            f, ax = plt.subplots(figsize=(16,12))
        ax.imshow(self.image,plt.cm.gray,origin='lower',extent=self.extent)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        return f,ax       
