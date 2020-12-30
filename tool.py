import numpy as np
import scipy.constants as cnst
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import scipy.fftpack
from scipy import stats
from lmfit.models import GaussianModel,ExponentialModel
import os
from scipy.optimize import fsolve
import re 
import sys
#from numpy import NaN, Inf, arange, isscalar, asarray, array




    
def listfldr(fldrname, ext='dat', sort=True, fullpath=False):
    """
    List files in the folder fldrname with ext extension
    
    if fullpath set to True, output will be a full path to each file,
    otherwise there are will be only file names
    
    """
    
    matchedfiles = [f for f in os.listdir(fldrname) if f.endswith(ext)]
    if sort:
        #matchedfiles = sorted(matchedfiles)
        matchedfiles = sorted_abcdgt(matchedfiles)
        
    if fullpath:
        matchedfiles = [os.path.join(fldrname, f) for f in matchedfiles]
    return matchedfiles


def cutdata(x,y,x1,x2):
	indxs = np.where((x>=x1) & (x<x2))
	sub_y = y[indxs]
	sub_x = x[indxs]
	return sub_x, sub_y

def smooth(x, window_len=20, window='blackman'):
	#print x
	if x.ndim != 1:
		raise ValueError, "smooth only accepts 1 dimension arrays."
	if x.size < window_len:
		raise ValueError, "Input vector needs to be bigger than window size."
    
	if window_len<3:
		return x
    
	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
		raise(ValueError,
           "Window is not one of '{0}', '{1}', '{2}', '{3}', '{4}'".format(
           *('flat', 'hanning', 'hamming', 'bartlett', 'blackman')))
   
	s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
	#print(len(s))
	if window == 'flat': #moving average
		w = np.ones(window_len,'d')
	else:
		w = eval('np.' + window + '(window_len)')
	y = np.convolve(w / w.sum(), s, mode = 'valid')
	return y[int((window_len-1)/2):-int((window_len-1)/2)-1]


def comb(x1,y1,x2,y2, b=1, usemin = False):
    
    f1 = interpolate.interp1d(x1, y1)
    f2 = interpolate.interp1d(x2, y2)
    
    minv = max(x1.min(), x2.min())
    maxv = min(x1.max(), x2.max())
    x1,y1 = cutdata(x1,y1, minv, maxv)
    x2,y2 = cutdata(x2,y2, minv, maxv)
    n1 = len(x1)
    n2 = len(x2)
    if usemin:
        nnew = int(min(n1,n2)*b)
    else:
        nnew = int(max(n1,n2)*b)
    
    x0 = np.linspace(minv, maxv, nnew)
    
    y1new = f1(x0)
    y2new = f2(x0)
    
    return x0, y1new, y2new

def comb2(x1,y1,x2,y2):
    
    f1 = interpolate.interp1d(x1, y1)
    f2 = interpolate.interp1d(x2, y2)
    
    minv = x2.min()
    maxv = x2.max()
    
    minv = max(x1.min(), x2.min())
    maxv = min(x1.max(), x2.max())
    x1,y1 = cutdata(x1,y1, minv, maxv)
    x2,y2 = cutdata(x2,y2, minv, maxv)
    
    n1 = len(x1)
    n2 = len(x2)
    
    
    x0 = x2
    
    y1new = f1(x0)
    y2new = f2(x0)
    
    return x0, y1new, y2new
    
def comb3(x1,y1,x2,y2):
    
    f1 = interpolate.interp1d(x1, y1)
    f2 = interpolate.interp1d(x2, y2)
    
    minv = x2.min()
    maxv = x2.max()
    
    minv = max(x1.min(), x2.min())
    maxv = min(x1.max(), x2.max())
    x1,y1 = cutdata(x1,y1, minv, maxv)
    x2,y2 = cutdata(x2,y2, minv, maxv)
    
    n1 = len(x1)
    n2 = len(x2)
    
    
    x0 = x2
    
    y1new = f1(x0)
    y2new = f2(x0)
    
    return x1,x2, y1new, y2new

    
def setxaxis(ax, majorloc, minorloc,formater):
	ax.xaxis.set_major_locator(MultipleLocator(majorloc))
	ax.xaxis.set_minor_locator(MultipleLocator(minorloc))
	ax.xaxis.set_major_formatter(FormatStrFormatter(formater))
    
def setyaxis(ax, majorloc, minorloc,formater):
	ax.yaxis.set_major_locator(MultipleLocator(majorloc))
	ax.yaxis.set_minor_locator(MultipleLocator(minorloc))
	ax.yaxis.set_major_formatter(FormatStrFormatter(formater))

def inverseax(ax1, major=100, minor=50):
    axev = ax1.twiny()
    
    ev_tick_locations = ev(np.arange(1.,4.,major))
    ev_tick_locations_minor = ev(np.arange(1.,4.,minor))
    
    axev.set_xticks(ev_tick_locations)
    axev.set_xticks(ev_tick_locations_minor, minor=True)
    axev.set_xticklabels(["%1.1f" % z for z in ev(ev_tick_locations)])	
    axev.set_xlim(ax1.get_xlim())
    return axev

def evaxis(ax1, major=0.2, minor=0.1):
    axev = ax1.twiny()
    
    ev_tick_locations = ev(np.arange(1.,4.,major))
    ev_tick_locations_minor = ev(np.arange(1.,4.,minor))
    
    axev.set_xticks(ev_tick_locations)
    axev.set_xticks(ev_tick_locations_minor, minor=True)
    axev.set_xticklabels(["%1.1f" % z for z in ev(ev_tick_locations)])	
    axev.set_xlim(ax1.get_xlim())
    return axev

def nmaxis(ax1, major=100., minor=50.):
    axnm = ax1.twiny()
    minx_ev, maxx_ev = ax1.get_xlim()
    minx, maxx = ev(maxx_ev), ev(minx_ev)
    
    nm_tick_locations = ev(np.arange(300.,1200.,major))
    #print nm_tick_locations
    nm_tick_locations_minor = ev(np.arange(300.,1200.,minor))
    
    axnm.set_xticks(nm_tick_locations)
    axnm.set_xticks(nm_tick_locations_minor, minor=True)
    axnm.set_xticklabels(["%1.0f" % z for z in ev(nm_tick_locations)])
    axnm.set_xlim(ax1.get_xlim())
    return axnm
    
     
def rffthz(x,y, skip = 0):
	N = len(x)
	srate = abs(1./(x[1]-x[0]))
	xf = np.linspace(0.0, srate*1.0/(2.0), N/2)
	#xf = xf[1:]
	
	yf = scipy.fftpack.fft(y)
	yf = 2.0/N*np.abs(yf[:N/2])	
	
	return xf[skip:], yf[skip:]


def hex_to_RGB(hex):
  ''' "#FFFFFF" -> [255,255,255] '''
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]


def RGB_to_hex(RGB):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  RGB = [int(x) for x in RGB]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])
def color_dict(gradient):
    return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
      "r":[RGB[0] for RGB in gradient],
      "g":[RGB[1] for RGB in gradient],
      "b":[RGB[2] for RGB in gradient]}


def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)
    # Initilize a list of the output colors with the starting color
    RGB_list = [s]
    # Calcuate a color at each evenly spaced value of t from 1 to n
    for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
        curr_vector = [
        int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
          for j in range(3)
        ]
        # Add it to our list of output colors
        RGB_list.append(curr_vector)

    return color_dict(RGB_list)

def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = np.arange(len(v))
    
    v = np.asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN
    
    lookformax = True
    
    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)


def tcspcprep(y, timeres=0.008, window=11., t1=5., t2 = 25., full = False, peakamp=10):
    x = np.arange(len(y))*timeres
    x,y = cutdata(x,y, t1, t2)
    if full:
        return x,y
    
    ysm = smooth(y, window_len=40)
    amax,bmin = peakdet(ysm, peakamp, x = x)

    xmaxpos = []
    ymaxpos = []
    for pair in amax:
        xmaxpos.append(pair[0])
        ymaxpos.append(pair[1])
    xmaxpos = np.asarray(xmaxpos)
    ymaxpos = np.asarray(ymaxpos)
    
    #print xmaxpos, ymaxpos
    
    xc,yc = cutdata(x,y, xmaxpos[0], xmaxpos[0]+window)
    
    return xc, yc, xmaxpos, ymaxpos

def doublexp(amp1, decay1, amp2, decay2):
    """
    
    """
    exp1 = ExponentialModel(prefix = 'exp1_')
    exp2 = ExponentialModel(prefix = 'exp2_')
    
    mod = exp1 + exp2
    
    pars = mod.make_params()
    pars.update( mod.make_params())

    pars['exp1_decay'].set(decay1,  min = decay1/5, max = decay1*10.)
    pars['exp1_amplitude'].set(amp1, min = amp1/100000., max = amp1*100000.)
   
    pars['exp2_decay'].set(decay2,  min = decay2/5, max = decay2*10.)
    pars['exp2_amplitude'].set(amp2, min = amp2/100000., max = amp2*100000.)
   
    return mod, pars

def sorted_abcdgt( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
