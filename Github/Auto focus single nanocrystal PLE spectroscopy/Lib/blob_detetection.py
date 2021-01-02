# 	Original code by Yurii Morozov


from math import sqrt
from skimage import data, io
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from datetime import datetime as dt


def blob_shift(im1_in, im2_in, scan_foldername, oversample = 5, lscale=1):
    if np.shape(im1_in)!=np.shape(im2_in):
        raise ValueError('Input images must have the same sizes!')

    nx, ny = np.shape(im1_in)
    x = np.arange(nx)
    y = np.arange(ny)

    xnew = np.linspace(0, nx, nx*oversample)
    ynew = np.linspace(0, ny, ny*oversample)

    f1 = interpolate.interp2d(x, y, im1_in, kind='cubic')
    f2 = interpolate.interp2d(x, y, im2_in, kind='cubic')

    im1 = f1(xnew, ynew)
    im2 = f2(xnew, ynew)

    im1 = im1 - np.min(im1)
    im1 = 100.*im1/np.max(im1)

    im2 = im2 - np.min(im2)
    im2 = 100.*im2/np.max(im2)

    level = np.mean(im1)*lscale

    blobs_log_im1 = blob_log(im1, max_sigma=10.*oversample/5, min_sigma=2.*oversample/5, num_sigma=10, threshold=level)
    blobs_log_im2 = blob_log(im2, max_sigma=10.*oversample/5, min_sigma=2.*oversample/5, num_sigma=10, threshold=level)

    # Compute radii in the 3rd column.
    blobs_log_im1[:, 2] = blobs_log_im1[:, 2] * sqrt(2)
    blobs_log_im2[:, 2] = blobs_log_im2[:, 2] * sqrt(2)

    amps_im1 = im1[blobs_log_im1[:,0].astype(int), blobs_log_im1[:,1].astype(int)]
    amps_im2 = im2[blobs_log_im2[:,0].astype(int), blobs_log_im2[:,1].astype(int)]
    
    blobs_log_im1 = np.append(blobs_log_im1, np.atleast_2d(amps_im1).T, axis=1)
    blobs_log_im2 = np.append(blobs_log_im2, np.atleast_2d(amps_im2).T, axis=1)
    
    blobs_log_im1 = blobs_log_im1[blobs_log_im1[:,-1].argsort()[::-1]]
    blobs_log_im2 = blobs_log_im2[blobs_log_im2[:,-1].argsort()[::-1]]

    
    x1, y1 = blobs_log_im1[0,0]/(1.*oversample), blobs_log_im1[0,1]/(1.*oversample)
    x2, y2 = blobs_log_im2[0,0]/(1.*oversample), blobs_log_im2[0,1]/(1.*oversample)

    dx = x2-x1
    dy = y2-y1

    
    fig, axes = plt.subplots(1, 1, figsize=(3, 3), sharex=True, sharey=True)
    axes.imshow(im2_in, interpolation='nearest')

    for blob in blobs_log_im2:
        y, x, r, a = blob
        c = plt.Circle((x/(1.*oversample), y/(1.*oversample)), r/(1.*oversample), color='red', linewidth=2, fill=False)
        c2 = plt.Circle((x/(1.*oversample), y/(1.*oversample)), r/(1.*oversample*5), color='red', linewidth=2, fill=True)
        axes.add_patch(c)
        axes.add_patch(c2)
    axes.set_axis_off()

    plt.tight_layout()
    f_name_prefix = "_"+str( dt.now().hour)+str( dt.now().minute)+str(dt.now().second)
    plt.savefig(scan_foldername+f_name_prefix+".png", dpi = 400)
    plt.close()
    return dx, dy, x2, y2
