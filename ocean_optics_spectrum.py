'''
Python script to measure spectrum using ocean optics usb2000. 

Original code by
Copyright (C) 2017  Shubin Zhang

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation
'''
import seabreeze.spectrometers as sb
import time
import os
from datetime import datetime as dt
import numpy as np 
import pyqtgraph as pg
import matplotlib.pyplot as plt
from scipy import interpolate

class oceanoptics():
    def __init__(self):
        self.devices = sb.list_devices()
        print self.devices
        self.spec = sb.Spectrometer(self.devices[0])
       


    def savedata(self,x,y, path2file):
        cur_time = dt.now()
        path2file = path2file + str(cur_time.year)+str(cur_time.month)+str(cur_time.day)+"_"+"%1.2d"%cur_time.hour+"%1.2d"%cur_time.minute
        print "data saved at",path2file
        np.savetxt(path2file+".dat", np.column_stack([x,y]), fmt='%1.10f   %1.10f')
        plt.plot(x,y,linewidth=0.5)
        plt.savefig(path2file+".png",dpi=400)
        plt.close()

	#create folder to save data
    def make_folder(self):
        cur_time = dt.now()
        f_name_prefix = str(cur_time.year)+str(cur_time.month)+str(cur_time.day)+"_"+"%1.2d"%cur_time.hour+"%1.2d"%cur_time.minute+"_Ocean_optics"
        
        dir_name = "data\\"+f_name_prefix+"\\"
        if os.path.exists(dir_name): #check if folder already exists
            if os.listdir(dir_name): #check if folder is empty
                dir_name = dir_name+"_1"#change folder name if folder is not empty
                os.makedirs(dir_name) #create another folder if folder is not empty
        else:
            os.makedirs(dir_name)   
        return dir_name

    #change the range of two 2D lists
    def cutdata(self,x,y,x1,x2):
        indxs = np.where((x>=x1) & (x<x2))
        sub_y = y[indxs]
        sub_x = x[indxs]
        return sub_x, sub_y
    
    def comb2(self,x1,y1,x2,y2):
        f1 = interpolate.interp1d(x1, y1)
        f2 = interpolate.interp1d(x2, y2)
        minv = max(x1.min(), x2.min())
        maxv = min(x1.max(), x2.max())
        x1,y1 = self.cutdata(x1,y1, minv, maxv)
        x2,y2 = self.cutdata(x2,y2, minv, maxv)
        x0 = x2
        y1new = f1(x0)
        y2new = f2(x0)
        return x0, y1new, y2new
    
    
    def load_bg(self,bgpath):
        #BG/20181017_1426_bg.dat
        xb, yb = np.loadtxt(bgpath, unpack = True) 
        return xb, yb   
 
    def start_measuring(self, time_int, avg, xmin, xmax, bgpath = None, mkdir=False, sub_bg=False, load_bg=False,m_xb=None,m_yb=None, norm=False):
        """
        Taking spectra
        "bgpath": relative path of spectrum background file
        "mkdir": if True, create folder and save data
        "sub_bg": if True, subtract background spectrum from recently measured spectrum
        """
        self.spec.integration_time_micros(time_int)
        if sub_bg:
            if load_bg:
                try: 
                    xb, yb = self.load_bg(bgpath)
                except:
                    print "no background file found"

            if not load_bg:
                xb, yb = m_xb, m_yb

        for i in range(avg):
            x,y = self.spec.spectrum()
            y=y[1:]
            x=x[1:]

            if sub_bg:
                x,yb_new,y = self.comb2(xb,yb,x,y)
                y -= yb_new

            x, y = self.cutdata(x,y,xmin,xmax)
            if i == 0:
                avg_y = y
            else:
                avg_y = (y+avg_y*(i-1))/i
            if norm:
                avg_y = avg_y/max(avg_y)


        if mkdir:
            fldrname = self.make_folder()
            self.savedata(x,avg_y,fldrname)
        return x,avg_y


if __name__ == "__main__":
    op = oceanoptics()
    bgpath = "BG.dat"
    time_int = 200000  #micro second 
    avg = 50
    xmin = 450
    xmax = 690
    load_bg = False
    mkdir = True
    if not load_bg:

        #measure bg
        m_xb = np.zeros(2048)
        m_yb = np.zeros(2048)
        x,y = op.start_measuring(time_int, avg, xmin, xmax, bgpath,  mkdir=mkdir, sub_bg=False, load_bg=load_bg,m_xb=m_xb,m_yb=m_yb,  norm=False)
    else:
        x,y = op.start_measuring(time_int, avg, xmin, xmax, bgpath,  mkdir=mkdir, sub_bg=True, load_bg=load_bg,m_xb=None,m_yb=None, norm=False)
    print sum(y)
    #bg can be loaded or measured right now.

    
        

