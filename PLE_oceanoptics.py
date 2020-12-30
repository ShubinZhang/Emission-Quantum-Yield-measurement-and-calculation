'''
Python script to measure the emission quantum yields with differrnt excitation wavelength.

Original code by
Copyright (C) 2017  Shubin Zhang

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation
'''
import os
import sys
import time
from time import sleep
import numpy as np
from datetime import datetime as dt
from fianium import fianium_aotf
from scipy import optimize,interpolate
import pyqtgraph as pg
import ocean_optics_spectrum as oos
import matplotlib.pyplot as plt
from config import setting_measurement


class QY():
    def __init__(self):
        self.cur_time = dt.now()
        ##initialize fianium
        self.fianium = fianium_aotf()
        self.fianium._open()
        self.fianium.enable()
        ##initialize ocean_optics
        self.op = oos.oceanoptics()


        

    def make_folder(self):
        cur_time = dt.now()
        self.f_name_prefix = str(cur_time.year)+str(cur_time.month)+str(cur_time.day)+"_"+"%1.2d"%cur_time.hour+"%1.2d"%cur_time.minute+"_"+"QY"
        self.dir_name = "data\\"+self.f_name_prefix
        if os.path.exists(self.dir_name): #check if folder already exists
            if os.listdir(self.dir_name): #check if folder is empty
                self.dir_name = self.dir_name+"_1"#change folder name if folder is not empty
                os.makedirs(self.dir_name) #create another folder if folder is not empty
            else:
                print "can't make a folder"
        else:
			os.makedirs(self.dir_name)
        print "making folder"

    def save_data(self, x,y, sufix="", figure = True, sortdata=False,header=''):
		if sortdata:
			sortind = np.argsort(x)
			x = x[sortind]
			y = y[sortind]
		
		fname = self.dir_name+"\\"+self.f_name_prefix+sufix+".dat"	
		print "------------------------------------------------------"
		print "Data saved to "
		print fname

		np.savetxt(fname, np.column_stack([x,y]),fmt='%1.7e	%1.7e', header = header)
		if figure:
			plt.close()
			plt.figure(figsize = (4.8*1.61, 4.8))
			plt.plot(x,y)
			plt.title(self.f_name_prefix)
			plt.savefig(fname[:-4]+".png", format = 'png', dpi=200)
			plt.close()
    
        #change the range of two 2D lists
    def cutdata(self,x,y,x1,x2):
        indxs = np.where((x>=x1) & (x<x2))
        sub_y = y[indxs]
        sub_x = x[indxs]
        return sub_x, sub_y

    def start_ple_scan(self, **setting):     
        wlen_start = setting["wlen_start"]                     
        wlen_end = setting["wlen_end"]
        wlen_step = setting["wlen_step"]
        time_int = setting["time_int"]                     #us for ocean optics, ms for photodiode (0.5ms to 500ms)
        avg = setting["avg"]
        xmin = setting["xmin"]
        xmax = setting["xmax"]
        mkdir = setting["mkdir"]
        rcurve = setting["rcurve"]
        if mkdir:
            self.make_folder()
        wlenrange = np.arange(wlen_start, wlen_end, wlen_step)  # excitation wavelength range
        Iexc = np.zeros(len(wlenrange))
        self.fianium.set_pwr(0.,aotf = "nir")
        print "Laser is off, measuring background"
        sleep(0.01)        

        #Start to measure background
        print "start measuring backgroud"
        xb,yb = self.op.start_measuring(time_int, avg, xmin, xmax, bgpath = None,  mkdir=mkdir, sub_bg=False,load_bg=False,m_xb = None, m_yb = None,  norm=False)
        print "backgroud measurement finished"
        xrc, yrc = np.loadtxt(rcurve, unpack=True)
        xcc_min,xcc_max = min(xrc), max(xrc)
        ccurve = interpolate.interp1d(xrc,yrc)

		#start measuring
        print "Turn on Laser"	
        self.fianium.set_pwr(100.,aotf = "nir")
        #wait 5s to make sure laser power is stable
        sleep(5)
        for i,wlen in enumerate(wlenrange):
            print "current wavelength is ", wlen
            self.fianium.set_wlen(wlen, aotf="nir")
            sleep(1)
            wlendata, countsdata = self.op.start_measuring(time_int, avg, xmin, xmax, bgpath = None,  mkdir=mkdir, sub_bg=True,load_bg=False,m_xb = xb, m_yb = yb,  norm=False)
            wlendata, countsdata = self.cutdata(wlendata, countsdata, xcc_min, xcc_max)
            countsdata = countsdata/ccurve(wlendata)      
            idx_l = (np.abs(wlendata-wlen+15)).argmin()
            idx_h = (np.abs(wlendata-wlen-15)).argmin()
            Iexc[i] = np.sum(countsdata[idx_l:idx_h])

            self.save_data(wlendata, countsdata, sufix="_%1.1f_nm"%wlen,figure = True,header="Excitation = %1.5f \n"%Iexc[i] +"Exposure time = %1.5f s \n"%time_int)

            sleep(0.001)
            print "****************************************************************"  

        self.fianium.set_pwr(0.,aotf = "nir")


if __name__ == "__main__":

    measurement = QY()
    measurement.start_ple_scan(**setting)
