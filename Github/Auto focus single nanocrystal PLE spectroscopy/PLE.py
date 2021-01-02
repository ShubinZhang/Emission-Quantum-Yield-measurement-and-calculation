import os
import sys
import time
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt
from Lib import blob_detetection as blob
from Lib import mcl_piezo_lib
from Lib import acton
from Lib import fianium
from Lib import andor_camera
from Lib import phaser
from Lib import transrefl_lib
from PyQt4 import QtCore, QtGui
from scipy import optimize
import pyqtgraph as pg
from config import setting_dict




class Auto_focus_PLE():
    def __init__(self,emission_area,background_area,response_file):
        self.app = QtGui.QApplication(sys.argv)
        self.camera = andor_camera.AndorCamera()
        self.mclp = mcl_piezo_lib.madpiezo()
        self.cur_time = dt.now()
        self.fianium = fianium.fianium_aotf()
        self.sia = transrefl_lib.sia()
        self.obis = phaser.obis(1)

        self.camera.SetCoolerMode(1)
        self.camera.CoolerON()
        self.camera.SetTemperature(-85)
        self.fianium.enable()
        self.abs_power_wlen, self.abs_power_r = np.loadtxt(response_file, unpack =True)
        
        self.counts_image_bot = emission_area[0]
        self.counts_image_top = emission_area[1]
        self.counts_image_left = emission_area[2]
        self.counts_image_right = emission_area[3]
        self.counts_bckgnd_bot = background_area[0]
        self.counts_bckgnd_top = background_area[1]
        self.counts_bckgnd_left = background_area[2]
        self.counts_bckgnd_right = background_area[3]
        
        self.create_folder()
        
    def camera_setting_default(self):

        self.exp_time = 0.025  #default exposure
        self.numb_accum = 1
        self.kinetic_series_length = 1
        self.numb_prescans = 0
        self.em_gain = 200
        self.aqu_mode = 1
        self.triggering = 0
        self.readmode = 4
        self.VSSpeed = 1 # 0.3 0.5 0.9 1.7 3.3
        self.VSAmplitude = 0 #0(Normal), +1, +2, +3, +4

        self.adc_chan = 0 
        self.HSSspeed = 0
        self.preamp_gain = 0 # 1.0 2.4 4.9

        self.image_left = 1
        self.image_right = 512
        self.image_bot = 1
        self.image_top = 512
        self.bin_horizontal = 1
        self.bin_vertical = 1
        
        self.camera.SetShutter(1, 1, 0, 0)
        print "Ser number = ", self.camera.GetCameraSerialNumber()
        print self.camera.GetTemperature()		
        

        self.camera.SetPreAmpGain(self.preamp_gain)
        self.camera.SetVSSpeed(self.VSSpeed)
        self.camera.SetADChannel(self.adc_chan)
         
        self.camera.SetEMCCDGain(self.em_gain)
        self.camera.SetAcquisitionMode(self.aqu_mode) # 1 - single scan
        self.camera.SetNumberAccumulations(self.numb_accum)
        self.camera.SetReadMode(self.readmode) # 0 - FVB, 3 - single track
        
        self.camera.SetImage(self.bin_horizontal,self.bin_vertical,self.image_left,self.image_right,self.image_bot,self.image_top)
        self.camera.SetExposureTime(self.exp_time)
        self.camera.SetTriggerMode(self.triggering)
        
        print "camera._width = ", self.camera._width
        print "camera._height = ", self.camera._height
        self.camera.GetAcquisitionTimings()
        slptm = float(self.camera._kinetic)
        print "sleeptime = ", slptm
        sleep(slptm+0.5)

        


    def camera_setting_change(self,name,exp_time,image=None):
        if name == "scan":
            self.image_bot = self.image[0]
            self.image_top = self.image[1]
            self.image_left = self.image[2]
            self.image_right = self.image[3]   
            self.camera.SetEMCCDGain = 200
            self.camera.SetImage(self.bin_horizontal,self.bin_vertical,self.image_left,self.image_right,self.image_bot,self.image_top)
            self.camera.SetExposureTime(exp_time)
       
        elif name == "ple":
            self.image_bot = self.image[0]
            self.image_top = self.image[1]
            self.image_left = self.image[2]
            self.image_right = self.image[3]   
            self.camera.SetEMCCDGain = 200
            self.camera.SetImage(self.bin_horizontal,self.bin_vertical,self.image_left,self.image_right,self.image_bot,self.image_top)
            self.camera.SetExposureTime(exp_time)
        
        elif name == "autofocus":
            self.image_bot = image[0]
            self.image_top = image[1]
            self.image_left = image[2]
            self.image_right = image[3]   
            self.camera.SetEMCCDGain = 0 
            self.camera.SetImage(self.bin_horizontal,self.bin_vertical,self.image_left,self.image_right,self.image_bot,self.image_top)
            self.camera.SetExposureTime(exp_time)
        


    def create_folder(self):
        self.dir_name = "data\\"+str(self.cur_time.year)+str(self.cur_time.month)+str(self.cur_time.day)+"_"+str(self.cur_time.hour)+str(self.cur_time.minute)+"_ple"
        if os.path.exists(self.dir_name): #check if folder already exists
            if os.listdir(self.dir_name): #check if folder is empty
                self.dir_name = self.dir_name+"_1"#change folder name if foder is not empty
                os.makedirs(self.dir_name) #create another foder if foder is not empty
        
        else:
            os.makedirs(self.dir_name)
        self.scan_foldername = self.dir_name+"\\scan"
        os.makedirs(self.scan_foldername)
        os.makedirs(self.dir_name+"\\ple")
        
    def scan(self,xstart,ystart,zstart,xend,yend,nx,ny,scan_wlen,scan_laser_power,exp_time):
        self.x_pattern, self.y_pattern = np.meshgrid(np.linspace(xstart, xend, nx), np.linspace(ystart, yend, ny))
        self.scan_shape = np.shape(self.x_pattern)

        self.mclp.goxy(xstart,ystart)
        self.mclp.goz(zstart)
        self.counts_data = np.zeros(self.scan_shape)
        self.camera_setting_change("scan",exp_time,self.image)
        self.fianium.set_wlen(scan_wlen, aotf="vis")
        self.fianium.set_pwr(scan_laser_power,aotf = "vis")
        # counts area in subimage
        self.counts_image_left_rel = abs(self.image_left-self.counts_image_left)
        self.counts_image_right_rel = abs(self.image_left-self.counts_image_right)
        self.counts_image_bot_rel = abs(self.image_bot-self.counts_image_bot)
        self.counts_image_top_rel = abs(self.image_bot-self.counts_image_top)
        # bg area in subimage
        self.counts_bckgnd_left_rel = abs(self.image_left-self.counts_bckgnd_left)
        self.counts_bckgnd_right_rel = abs(self.image_left-self.counts_bckgnd_right)
        self.counts_bckgnd_bot_rel = abs(self.image_bot-self.counts_bckgnd_bot)
        self.counts_bckgnd_top_rel = abs(self.image_bot-self.counts_bckgnd_top)

        for index in np.ndindex(self.scan_shape):
            self.cur_index_position = index
            self.mclp.goxy(self.x_pattern[index],self.y_pattern[index])
            self.camera.StartAcquisition()
            self.camera.WaitForAcquisition()
            self.current_data = self.camera.GetAcquiredData([])
            self.current_data = np.asarray(self.current_data)
            self.current_data = np.resize(self.current_data, (self.image[1] - self.image[0], self.image[3] - self.image[2]))   
            self.counts_data[index] = np.sum(self.current_data[self.counts_image_bot_rel:self.counts_image_top_rel, self.counts_image_left_rel:self.counts_image_right_rel]) - np.sum(self.current_data[self.counts_bckgnd_bot_rel:self.counts_bckgnd_top_rel, self.counts_bckgnd_left_rel:self.counts_bckgnd_right_rel])
            QtCore.QCoreApplication.processEvents()
            print "counts",self.counts_data[index]
        print "self.counts_data",self.counts_data
        print "self.counts_data shape = ", np.shape(self.counts_data)
        self.fianium.set_pwr(0.,aotf = "vis")
        self.f_name_prefix =str(self.cur_time.year)+str(self.cur_time.month)+str(self.cur_time.day)+"_"+str(self.cur_time.hour)+str(self.cur_time.minute)+"_"+str(self.cur_time.second)
        np.savetxt(self.scan_foldername+"\\"+self.f_name_prefix+"_scan.dat", self.counts_data, fmt='%1.7f',delimiter=",", newline=",\n")



    def fin_get_dark_level(self, int_time = 500.):
		self.fianium.set_pwr(0.,aotf = "vis")
		self.fianium.set_pwr(0.,aotf = "nir")
		self.sia.int_time = int_time
		self.sia.configure_devices()
		
		self.sia.start_cycle()
		self.dark_level = self.sia.calc_amplitude(darksignal=False)

		return self.dark_level
    
    
    def get_pl_and_power(self,exp_time,wlen=False):
        self.camera.GetAcquisitionTimings()
        real_exptime = self.camera._exposure
        if ((self.desired_int_time>=(real_exptime*1000/2.)) and self.desired_int_time>self.max_exp_sia):
            self.sia.set_int_time(real_exptime*1000/2.)
        powerlevel1 = self.sia.ref_amp(12.)
        self.camera.StartAcquisition()
        self.camera.WaitForAcquisition()
        self.ple_current_data = self.camera.GetAcquiredData([])
        QtCore.QCoreApplication.processEvents()
        self.ple_current_data = np.asarray(self.ple_current_data)
        self.ple_current_data = np.resize(self.ple_current_data, (self.image[1] - self.image[0], self.image[3] - self.image[2]))
        sleep(0.01)
        self.ple_counts_data = np.sum(self.ple_current_data[self.counts_image_bot_rel:self.counts_image_top_rel, self.counts_image_left_rel:self.counts_image_right_rel]) - np.sum(self.ple_current_data[self.counts_bckgnd_bot_rel:self.counts_bckgnd_top_rel, self.counts_bckgnd_left_rel:self.counts_bckgnd_right_rel])
        powerlevel2 = self.sia.ref_amp(12.)
        powerlevel = 0.5*(powerlevel1+powerlevel2)
        sia_int_time = self.sia.int_time
        if powerlevel < 100*self.dark_level:
            #get new darklevel
            self.dark_level = self.fin_get_dark_level()
        powerlevel = powerlevel - self.dark_level
        pllevel = np.mean(self.ple_counts_data)/real_exptime
        if wlen:
            #get response coefficient
            powerlevel = powerlevel*self.get_response(wlen)
        return powerlevel, pllevel, real_exptime, self.dark_level, sia_int_time
			
    def get_response(self, wlen):
		idx = (np.abs(self.abs_power_wlen-wlen)).argmin()
		return self.abs_power_r[idx]
        
    
    def ple(self,ple_laser_power,exp_time):
        min_exp = exp_time
        min_exp_sia = 0.5 
        self.max_exp_sia = 500.
        wlenrange = np.arange(self.wlen_start, self.wlen_end, self.wlen_step)  
        wlenrange = wlenrange[::-1] #excitation wavelength scan from red to blue scan       
        plesignals_all = np.zeros(len(wlenrange)) 
        initial_int_time = 1.
        self.camera_setting_change("ple",exp_time)
        # counts area in subimage
        self.counts_image_left_rel = abs(self.image_left-self.counts_image_left)
        self.counts_image_right_rel = abs(self.image_left-self.counts_image_right)
        self.counts_image_bot_rel = abs(self.image_bot-self.counts_image_bot)
        self.counts_image_top_rel = abs(self.image_bot-self.counts_image_top)
        # bg area in subimage
        self.counts_bckgnd_left_rel = abs(self.image_left-self.counts_bckgnd_left)
        self.counts_bckgnd_right_rel = abs(self.image_left-self.counts_bckgnd_right)
        self.counts_bckgnd_bot_rel = abs(self.image_bot-self.counts_bckgnd_bot)
        self.counts_bckgnd_top_rel = abs(self.image_bot-self.counts_bckgnd_top)
        

        if self.n_scans_cur == 0:
            line_plot = pg.plot()
            line_plot.showGrid(x=True, y=True, alpha=1.)
            self.powercrv = line_plot.plot( pen=None, symbolPen=None, symbolSize=10, symbolBrush=(255, 0, 0))

            line_plot2 = pg.plot()
            line_plot2.showGrid(x=True, y=True, alpha=1.)
            self.powercrv22 = line_plot2.plot( pen=None, symbolPen=None, symbolSize=10, symbolBrush=(0, 255, 0))
            self.finalcrv = line_plot2.plot(pen = pg.mkPen('g', width=2.0))
        else:
            self.finalcrv.setData(wlenrange,self.plesignals_total)
            QtCore.QCoreApplication.processEvents()
 
        n_scans = 0
        while n_scans < 2:


            pllevels = np.zeros(len(wlenrange))
            powerlevels = np.zeros(len(wlenrange))
            plesignals = np.zeros(len(wlenrange))
            dark_level = self.fin_get_dark_level()
            for i,wlen in enumerate(wlenrange):
                self.fianium.set_wlen(wlen, aotf="vis")
                self.fianium.set_pwr(ple_laser_power,aotf = "vis")
                sleep(0.01)
                if i==0:
                    cur_int_time = initial_int_time
                else:
                    if last_int_time>10:
                        cur_int_time = last_int_time/10.
                    else:
                        cur_int_time = last_int_time
                self.sia.set_int_time(cur_int_time)		
                self.sia.start_cycle()
                lightlevel = self.sia.ch1_level
                adjust_int_time = cur_int_time*3/lightlevel
                self.desired_int_time = int(adjust_int_time*1000.)/1000.
                adjust_int_time = self.desired_int_time
                if adjust_int_time<=min_exp_sia:
                    adjust_int_time = min_exp_sia
                elif adjust_int_time>self.max_exp_sia:
                    adjust_int_time = self.max_exp_sia
                self.sia.set_int_time(adjust_int_time) 
                #Estimate the exposure time for the camera
                if i==0:
                    exptime = 0.01
                    exptime = int(exptime*100000)/100000.
                    if exptime<min_exp:
                        exptime = min_exp
                    last_exptime = exptime
                else:
                    exptime = last_exptime

                powerlevel, pllevel, real_exptime, dark_level, sia_int_time = self.get_pl_and_power(min_exp, wlen)
                #At this point acquired frame can be overexposed, underexposed or fine 
                #so, figure out what is it and take another frame if needed
                pllevels[i] = pllevel
                powerlevels[i] = powerlevel
                plesignals[i] = pllevel/powerlevel      
                self.powercrv.setData(wlenrange[:i], pllevels[:i])
                self.powercrv22.setData(wlenrange[:i], plesignals[:i])
                QtCore.QCoreApplication.processEvents()
                last_exptime = real_exptime
                last_int_time = sia_int_time
				
            plesignals_all = (plesignals_all*(n_scans-1) + plesignals)/(n_scans)
            self.fianium.set_pwr(0.,aotf = "vis")
            if  self.n_scans_cur == 0:
                self.finalcrv.setData(wlenrange,plesignals_all)
            else:
                self.finalcrv.setData(wlenrange,self.plesignals_total)
            QtCore.QCoreApplication.processEvents()
            if (self.n_scans_cur + n_scans) == self.n_scans_all:
                print "PLE measurement is finished"
                break
            
            n_scans += 1
        #total ple signals during whole measurement
        print "Number of PLE during current areamapping interval", n_scans
        self.plesignals_total = (self.plesignals_total*self.n_scans_cur + plesignals_all*n_scans)/(self.n_scans_cur + n_scans)
        self.n_scans_cur += n_scans
        ple_data = zip(wlenrange,self.plesignals_total)
        power_data = zip(wlenrange,powerlevels)
        sleep(0.5)
        QtCore.QCoreApplication.processEvents()
        f_name_prefix = "_"
        plt.plot(wlenrange,self.plesignals_total)
        plt.savefig(self.dir_name+"\\ple\\"+f_name_prefix+"_ple.png", dpi = 400)
        plt.close()
        plt.plot(wlenrange,powerlevels)
        plt.savefig(self.dir_name+"\\ple\\"+f_name_prefix+"_power.png", dpi = 400)
        plt.close()
        np.savetxt(self.dir_name+"\\ple\\"+f_name_prefix+"_ple.dat", ple_data, fmt='%1.7f',delimiter=",", newline=",\n")
        np.savetxt(self.dir_name+"\\ple\\"+f_name_prefix+"_power.dat", power_data, fmt='%1.7f',delimiter=",", newline=",\n")
        


    def autofocus(self,zstart, z_range, n_step, exp_time,laser_power,laser_position):#, laser_pos_x, laser_pos_y):
        self.mclp.goz(zstart)
        self.obis.start()
        self.obis.set_cw_power()
        self.obis.set_power(laser_power)
        size = 64
        zrange = np.linspace(zstart - z_range/2., zstart + z_range/2., n_step + 1)
        image_area = [laser_position[0] - size/2,laser_position[0] + size/2, laser_position[1]-size/2, laser_position[1] + size/2]
        if np.max(self.max_values) > 0:
            exp_time = self.simple_adjust_exposure(14000, exp_time)
        print "current exposure time", exp_time
        self.camera_setting_change("autofocus",exp_time,image_area)
        fwhm_values = np.zeros(len(zrange))
        gaus_amp_values = np.zeros(len(zrange))
        for ind, cur_z in enumerate(zrange):
            self.mclp.goz(cur_z)
            self.camera.StartAcquisition()
            self.camera.WaitForAcquisition()
            current_data = self.camera.GetAcquiredData([])
            
            self.max_values[ind] = np.max(current_data) - np.min(current_data)
            if self.max_values[ind] > 40000:
                print "autofocus laser intensity is too large"

                break
            print "maximum counts",self.max_values[ind]
            print "z position", cur_z
            current_data = np.asarray(current_data)
            current_data = np.resize(current_data,(size+1,size+1))
            gauss_fit_parameters = self.fitgaussian(current_data)
            gaus_amp_values[ind] = gauss_fit_parameters[0]
            fwhm_values[ind] = 0.25*2*np.log(2)*(np.abs(gauss_fit_parameters[3]*gauss_fit_parameters[4]))**(1./4.)
        min_element = np.unravel_index(fwhm_values.argmin(), fwhm_values.shape)
        z_position = zrange[min_element[0]]
        
        self.mclp.goz(z_position)
        self.obis.set_power(0)
        print "z_position for focusing", z_position
        return z_position,exp_time

    def moments(self,data):
        """
        Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments 
        """
        total = data.sum()
        X, Y = np.indices(data.shape)
        x = (X*data).sum()/total
        y = (Y*data).sum()/total
        col = data[:, int(y)]
        width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
        row = data[int(x), :]
        width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
        height = data.max()
        return height, x, y, width_x, width_y

    def fitgaussian(self,data):
        """
        Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit
        """
        data = data-data.min()
        params = self.moments(data)
        errorfunction = lambda p: np.ravel(self.gaussian(*p)(*np.indices(data.shape))-data)
        p, success = optimize.leastsq(errorfunction, params)
        return p

    def gaussian(self,height, center_x, center_y, width_x, width_y):
        """Returns a gaussian function with the given parameters"""
        width_x = float(width_x)
        width_y = float(width_y)
        return lambda x,y: height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

    def adjust_power(self, desired_ampl, pwr0):
		#config camera
        last_pwr = pwr0/2.
        self.obis.set_power(last_pwr)
        self.camera.StartAcquisition()
        self.camera.WaitForAcquisition()
        current_data = self.camera.GetAcquiredData([])
        current_data = np.asarray(current_data)
        peak_ampl_last = np.max(current_data) - np.min(current_data)
        cur_pwr = pwr0
        n_iterations = 0
        while (np.abs((peak_ampl_last-desired_ampl))>desired_ampl*0.2) and n_iterations<20:
            self.obis.set_power(cur_pwr)
            self.camera.StartAcquisition()
            self.camera.WaitForAcquisition()
            current_data = self.camera.GetAcquiredData([])
            current_data = np.asarray(current_data)
            peak_ampl_cur = np.max(current_data) - np.min(current_data)
            
            
            guess_pwr = ((desired_ampl-peak_ampl_last)/(peak_ampl_cur-peak_ampl_last))*(cur_pwr-last_pwr)+last_pwr
            peak_ampl_last = peak_ampl_cur
            last_pwr = cur_pwr
            if guess_pwr==last_pwr:
                return (last_pwr+cur_pwr)/2., peak_ampl_last
            elif guess_pwr>30:
                cur_pwr = 30.
            elif guess_pwr<0:
                cur_pwr = 1.
            else:
                cur_pwr = guess_pwr
            print "Current iteration = ", n_iterations, " current counts amplitude = ", peak_ampl_cur
            print "guessed power = ", guess_pwr
            n_iterations+=1
        return cur_pwr, peak_ampl_last


    def simple_adjust_exposure(self,desired_cnts,initial_exp_time):
        first_estimation_exp_time = initial_exp_time*desired_cnts/np.max(self.max_values)
        print "first_estimation_exp_time = ", first_estimation_exp_time
        return first_estimation_exp_time

    def ple_and_scan(self, **setting_dict):
        stime = time.time()
        #set parameters for rescan
        self.image = setting_dict["subimage"]    #sub image for scan and ple.  bot, top,left, right
        xstart = setting_dict["xstart"]
        xend = setting_dict["xend"]
        ystart = setting_dict["ystart"]
        yend = setting_dict["yend"]
        zstart = setting_dict["zstart"]
        zrange = setting_dict["zend"]
        nx = setting_dict["nx"]
        ny = setting_dict["ny"]
        nz = setting_dict["nz"]
        xstep = (xend - xstart)/(nx-1)
        ystep = (yend - ystart)/(ny-1)
        scan_exp_time = setting_dict["xyscan_exp_t"]
        scan_wlen = setting_dict["xyscan_wlen"]
        scan_laser_power = setting_dict["xyscan_power"]
        #set parameters for auto focus
        af_laser_position =  setting_dict["zscan_pos"]  # vertical, horizontal position of obis laser
        af_laser_power = setting_dict["zscan_power"]    # 0 to 30
        af_exp_time = setting_dict["zscan_exp_t"]       
        self.max_values = np.zeros(nz+1)
        #set parameters for ple
        self.wlen_start = setting_dict["wlen_start"] 
        self.wlen_end = setting_dict["wlen_end"] 
        self.wlen_step = setting_dict["wlen_step"] 
        ple_laser_power = setting_dict["PLE_power"] 
        self.plesignals_total = np.zeros((self.wlen_end - self.wlen_start)/self.wlen_step)
        ple_exp_time = setting_dict["PLE_exp_t"] 
        self.n_scans_all = setting_dict["PLE_scan_num"]        #total number of ple spectra for averaging
        self.n_scans_cur = 0        #current number of ple  
        self.camera_setting_default()
        while (self.n_scans_cur < self.n_scans_all):
            #search for dots between each ple measurement
            self.scan(xstart,ystart,zstart,xend,yend,nx,ny,scan_wlen,scan_laser_power,scan_exp_time)
            if self.n_scans_cur == 0:
                initial_scan = self.counts_data
            dx, dy, x2, y2  = blob.blob_shift(initial_scan, self.counts_data, self.scan_foldername,oversample=4,lscale=0.5)
            x2, y2 = x2*xstep + xstart, y2*ystep + ystart
            self.mclp.goxy(x2, y2)
            z2,af_exp_time = self.autofocus(zstart, zrange, nz, af_exp_time, af_laser_power,af_laser_position)
            print "##############################################################"
            print "current position", x2, y2, z2
            print "Shift um = ", dx*xstep, dy*ystep
            print "#############################################################"
            cur_time = time.time() - stime
            cur_pos = [cur_time, x2, y2, z2]
            with open(self.scan_foldername+"\\current_position.txt", "a+") as file:
                np.savetxt(file, cur_pos, newline = "   ")
                file.write("\n")
            self.ple(ple_laser_power,ple_exp_time)
            
        print "PLE finished"
        print "number of scans = ", self.n_scans_cur
        
  
    


if __name__ == "__main__":
    measurement = Auto_focus_PLE(setting_dict["emission"],setting_dict["background"],setting_dict["response"]) 
    measurement.ple_and_scan(**setting_dict)

