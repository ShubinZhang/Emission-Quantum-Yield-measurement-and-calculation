'''
Python script to calculate the emission quantum yields with differrnt excitation wavelength.

Original code by
Copyright (C) 2017  Shubin Zhang

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation
'''
import numpy as np
import matplotlib.pyplot as plt
import kit as tk
import os
import re
from parse import parse
from scipy import interpolate
from lmfit.models import GaussianModel, LinearModel
from config import setting_calculation, exc_guess, em_guess

class gausfit():
    def fit_2gaus_em(self, xs, ys, guess, vr12, vrpl = 1.2):
        """
        Fit emission spectrum with 2 Gaussian distributions 
        xs/ys: wavelength/counts of emission spectrum
        guess: inital guess of fitting Gaussian distribution parameters (amplitude, center, width)
        vr12: if True, change centers and widths of fitting Gaussian distribution under differnt excitation wavelengths
        vrpl: maximum change factor of fitting Gaussian distribution amplitudes under differnt excitation wavelengths
        """
    
        gauss1  = GaussianModel(prefix='g1_')
        pars = gauss1.make_params()
        
        pars.update( gauss1.make_params())
        pars['g1_center'].set(guess['g1_center'], min = guess['g1_center']/vrpl, max = guess['g1_center']*vrpl, vary = vr12)
        pars['g1_sigma'].set(guess['g1_sigma'], min = guess['g1_sigma']/vrpl, max = guess['g1_sigma']*vrpl, vary = vr12)
        pars['g1_amplitude'].set(guess['g1_amplitude'], min = guess['g1_amplitude']/2, max = guess['g1_amplitude']*2)
        
        gauss2  = GaussianModel(prefix='g2_')
        pars.update(gauss2.make_params())

        pars['g2_center'].set(guess['g2_center'], min = guess['g2_center']/vrpl, max = guess['g2_center']*vrpl, vary = vr12)
        pars['g2_sigma'].set(guess['g2_sigma'], min = guess['g2_sigma']/vrpl, max = guess['g2_sigma']*vrpl, vary = vr12)
        pars['g2_amplitude'].set(guess['g2_amplitude'], min = guess['g2_amplitude']/2, max = guess['g2_amplitude']*2)
        

        mod1 = gauss1 + gauss2
        
        init = mod1.eval(pars, x=xs)
        out = mod1.fit(ys, pars, x=xs)
        total_amp = out.params['g1_amplitude'].value + out.params['g2_amplitude'].value 
        parsl = out.params
        return out,xs, ys,total_amp



    def fit_3gaus_em(self, xs,ys, guess, vr123, vrpl = 1.2):
        """
        Fit emission spectrum with 3 Gaussian distributions 
        xs/ys: wavelength/counts of emission spectrum
        guess: inital guess of fitting Gaussian distribution parameters (amplitude, center, width)
        vr123: if True, change centers and widths of fitting Gaussian distribution under differnt excitation wavelengths
        vrpl: maximum change factor of fitting Gaussian distribution amplitudes under differnt excitation wavelengths
        """
        gauss1  = GaussianModel(prefix='g1_')
        pars = gauss1.make_params()
        pars.update( gauss1.make_params())
        pars['g1_center'].set(guess['g1_center'], min = guess['g1_center']/vrpl, max = guess['g1_center']*vrpl, vary = vr123)
        pars['g1_sigma'].set(guess['g1_sigma'], min = guess['g1_sigma']/vrpl, max = guess['g1_sigma']*vrpl, vary = vr123)
        pars['g1_amplitude'].set(guess['g1_amplitude'], min = guess['g1_amplitude']/2, max = guess['g1_amplitude']*2)
        
        gauss2  = GaussianModel(prefix='g2_')
        pars.update(gauss2.make_params())

        pars['g2_center'].set(guess['g2_center'], min = guess['g2_center']/vrpl, max = guess['g2_center']*vrpl, vary = vr123)
        pars['g2_sigma'].set(guess['g2_sigma'], min = guess['g2_sigma']/vrpl, max = guess['g2_sigma']*vrpl, vary = vr123)
        pars['g2_amplitude'].set(guess['g2_amplitude'], min = guess['g2_amplitude']/2, max = guess['g2_amplitude']*2)
        
        gauss3  = GaussianModel(prefix='g3_')
        pars.update(gauss3.make_params())

        pars['g3_center'].set(guess['g3_center'], min = guess['g3_center']/vrpl, max = guess['g3_center']*vrpl, vary = vr123)
        pars['g3_sigma'].set(guess['g3_sigma'], min = guess['g3_sigma']/vrpl, max = guess['g3_sigma']*vrpl, vary = vr123)
        pars['g3_amplitude'].set(guess['g3_amplitude'], min = guess['g3_amplitude']/2, max = guess['g3_amplitude']*2)
        

        mod1 = gauss1 + gauss2 + gauss3
        
        init = mod1.eval(pars, x=xs)
        out = mod1.fit(ys, pars, x=xs)
        total_amp = out.params['g1_amplitude'].value + out.params['g2_amplitude'].value + out.params['g3_amplitude'].value
        pars = out.params
        return out,xs, ys,total_amp



    def fit_2gaus(self,x,y, deltax, guess = None):
        """
        Fit excitation spectrum with 2 Gaussian distributions 
        x/y: wavelength/counts of excitation spectrum
        deltax: spectrum range where excitation will be fitted
        guess: inital guess of fitting Gaussian distribution parameters (amplitude, center, width), if None, guess fitting paramters automatically
        """
        
        # estimate initial gueses of the fitting parameters
        maxs = np.max(y)
        curw = x[np.where(y==maxs)]
        xs, ys = tk.cutdata(x,y, curw-deltax, curw+deltax)
        xpos = np.sum(xs*ys)/np.sum(ys)     
        xs, ys = tk.cutdata(x,y, xpos-deltax, xpos+deltax)
        amps = np.sum(ys)
        sigmaguess = 5
        
        #initialize the fitting function 
        if guess == None:
            gauss1  = GaussianModel(prefix='g1_')
            parsl = gauss1.make_params()
            parsl.update( gauss1.make_params())

            parsl['g1_center'].set(xpos , min=xpos-0.2*sigmaguess, max = xpos+0.2*sigmaguess)
            parsl['g1_sigma'].set(sigmaguess/3., min = sigmaguess/5., max = sigmaguess/2.)
            parsl['g1_amplitude'].set(amps/1.1, min = amps/10., max = amps)

            gauss2  = GaussianModel(prefix='g2_')
            parsl.update(gauss2.make_params())

            parsl['g2_center'].set(xpos + sigmaguess, min=xpos+0.8*sigmaguess, max = xpos+1.2*sigmaguess)
            parsl['g2_sigma'].set(sigmaguess/5, min = sigmaguess/10., max = sigmaguess/2.)
            parsl['g2_amplitude'].set(amps/20., min = amps*10**(-2), max = amps/10.)
        else:
            gauss1  = GaussianModel(prefix='g1_')
            parsl = gauss1.make_params()
            parsl.update( gauss1.make_params())

            parsl['g1_center'].set(guess['g1_center'] , min=guess['g1_center']-0.2*sigmaguess, max = guess['g1_center']+0.2*sigmaguess, vary = False)
            parsl['g1_sigma'].set(guess['g1_sigma'], min = guess['g1_sigma']/5., max = guess['g1_sigma']*2., vary = False)
            parsl['g1_amplitude'].set(guess['g1_amplitude'], min = guess['g1_amplitude']/10., max = guess['g1_amplitude']*10)

            gauss2  = GaussianModel(prefix='g2_')
            parsl.update(gauss2.make_params())

            parsl['g2_center'].set(guess['g2_center'] , min=guess['g2_center'] -0.2*sigmaguess, max = guess['g2_center'] +0.2*sigmaguess, vary = False)
            parsl['g2_sigma'].set(guess['g2_sigma'], min = guess['g2_sigma']/10., max = guess['g2_sigma']*2.,vary = False)
            parsl['g2_amplitude'].set(guess['g2_amplitude'], min = guess['g2_amplitude']*10**(-2), max = guess['g2_amplitude']*10.)



        mod1 = gauss1 + gauss2 #2020213_1012_ple
        
        init = mod1.eval(parsl, x=xs)
        out = mod1.fit(ys, parsl, x=xs)
        
        parsl = out.params
        total_amp = out.params['g1_amplitude'].value + out.params['g2_amplitude'].value
        return out,xs, ys, total_amp

class QY_calc(gausfit):
    def __init__(self, setting_calculation):
        self.blank = self.load_files(setting_calculation["blank"])
        self.sample = self.load_files(setting_calculation["sample"])
        self.exc_guess = self.load_files(setting_calculation["exc_guess"])
        self.em_guess = self.load_files(setting_calculation["em_guess"])
        self.x_exc_min, self.x_exc_max = setting_calculation["excitation_range"]
        self.x_em_min, self.x_em_max = setting_calculation["emission_range"]
        self.plot = setting_calculation["plot"]


    def sorted_abcdgt(self, l ): 
        """ Sort the given iterable in the way that humans expect.""" 
        convert = lambda text: int(text) if text.isdigit() else text 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(l, key = alphanum_key)

    def listfldr(self, fldrname, ext='dat', sort=True, fullpath=False):
        """
        List files in the folder fldrname with ext extension
        
        if fullpath set to True, output will be a full path to each file,
        otherwise there are will be only file names
        
        """
        
        matchedfiles = [f for f in os.listdir(fldrname) if f.endswith(ext)]
        if sort:
            #matchedfiles = sorted(matchedfiles)
            matchedfiles = self.sorted_abcdgt(matchedfiles)
            
        if fullpath:
            matchedfiles = [os.path.join(fldrname, f) for f in matchedfiles]
        return matchedfiles


    def cutdata(self, x,y,x1,x2):
        indxs = np.where((x>=x1) & (x<x2))
        sub_y = y[indxs]
        sub_x = x[indxs]
        return sub_x, sub_y

    def load_files(self, fldrname):
        path = os.path.normpath("data\\"+fldrname)
        fldr = os.path.basename("data\\"+fldrname)
        files = tk.listfldr(path,  ext='nm.dat', sort = True, fullpath = True)
        QYdict = {}
        for i, f in enumerate(files):
            curw = float(parse(fldr+"_{}_nm.dat", os.path.basename(f))[0])
            x,y = np.loadtxt(f, unpack = True)
            QYdict[curw] = [x,y]
        return QYdict
    
    def calculation(self, exc_guess, em_guess):
        blank_exc_dict = {}  #dictionary to save final excitation fitting paramters with blank 
        sample_exc_dict = {} #dictionary to save final excitation fitting paramters with sample 
        sample_em_dict = {}  #dictionary to save final emission fitting paramters with sample 
        guess_b_exc = {}     #dictionary to save initial guess for excitation fitting paramters with blank 
        guess_s_exc = {}     #dictionary to save initial guess for excitation fitting paramters with sample 
        guess_s_em = {}      #dictionary to save initial guess for emission fitting arameters with sample 
        for _, wl in enumerate(self.sample.keys()):
            xb, yb = self.blank[wl]
            xs, ys = self.sample[wl]
            xb_exc, yb_exc = self.cutdata(xb,yb, self.x_exc_min, self.x_exc_max)
            xs_exc, ys_exc = self.cutdata(xs,ys, self.x_exc_min, self.x_exc_max)
            xs_em, ys_em = self.cutdata(xs,ys, self.x_em_min, self.x_em_max)

            #fit excitation spectrum with 2 Gaussian distribution
            out_b_exc, xb_fit_exc, yb_fit_exc, total_amp_b_exc = self.fit_2gaus(xb_exc, yb_exc, deltax = 10, guess = exc_guess)
            guess_b_exc['g1_amplitude'], guess_b_exc['g1_center'], guess_b_exc['g1_sigma'] = out_b_exc.params['g1_amplitude'].value, out_b_exc.params['g1_center'].value, out_b_exc.params['g1_sigma'].value
            guess_b_exc['g2_amplitude'], guess_b_exc['g2_center'], guess_b_exc['g2_sigma'] = out_b_exc.params['g2_amplitude'].value, out_b_exc.params['g2_center'].value, out_b_exc.params['g2_sigma'].value
            blank_exc_dict[wl] = {'amp':total_amp_b_exc,
                     'g1_amplitude':out_b_exc.params['g1_amplitude'], 'g1_center':out_b_exc.params['g1_center'], 'g1_sigma':out_b_exc.params['g1_sigma'],
                     'g2_amplitude':out_b_exc.params['g2_amplitude'], 'g2_center':out_b_exc.params['g2_center'], 'g2_sigma':out_b_exc.params['g2_sigma'],
            }

            #fit excitation spectrum with 2 Gaussian distribution
            out_s_exc, xs_fit_exc, ys_fit_exc, total_amp_s_exc = self.fit_2gaus(xs_exc, ys_exc, deltax = 10, guess = exc_guess)
            guess_s_exc['g1_amplitude'], guess_s_exc['g1_center'], guess_s_exc['g1_sigma'] = out_s_exc.params['g1_amplitude'].value, out_s_exc.params['g1_center'].value, out_s_exc.params['g1_sigma'].value
            guess_s_exc['g2_amplitude'], guess_s_exc['g2_center'], guess_s_exc['g2_sigma'] = out_s_exc.params['g2_amplitude'].value, out_s_exc.params['g2_center'].value, out_s_exc.params['g2_sigma'].value
            sample_exc_dict[wl] = {'amp':total_amp_s_exc,
                     'g1_amplitude':out_s_exc.params['g1_amplitude'], 'g1_center':out_s_exc.params['g1_center'], 'g1_sigma':out_s_exc.params['g1_sigma'],
                     'g2_amplitude':out_s_exc.params['g2_amplitude'], 'g2_center':out_s_exc.params['g2_center'], 'g2_sigma':out_s_exc.params['g2_sigma'],
            }

            #fit emission spectrum with 3 Gaussian distribution
            out_s_em, xs_fit_em, ys_fit_em, total_amp_s_em = self.fit_3gaus_em(xs_em, ys_em, guess = em_guess, vr123 = True)
            guess_s_em['g1_amplitude'], guess_s_em['g1_center'], guess_s_em['g1_sigma'] = out_s_em.params['g1_amplitude'].value, out_s_em.params['g1_center'].value, out_s_em.params['g1_sigma'].value
            guess_s_em['g2_amplitude'], guess_s_em['g2_center'], guess_s_em['g2_sigma'] = out_s_em.params['g2_amplitude'].value, out_s_em.params['g2_center'].value, out_s_em.params['g2_sigma'].value
            sample_em_dict[wl] = {'amp':total_amp_s_em,
                     'g1_amplitude':out_s_em.params['g1_amplitude'], 'g1_center':out_s_em.params['g1_center'], 'g1_sigma':out_s_em.params['g1_sigma'],
                     'g2_amplitude':out_s_em.params['g2_amplitude'], 'g2_center':out_s_em.params['g2_center'], 'g2_sigma':out_s_em.params['g2_sigma'],
            }
            if self.plot:
                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.plot(xb_fit_exc, yb_fit_exc, "o")
                ax1.plot(xb_fit_exc, yb_exc)
                ax2.plot(xb_fit_exc, yb_fit_exc-yb_exc)

                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.plot(xs_fit_exc, ys_fit_exc, "o")
                ax1.plot(xs_fit_exc, ys_exc)
                ax2.plot(xs_fit_exc, ys_fit_exc-ys_exc)

                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.plot(xs_fit_em, ys_fit_em, "o")
                ax1.plot(xs_fit_em, ys_em)
                ax2.plot(xs_fit_em, ys_fit_em-ys_em)
        plt.show()

        qy = np.zeros(len(self.sample.keys()))
        trans = np.zeros(len(self.sample.keys()))
        wlngths = list(self.sample.keys())

        for i, w in enumerate(wlngths):
            l1 = blank_exc_dict[w]['amp']
            l2 = sample_exc_dict[w]['amp']
            pl = sample_em_dict[w]['amp']
            qy[i] = pl/(l1-l2)
            trans[i] = l2/l1
            print w,qy[i]
        print 'Average qy = %1.5f'%np.mean(qy), ' std = %1.5f'%np.std(qy)
        print 'Transmittance = %1.5f'%np.mean(trans)