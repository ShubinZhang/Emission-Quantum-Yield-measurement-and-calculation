#setting parameters for QY measurement    
setting_measurement = {
    "rcurve": "rcurve.dat",     #previouse measured spectral responsivity curve file name
    "wlen_start":450,           #start exictation wavelength  (nm)
    "wlen_end":480,             #end exictation wavelength  (nm)
    "wlen_step":8,              #wavelength step size (nm)
    "time_int":60000,           #exposure time (us)
    "avg":50,                   #number of averaging 
    "xmin":350,                 #minimum wavelength in spectrum (nm)
    "xmax":850,                 #maximum wavelength in spectrum (nm)
    "mkdir":False,              #create directory and save data
}

#setting parameters for calculating QY  
setting_calculation = {
    "blank": "2018624_1558_QY",     #blank data folder name
    "sample": "2018624_1610_QY",    #blank data folder name
    "excitation_range":[440, 490],  #excitation fitting range          
    "emission_range":[500, 650],    #emssion fitting range    
    "plot": True                    #if True, show fitting result
}

#excitation spectrum fitting parameter initial guess, optional
exc_guess = None

#emission spectrum fitting parameter initial guess, required
em_guess = {
"g1_center": 515.2596603475121
"g1_amplitude": 0.09692581023857087
"g1_sigma": 11.387930832721327
"g2_center": 514.8964191482476
"g2_amplitude": 0.04663039189024973
"g2_sigma": 6.112296235878089
"g3_center": 520.2925364847694
"g3_amplitude": 0.0375074672177262
"g3_sigma": 19.45467875195471   
}