#Settings for auto focus PLE
setting_dict = {
    #Instrument connection setting
    "obis_port": 1,     #obis laser port number

    #Three image areas on camera (xstart, xend, ystart, yend)
    "emission":  [162,167, 296,301] ,
    "background": [184,189, 296,301],
    "subimage": [153,193, 291,306],

    #Scanning parameters xy and z 
    "xstart" : 40.,  #start position in x (um)
    "xend" : 43.,    #end position in x (um)
    "ystart" : 34.,  #start position in y (um)
    "yend" : 37.,    #end position in y (um)
    "zstart" : 10.,  #start position in z (um)
    "zrange" : 8.,   #end position in z (um)
    "nx" : 13,       #number of steps in x
    "ny" : 13,       #number of steps in y
    "nz" : 16,       #number of steps in z
    "xyscan_exp_t": 0.025,    #camera exposure time during xy scanning (s)
    "xyscan_wlen": 460,       #fianium laser wavelength during xy scanning (nm)
    "xyscan_power": 20,       #fianium laser power during xy scanning (%)
    "zscan_pos": [164,299],  #vertical, horizontal position of obis laser
    "zscan_power": 2,         #obis laser power (mw) during z scanning
    "zscan_exp_t": 0.025,    #camera exposure time during z scanning (s)

    #Measuremnt setting
    "wlen_start":450,  #start wavelength of PLE spectrum (nm)
    "wlen_end":520,    #end wavelength of PLE spectrum (nm)
    "wlen_step":2,     #wavelength step size (nm)
    "PLE_power": 17,   #fianium laser power for PLE (%)
    "PLE_exp_t": 1,    #camera exposure time for PLE (s)
    "PLE_scan_num": 16,  #total number of ple spectra for averaging

    "response":"response_data.dat"  #reponse curve file name
    } 