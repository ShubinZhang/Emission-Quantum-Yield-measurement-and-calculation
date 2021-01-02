# **Emission-Quantum-Yield-measurement-and-calculation**

## **About the project**

Emission Quantum Yield (QY) is an important optical parameter for emissive material. It indicates how efficient material emits light. Measuring emission QY pricisely is critical 
in many research area, including optical refrigeration, LED device, etc. This project provides two python scripts to measure and calculate emission QY integrating sphere based home-built optical setup. More detail can be found in https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10936/2507051/Evaluation-of-CsPbBr3-nanocrystals-for-laser-cooling/10.1117/12.2507051.short?SSO=1


## **Prerequisites**

* Instruments are properly connected physically.
* Connection ports in "config.py" are setted correctly.
* Instrument spectral responsivity is measured and saved. A sample respsonsivity file ("rcurve.dat") is provided.
* Python 2.7. Other required packages can be found in "requirements.txt". 

## **Usage**

1. Set up the measurement parameters accoding to the optical properties of measured sample in "config.py" file.
2. Measure blank and sample by executing "QY_measurement.py". Blank and sample data will be save seperately in "data" folder.
3. Input blank and sample data folder name as well as initial guess of fitting parameters in "config.py" file, execute "QY_calculation.py"

## **Liscence**

Distributed under the MIT License. See `LICENSE` for more information.



