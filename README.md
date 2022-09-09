# NTE_ExposureTimeCalculator
An exposure time calculator for NTE.
Very much a work in progress and is currently not working at all.
# How to use:
At this current moment there are 12 inputs to the calculator:
* Exposure time in seconds
* Template {P|B|T}{Index|T in K|FWHM in A}
* AB magnitude of object
* Wavelength (in A) where AB mag is given
* Lower limit wavelength in A
* Upper limit wavelength in A
* Slit width in arcsec
* FWHM of (Gaussian) seeing in arcsec
* Detector binning in dispersion direction
* Detector binning in spatial direction
* Post-detector binning of signal-to-noise
* Number of exposures used to calculate S/N
All of these should be given on a separate line in a file named 'etc\_input.dat' in a folder named 'data', located in the same folder as the 'nte\_etc.py' file. 'nte\_etc.py' is then the file that you run.
