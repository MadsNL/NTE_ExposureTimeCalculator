# NTE_ExposureTimeCalculator
An exposure time calculator for NTE.
Very much a work in progress and is currently not working at all.
# How to use:
At this current moment there are 12 inputs to the calculator:
    * Exposure time in seconds
    \item Template {P|B|T}{Index|T in K|FWHM in A}
    \item AB magnitude of object
    \item Wavelength (in A) where AB mag is given
    \item Lower limit wavelength in A
    \item Upper limit wavelength in A
    \item Slit width in arcsec
    \item FWHM of (Gaussian) seeing in arcsec
    \item Detector binning in dispersion direction
    \item Detector binning in spatial direction
    \item Post-detector binning of signal-to-noise
    \item Number of exposures used to calculate S/N
All of these should be given on a separate line in a file named 'etc\_input.dat' in a folder named 'data', located in the same folder as the 'nte\_etc.py' file. 'nte\_etc.py' is then the file that you run.
