# This is a translation of code originally by Bjarne Thomsen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, splint, splrep, splev, interp1d
from scipy.special import erf

np.random.seed(32)

def gauss (x):
    return np.exp(-0.5 * x**2)/np.sqrt(2.0 * np.pi)

def hgauss (x,s):
    return np.exp(-2.0*(np.sinh(0.5 * asinh(x*s))/x)**2)

def asinh (x):
    # use that asinh(x) = sign(x)*asinh(abs(x))
    z = abs(x)
    # calculate asinh(x) for positive arguments, only
    i = x >= 0
    x[i] = np.log(np.sqrt(z[i]**2 + 1) + z[i])
    j = x < 0
    x[j] = -np.log(np.sqrt(z[j]**2 + 1) + z[j])
    return x

def planck (x, T):
    return 1.40929e16/(np.exp(1.43879e4/(T*x)) - 1)/x**4

def objem_at_all (func, abmag, maglam, param, sigma, lambd):
    # Airmass
    am = 1.2
    # convolve the object spectrum with a Gaussian
    y = lambd.copy()
    # number of points on each side of x0
    N = 30
    tmp_arange = np.arange(2*N + 1) - N
    global lambda_split
    lambda_split = 0.814560
    for i in range(len(lambd)):
        # wavelength of sample point No. i
        x0 = lambd[i]
        #print(x0)
        # Gaussian sigma of sample point No. i
        s0 = sigma[i]
        # step size
        step = 0.4 * s0
        # range [-12*s0, +12*s0] of wavelengths around the point x0
        x = (tmp_arange)*step + x0
        #print(s0)
        # photon fluxes of the transmitted objec spectrum at the discrete wavelengths
        flux = step * atmos_trans(x)**am * call_func(func, abmag, maglam, param, x)
        # if i == 0:
        #     print(atmos_trans(x))
        # calculate the Gaussian weighted sum of the transmitted object spectrum
        y[i] = sum(flux * gauss((x - x0)/s0))/s0
        #print(y[i])
    return y

def atmos_trans (lambd):
    # Split between a UV-Visual part and a NIR part of the spectrum
    y = lambd.copy()
    uvis = lambd < lambda_split
    y[uvis] = dqes_and_intps('uvis', lambd[uvis], 'trans_at_uvis')
    inv_uvis = np.invert(uvis)
    y[inv_uvis] = dqes_and_intps('nir', lambd[inv_uvis], 'trans_at_nir')
    #print(lambd)
    return y



def powerlaw (abmag, maglam, alpha, lambd):
    y = (lambd/maglam)**alpha * 10**(0.4*(26.847 - abmag))/lambd
    return y

def plancklaw (abmag, maglam, T, lambd):
    c2 = 14387.9
    a = 0.5*c2/T
    y = np.exp(a/maglam - a/lambd)*np.sinh(a/maglam)/np.sinh(a/lambd)* \
        (maglam/lambd)**4 * 10**(0.4*(26.847 - abmag))/maglam
    return y

def template (abmag, maglam, bandwidth, lambd):
    n = len(template_x)
    x = template_x
    y = template_y
    # the bandpass must not be wider than half the spectral range
    fwhm = min(bandwidth, (0.5*(x[-1] - x[0])))
    # the central wavelength must not be too close to the endpoints
    xc = min(max(maglam, (x[0] + fwhm)), (x[-1] - fwhm))
    # select all spectral samples within [xc - fwhm, xc + fwhm]
    k = (x >= (xc - fwhm)) & (x <= (xc + fwhm))
    # what should we do if there are no samples in this interval?
    if sum(k) == 0:
        fwhm = 0.5*(x[-1] - x[0])
        xc = 0.5*(x[-1] + x[0])
        k = np.arange(n)
    # calculate the bandpass profile
    P = np.exp(-np.log(2.0) * (2.0 * (x[k] - xc)/fwhm)**4)
    # calculate the scaling factor
    S = 10.0**(0.4 * (26.847 - abmag)) * sum(P/x[k])/sum(P * x[k] * y[k])
    # calculate the photon flux (photons/s/m^2/um) at the sample points
    y = S * x * y
    # we should not extrapolate outside endpoints, instead we use the values at the endpoints.
    x_new = np.array([min(max(xs, x[0]), x[-1]) for xs in x])
    y_spl_template = UnivariateSpline(x, y, s=0, k=2)
    f = y_spl_template(x_new)
    return f

def call_func (func, abmag, maglam, param, x):
    if func == 'powerlaw':
        f = powerlaw(abmag, maglam, param, x)
    elif func == 'placklaw':
        f = plancklaw(abmag, maglam, param, x)
    elif func == 'template':
        f = template(abmag, maglam, param, x)
    return f

def read_vis_sky ():
    global vissky_x, vissky_y
    sky_data = np.loadtxt('data/VIS-sky.dat')
    # extract wavelengths into um
    vissky_x = sky_data[:,0]/10000.0
    # and photon fluxes in photons/m^2/s/um/arcsec^2
    vissky_y = 5034.1125 * vissky_x * sky_data[:,1]
    # the flux must be non negative
    vissky_y = vissky_y.clip(min=0.0)

def read_nir_abs ():
    global nir_x, nir_y
    abs_data = np.loadtxt('data/NIR-abs.dat')
    # extract wavelengths (in um) and transmissions.
    nir_x = abs_data[:,0]
    nir_y = abs_data[:,1]
    # the transmission must be in the range [0,1].
    nir_y = nir_y.clip(min=1.0e-6, max=1.0)

def read_uvis_ext ():
    global uvis_x, uvis_y, uvis_y2
    ext_data = np.loadtxt('data/UVIS-ext.dat')
    # extract wavelength (in um) and transmission of the atmosphere.
    uvis_x = ext_data[:,0]/1000.0
    uvis_y = 10.0**(-0.4 * ext_data[:,1])
    # optain the second derivative of the interpolating function.
    y_spl_uvis = UnivariateSpline(uvis_x, uvis_y, s=0, k=2)
    y_spl_2d = y_spl_uvis.derivative(n=2)
    uvis_y2 = y_spl_2d(uvis_x)

def read_vis_dqe ():
    global vis_x, vis_y, vis_y2
    dqe_data = np.loadtxt('data/VIS_arm_DQE.dat')
    # extract wavelength (in um) and DQE of VIS arm of X-Shooter.
    vis_x = dqe_data[:,0]
    vis_y = dqe_data[:,1]
    # the DQE must be in the range [0,1].
    vis_y = vis_y.clip(min=1.0e-6, max=1.0)
    # optain the second derivative of the interpolating function.
    y_spl_vis = UnivariateSpline(vis_x, vis_y, s=0, k=2)
    y_spl_2d = y_spl_vis.derivative(n=2)
    vis_y2 = y_spl_2d(vis_x)

def read_oh ():
    global oh_x, oh_y
    oh_data = np.loadtxt('data/OH-lines.dat')
    # convert wavelengths to um.
    oh_x = oh_data[:,0]/10000.0
    # convert wavelengths from vacuum to air
    oh_x = oh_x / ((255.4/(41 - (1/oh_x)**2) + 29498.1/(146 - (1/oh_x)**2) + 64.328)*1e-6 + 1)
    # save calibrated photon fluxes (in ph/m^2/s/arcsec^2). the calibration is done by using ESO's ETC.
    oh_y = oh_data[:,1]/38.4
    # the fluxes must be non-negative.
    oh_y = oh_y.clip(min=0.0)

def read_ir_dqe ():
    global ir_x, ir_y, ir_y2
    dqe_data = np.loadtxt('data/IR_arm_DQE.dat')
    # extract wavelength (in um) and DQE of IR arm of X-Shooter.
    ir_x = dqe_data[:,0]
    ir_y = dqe_data[:,1]
    # the DQE must be in the range [0,1].
    ir_y = ir_y.clip(min=1.0e-6, max=1.0)
    # optain the second derivative of the interpolating function.
    y_spl = UnivariateSpline(ir_x, ir_y, s=0, k=2)
    y_spl_2d = y_spl.derivative(n=2)
    ir_y2 = y_spl_2d(ir_x)


def init_snc_at_vis_and_ir (type, slit_width, disk_scale, psf_fwhm , bin_w=0, bin_l=0):
    if type == 'vis':
        # read night sky emission data
        read_vis_sky()
        # read DQE data for the VIS arm of X-Shooter into
        read_vis_dqe()
    elif type == 'ir':
        # read OH data
        read_oh()
        # read DQE data for the IR arm of X-Shooter
        read_ir_dqe()

    # read atmospheric absorption data
    read_nir_abs()
    # read atmospheric extinction data
    read_uvis_ext()
    
    # copy parameters into the common block
    global s_width, s_eff, psf_sig, h_s, s_length, pix_length, pix_width, \
        pix_ron, pix_dark, ff_error, tel_area
    s_width = slit_width    # slit width in arcsec
    s_eff = erf(np.sqrt(np.log(2.0)) * slit_width / psf_fwhm)   # flux fraction passing slit
    psf_sig = psf_fwhm/np.sqrt(8.0*np.log(2.0)) # Sigma of Gaussian PSF in arcsec
    h_s = disk_scale/psf_sig    # disk scale of galaxy to PSF sigma
    s_length = 20.0 # slit length in arcsec
    if type == 'vis':
        pix_length = 0.40 * bin_l   # pixel length in arcsec
        pix_width = 0.40 * bin_w    # pixel width in arcsec
        pix_ron = 2.5   # readout noise in electrons/pixel
        pix_dark = 0.001 * bin_l * bin_w    # dark current in electrons/pixel/s
    elif type == 'ir':
        pix_length = 0.40   # pixel length in arcsec
        pix_width  = 0.40   # pixel width in arcsec
        pix_ron = 3.0   # readout noise in electrons/pixel
        pix_dark = 0.01     # dark current in electrons/pixel/s
    ff_error = 0.001    # flat field error as a fraction
    tel_area = 4.79833  # telescope area in m^2


#init_snc_at_vis(1,1,1,1,1)


def lamgen_at_vis_and_ir (lambda_min, lambda_max):
    # the middle wavelength on a logarithmic scale
    lambda0 = np.sqrt(lambda_min * lambda_max)
    # size of a pixel in wavelength units at lambda0
    dlambda0 = pix_width / (4000.0/lambda0)
    x_min = min(lambda_min, lambda_max)
    x_max = max(lambda_min, lambda_max)
    ratio = x_max/x_min
    # number of wavelength points in output array
    N = np.round(lambda0 * np.log(ratio)/dlambda0) + 2
    # float array in interval [0.0, 1.0]
    x = np.arange(N, dtype=float)/(N - 1)
    x = x_min * ratio**x
    return x

#lamgen_at_vis(0.1,1,1)

def y_spls (type):
    global y_spl_vis, y_spl_nir, y_spl_uvis, y_spl_vissky, y_spl_ir
    if type == 'vis':
        y_spl_vis = UnivariateSpline(vis_x, vis_y, s=0, k=3)
        y_spl_nir = UnivariateSpline(nir_x, nir_y, s=0, k=3)
        y_spl_uvis = UnivariateSpline(uvis_x, uvis_y, s=0, k=3)
        #y_spl_vissky = UnivariateSpline(vissky_x, vissky_y, s=0, k=3)
        y_spl_vissky = interp1d(vissky_x, vissky_y, kind='slinear')
    elif type == 'ir':
        y_spl_nir = UnivariateSpline(nir_x, nir_y, s=0, k=3)
        y_spl_uvis = UnivariateSpline(uvis_x, uvis_y, s=0, k=3)
        y_spl_ir = UnivariateSpline(ir_x, ir_y, s=0, k=3)


def dqes_and_intps (name, lambd, t):
    x_name = globals()[name + '_x']
    y_name = globals()['y_spl_' + name]
    x = np.array([min(max(l, x_name[0]), x_name[-1]) for l in lambd])
    y = y_name(x)
    if t == 'dqe_at_ir' or t == 'dqe_at_vis':
        y = 0.8 * y
    elif t == 'trans_at_nir':
        y = y.clip(min=0.0)
        #print(x)
    elif t == 'intp_vis_sky':
        y = y
    elif t == 'trans_at_uvis':
        y = y
    elif t == 'test':
        print(x)
        
    else:
        print('MISTAKE')
        exit()
    return y

def skycont_at_ir (lambd):
    lamb_J = 1.25
    flux_J = 310.0
    lamb_H = 1.665
    flux_H = 590.0
    x = lambd
    HoJ = np.log(lamb_H/lamb_J)
    exp_J = np.log(lamb_H/x)/HoJ
    exp_H = np.log(x/lamb_J)/HoJ
    y = flux_J**exp_J * flux_H**exp_H
    return y

def thermalem_at_ir (lambd):
    Temp = 288.0
    emis = 0.25
    x = lambd
    y = (1 - (1 - emis) * dqes_and_intps('nir', x, 'trans_at_nir')) * planck(Temp, x)
    return y

def skyem_at_vis_and_ir (type, sigma, lambd):
    # Airmass
    am = 1.2
    if type == 'vis':
        # step size
        step = 0.0000030
    elif type == 'ir':
        # photon fluxes of the transmitted OH lines
        oh_flux = dqes_and_intps('nir', oh_x, 'trans_at_nir')**am * oh_y
    y = lambd.copy()
    # convolve the sky spectrum with a Gaussian
    for i in range(len(lambd)):
        # wavelength of sample point No. i
        x0 = lambd[i]
        # Gaussian sigma of sample point No. i
        s0 = sigma[i]
        if type == 'vis':
            # number of points on each side of x0
            N = np.round(12.0*s0/step)
        elif type == 'ir':
            # number of points on each side of x0
            N = 30
            # step size
            step = 0.4 * s0
        # range [-12*s0, +12*s0] of wavelengths around the point x0
        x = (np.arange(2*N + 1) - N)*step + x0
        if type == 'vis':
            # photon fluxes of the transmitted spectrum at the discrete wavelengths
            flux = step * dqes_and_intps('uvis', x, 'trans_at_uvis')**am * dqes_and_intps('vissky', x, 'intp_vis_sky')

        elif type == 'ir':
            # photon fluxes of the transmitted continuum at the discrete wavelengths
            flux = step * dqes_and_intps('nir', x, 'trans_at_nir')**am * skycont_at_ir(x)
        # calculate the Gaussian weighted sum of the transmitted sky spectrum.
        y[i] = sum(flux * gauss((x - x0)/s0))/s0
        if type == 'ir':
            # distances of the OH lines from the sample point x0, in units of the Gaussian sigma s0.
            z = (oh_x - x0)/s0
            # select the lines that deviates less than 12*sigma, and add the Gaussian weighted sum of the transmitted fluxes.
            k = abs(z) < 12.0
            y[i] = y[i] + sum(oh_flux[k]*gauss(z[k]))/s0
    
    if type == 'ir':
        y = y + thermalem_at_ir(lambd)

    return y

def s2n (ron, fe, dark, h_s, sig, flux, sky, x):
    # calculate the object profile
    P = hgauss(h_s, x/sig)
    # normalize the profile sum to 1.0
    P = P/sum(P)
    # calculate the backgrount variance per pixel
    bg = sky + dark + ron**2 + (fe * sky)**2
    # calculate the weight per pixel along the slit
    W = 1/(flux * P + bg)
    # shift the profile such that total(W*Q) = 0
    Q = P - sum(W*P)/sum(W)
    # calculate the signal-to-noise per pixel
    y = flux * np.sqrt(sum(W*Q**2))
    return y



def snc_at_vis_and_ir(type, nexp, etime, func, abmag, maglam, param, lambd):
    # number of wavelength points
    Npts = len(lambd)
    # Gaussian sigmas in wavelength space corresponding to the given slit width
    sigma = s_width * gauss(0.0) / (4000.0/lambd)
    #print(sigma)
    # number of pixels along the slit
    Npix = np.round(s_length / pix_length)
    # wavelength interval corresponding to the size of a pixel
    dlambda = pix_width / (4000.0/lambd)
    # DQE at each of the wavelength points
    if type == 'vis':
        dqe = dqes_and_intps('vis', lambd, 'dqe_at_vis')
    elif type == 'ir':
        dqe = dqes_and_intps('ir', lambd, 'dqe_at_ir')
    # sky emission in photoelectrons/m^2/s/arcsec^2
    if type == 'vis':
        skyem = dqe * dlambda * skyem_at_vis_and_ir('vis', sigma, lambd)
        # print(skyem_at_vis_and_ir('vis', sigma, lambd))
    elif type == 'ir':
        skyem = dqe * dlambda * skyem_at_vis_and_ir('ir', sigma, lambd)
    # sky emission in photoelectrons/pixel/exposure
    skyem = (tel_area * etime * pix_length * s_width)*skyem
    # object flux in photons/m^2/s/um
    objem = objem_at_all(func, abmag, maglam, param, sigma, lambd)
    #print(objem)
    # object flux in photoelectrons/m^2/s
    objem = dqe * dlambda * objem
    # object flux in photoelectrons/exposure
    objem = (tel_area * etime * s_eff) * objem
    # dark current in electrons/pixel/exposure
    dark = pix_dark * etime
    # sigma of the Gaussian seeing core in pixels along the slit.
    sig_pix = psf_sig / pix_length
    #print(skyem)
    # calculate the S/N for each wavelength
    y = np.zeros((Npts, 2))
    # there are Npix pixels along the slit
    x_s2n = np.arange(Npix, dtype=float)
    # place the object profile at the center of the slit
    x_s2n = x_s2n - (Npix - 1)/2
    for i in range(Npts):
        # let us first obtain the S/N for a single exposure
        y[i,0] = s2n(pix_ron, ff_error, dark, h_s, sig_pix, objem[i], skyem[i], x_s2n)
        # and then the S/N for the sum of nexp exposures
        y[i,0] = np.sqrt(float(nexp)) * y[i,0]
        # finally we simulate the sum of nexp exposures
        y[i,1] = (1.0 + np.random.normal() / y[i,0]) * float(nexp) * objem[i]
        #y[i,1] = (1.0 + 0.1 / y[i,0]) * float(nexp) * objem[i]
    #print(y)
    return y


#snc_at_vis(1,1,'template',1,1,1,np.array([1,2,3]),1,1,1,1,np.array([1,2,3]),np.array([1,2,3]),np.array([1,2,3]),np.array([1,2,3]),np.array([1,2,3]),np.array([1,2,3]),1,np.array([1,2,3]),np.array([1,2,3]),np.array([1,2,3]),np.array([1,2,3]), 1, 1, 1, 1, 1, 1)




if __name__ == '__main__':
    # First we have to get the input, and decode it
    data = pd.read_csv('data/etc_input.dat', header=None)
    #print(data)
    data = np.array(data[0])
    #print(type(data))
    data = np.array([dat.split()[0] for dat in data])
    #print('HERE')
    

    Exp_Time = float(data[0])
    ObjectType = data[1][0]
    Parameter = float(data[1][1:])
    AB_mag = float(data[2])
    MagLam = float(data[3])
    LambdaMin = float(data[4])
    LambdaMax = float(data[5])
    SlitWidth = float(data[6])
    PSF_FWHM = float(data[7])
    Binning_W = int(data[8])
    Binning_L = int(data[9])
    SN_Binning = int(data[10])
    N_Exp = int(data[11])

    # The binning factors must be greater than or equal to 1
    Binning_W = max(Binning_W, 1)
    Binning_L = max(Binning_L, 1)
    SN_Binning = max(SN_Binning, 1)
    N_Exp = max(N_Exp, 1)

    # The kernel for binning the Signal-to-Noise must have an odd number of elements: (2*N2 + 1)
    N2 = int(SN_Binning/2)
    N_kernel = int(2*N2 + 1)

    # The kernel elements are set to 1.0
    Kernel = np.ones(N_kernel, dtype=float)

    # I assume that wavelengths are given in Angstrom
    if ObjectType == 'T':
        Parameter /= 10000.0
    MagLam /= 10000.0
    LambdaMin /= 10000.0
    LambdaMax /= 10000.0

    # For now, we patch object type
    if ObjectType == 'P':
        ObjectType = 'powerlaw'
    if ObjectType == 'B':
        ObjectType = 'plancklaw'
    if ObjectType == 'T':
        ObjectType = 'template'

    ##ObjectType = 'template'
    # Read the template spectrum if needed
    if ObjectType == 'template':
        template_data = np.loadtxt('data/etc_template.dat')

        # Wavelength in um
        template_x = template_data[:,0]

        # Spectral flux in arbitrary units per wavelengthinterval
        template_y = template_data[:,1]

        # Flux density must be non negative
        template_y = template_y.clip(min=1.0E-30)

        #The smapling must be sorted according to increasing wavelength
        k = np.argsort(template_x)
        template_x = template_x[k]
        template_y = template_y[k]
    else:
        template_x = np.array([1,2,3])
        template_y = np.array([1,2,3])

    # For now we set DiskScale to a fixed low value of 0.05
    DiskScale  = 0.05

    # Now we get the wavelength ranges for the individual arms
    Range_VIS = np.array([0.32, 0.76])
    Range_IR = np.array([0.76, 1.80])

    Cov_VIS = [max(LambdaMin, Range_VIS[0]), min(LambdaMax, Range_VIS[1])]
    Cov_IR = [max(LambdaMin, Range_IR[0]), min(LambdaMax, Range_IR[1])]
    

    # Now calculate the S/N for each of the arms
    # VIS arm
    s_n_VIS = 0.0
    sim_VIS = 0.0

    #Cov_VIS[0] = 0.32
    if Cov_VIS[0] < Cov_VIS[1]:
        type = 'vis'
    if Cov_VIS[0] < Cov_VIS[1]:
        init_snc_at_vis_and_ir('vis', SlitWidth, DiskScale, PSF_FWHM , Binning_W , Binning_L)
        y_spls ('vis')
        lam_VIS = lamgen_at_vis_and_ir(Cov_VIS[0], Cov_VIS[1])
        #print(lam_VIS)
        snc = snc_at_vis_and_ir('vis', N_Exp,Exp_Time,ObjectType,AB_mag,MagLam,Parameter,lam_VIS)
        #print(snc)
        s_n_VIS = snc[:,0]
        sim_VIS = snc[:,1]
        s_n_sqr = s_n_VIS**2
        s_n_VIS = np.sqrt(np.convolve(s_n_sqr, Kernel, mode='same'))
        # The first and last N2 elements of the convolved array are set to 0.0 by the function convol(), so we must truncate s_n_VIS and lam_vis by N2 elements at both ends.
        lam_VIS = lam_VIS[N2:len(lam_VIS)-N2]
        s_n_VIS = s_n_VIS[N2:len(s_n_VIS)-N2]
        sim_VIS = sim_VIS[N2:len(sim_VIS)-N2]
        print('Median VIS S/N = ' + str(np.median(s_n_VIS)))
        print(np.mean(s_n_VIS))
        print(np.min(s_n_VIS))
        print(np.max(s_n_VIS))
    
    # IR arm
    s_n_IR = 0.0
    sim_IR = 0.0

    #Cov_IR[0] = 0.001
    if Cov_IR[0] < Cov_IR[1]:
        type = 'ir'
    if Cov_IR[0] < Cov_IR[1]:
        init_snc_at_vis_and_ir('ir', SlitWidth, DiskScale, PSF_FWHM)
        y_spls ('ir')
        lam_IR = lamgen_at_vis_and_ir(Cov_IR[0], Cov_IR[1])
        snc = snc_at_vis_and_ir('ir', N_Exp,Exp_Time,ObjectType,AB_mag,MagLam,Parameter,lam_IR)
        #print(snc)
        s_n_IR = snc[:,0]
        sim_IR = snc[:,1]
        s_n_sqr = s_n_IR**2
        s_n_IR = np.sqrt(np.convolve(s_n_sqr, Kernel, mode='same'))
        # The first and last N2 elements of the convolved array are set to 0.0 by the function convol(), so we must truncate s_n_IR and lam_ir by N2 elements at both ends.
        lam_IR = lam_IR[N2:len(lam_IR)-N2]
        s_n_IR = s_n_IR[N2:len(s_n_IR)-N2]
        sim_IR = sim_IR[N2:len(sim_IR)-N2]
        print('Median IR S/N = ' + str(np.median(s_n_IR)))
        print(np.mean(s_n_IR))
        print(np.min(s_n_IR))
        print(np.max(s_n_IR))


    # Now to the plotting
    text	= np.zeros(11, dtype='object')
    param	= np.zeros(11, dtype='object')
    text[0] = 'Exposure Time (single exposure):'
    text[1] = 'Slit Width:'
    text[2] = 'FWHM of PSF:'
    text[3] = 'Object Type:'
    if ObjectType == 'powerlaw':
        text[4] = 'Spectral Index:'
    elif ObjectType == 'plancklaw':
        text[4] = 'Temperature:'
    elif ObjectType == 'template':
        text[4] = 'FWHM of Photometric Band:'
    text[5]  = 'AB Magnitude:'
    text[6]  = 'at wavelength:'
    text[7]  = 'Detector binning along dispersion:'
    text[8]  = 'Detector binning along slit:'
    text[9]  = 'Signal-to-Noise binning factor:'
    text[10] = 'Number of exposures:'

    param[0]  = str('{:.1f}'.format(Exp_Time)) + ' s' #, FORMAT='(F7.1," s")'),2)
    param[1]  = str('{:.2f}'.format(SlitWidth)) + ' arcsec'
    param[2]  = str('{:.2f}'.format(PSF_FWHM)) + ' arcsec'
    param[3]  = ObjectType
    if ObjectType == 'powerlaw':
        param[4]  = str('{:.2f}'.format(Parameter))
    elif ObjectType == 'plancklaw':
        param[4]  = str('{:.1f}'.format(Parameter)) + ' K'
    elif ObjectType == 'template':
        param[4]  = str('{:.1f}'.format(10000.0*Parameter)) + ' A'
    param[5]  = str('{:.2f}'.format(AB_mag))
    param[6]  = str('{:.1f}'.format(10000.0*MagLam)) + ' A'
    param[7]  = str(np.round(Binning_W)) + ' pixels'
    param[8]  = str(np.round(Binning_L)) + ' pixels'
    param[9]  = str(np.round(N_kernel)) + ' binned pixels/channel'
    param[10] = str(np.round(N_Exp)) + ' exposures'

    # plot the Signal-to-Noise

    plt.figure(figsize=(15,10))
    plt.title('Signal-to-Noise', size=22)
    plt.xlabel(r'Wavelength  [ $\mu m$ ]', size=18)
    plt.ylabel('S/N  [ per channel ]', size=18)
    plt.xlim(LambdaMin, LambdaMax)
    y_max	= np.max( np.append(s_n_VIS, s_n_IR) )
    plt.ylim(-1.4*y_max , 1.4*y_max)

    if ( Cov_VIS[0] < Cov_VIS[1] ):
        plt.plot(lam_VIS, s_n_VIS, lw=1, color='k') #, psym=10
    if ( Cov_IR[0] < Cov_IR[1] ):
        plt.plot(lam_IR,  s_n_IR, lw=1, color='k') #, psym=10

    dx	= ( LambdaMax - LambdaMin ) / 15.0
    dy	= y_max / 9.0
    x1	= LambdaMin + dx
    x2	= LambdaMin + 9*dx
    y = -1.4*dy
    for i in range(len(text)):
        plt.text(x1, y, text[i], size=14)
        plt.text(x2, y, param[i], size=14)
        y = y - dy

    plt.savefig('Signal-to-Noise.eps')

    # plot the simulated spectrum

    y_max = np.max( np.append(sim_VIS, sim_IR) )
    y_min = np.min( np.append(sim_VIS, sim_IR) )
    y_ave = 0.5*(y_min + y_max)
    y_hra = 0.5*(y_max - y_min)

    plt.figure(figsize=(15,10))
    plt.title('Simulated Spectrum', size=22)
    plt.xlabel(r'Wavelength  [ $\mu m$ ]', size=18)
    plt.ylabel('Counts per Channel', size=18)
    plt.xlim(LambdaMin, LambdaMax)
    plt.ylim(y_ave + -1.2*y_hra , y_ave + 1.2*y_hra)

    if ( Cov_VIS[0] < Cov_VIS[1] ):
        plt.plot(lam_VIS, sim_VIS, lw=1, color='k')
    if ( Cov_IR[0] < Cov_IR[1] ):
        plt.plot(lam_IR,  sim_IR, lw=1, color='k')

    plt.savefig('Simulated-Spectrum.eps')








