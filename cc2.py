# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 19:55:07 2025

@author: lewis
"""
from sys import platform
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import h5py
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.interpolate import interp1d
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from colossus.lss import peaks
from colossus.lss import bias
from scipy.integrate import simpson as simps
plt.rcParams.update({'font.size': 18})

# Planck 15 cosmology to match TNG50-1-Dark
cosmo = cosmology.setCosmology('planck15')


def nearest(vector, value, DEBUG=True):
    ''' Returns index of nearest value in vector to value '''
    res = (np.abs(vector - value)).argmin()
    return res


def get_HMF(z, Mh):
    ''' #### get_HMF ####
    FUNCTION :
        Retrieves the Halo mass function for the input redshift and halo mass values
    INPUT :
        z  - 1D array of redshift values of shape (nz,)
        Mh - 1D array of halo mass values of shape (nm,)
    OUTPUT :
        HMF - 2D array of shape (nz x nm) containing the logged values of the HMF
        log10(d^2 N / dV / dlog10(Mstar))
    '''
    HMF = np.array([np.log10(mass_function.massFunction(10**Mh*cosmo.h, z[iz], mdef='sp-apr-mn',
                   model='diemer20', q_out='dndlnM') * np.log(10.) * cosmo.h**3.) for iz in range(z.size)])
    return HMF


def get_SMF(zvals, Mvals, work='Weaver+2023'):
    ''' #### get_SMF ####
    FUNCTION :
        Retrieves the galaxy stellar mass function for the input redshift
        and stellar mass values
    INPUT :
        zvals - 1D Array of redshift values of length nz
        Mvals - 1D Array of stellar mass values of length nm
        work  - str Name of the work to use the SMF from
    OUTPUT :
        SMF   - 2D Array of shape (nz x nm) containing the logged values of the
                stellar mass function log10(d^2 N / dV / dlog10(Mstar))
    '''
    SMF = np.zeros((zvals.size, Mvals.size))
    if work == 'Davidzon+2017':
        z_sample = np.array([0.35, 0.65, 0.95, 1.30, 1.75,
                            2.25, 2.75, 3.25, 4.00, 5.00])
        Mref = np.array([10.78, 10.77, 10.56, 10.62, 10.51, 10.60,
                        10.59, 10.83, 11.10, 11.30]) / (cosmo.H0 / 70)**2
        a1 = np.array([-1.38, -1.36, -1.31, -1.28, -1.28, -
                      1.57, -1.67, -1.76, -1.98, -2.11])
        Phi1 = np.array([1.187, 1.070, 1.428, 1.069, 0.969, 0.295,
                        0.228, 0.090, 0.016, 0.003]) * 1e-3 * (cosmo.H0 / 70)**3
        a2 = np.array([-0.43, 0.03, 0.51, 0.29, 0.82,
                      0.07, -0.08, 0.00, 0.00, 0.00])
        Phi2 = np.array([1.92, 1.68, 2.19, 1.21, 0.64, 0.45,
                        0.21, 0.00, 0.00, 0.00]) * 1e-3 * (cosmo.H0 / 70)**3
    elif work == 'Weaver+2023':
        z_sample = np.array([0.35, 0.65, 0.95, 1.30, 1.75,
                            2.25, 2.75, 3.25, 4.00, 5.00, 6.00, 7.00])
        Mref = np.array([10.89, 10.96, 11.02, 11.00, 10.86,
                        10.78, 10.97, 10.83, 10.46, 10.30, 10.14, 10.18])
        a1 = np.array([-1.42, -1.39, -1.32, -1.33, -1.48, -
                      1.46, -1.46, -1.46, -1.46, -1.46, -1.46, -1.46])
        Phi1 = np.array([0.73, 0.66, 0.84, 0.72, 0.29, 0.27,
                        0.24, 0.21, 0.20, 0.14, 0.06, 0.03]) * 1e-3
        a2 = np.array([-0.46, -0.61, -0.63, -0.51, -0.43, 0.07,
                      0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
        Phi2 = np.array([1.09, 0.83, 0.66, 0.34, 0.64, 0.27,
                        0.00, 0.00, 0.00, 0.00, 0.00, 0.00]) * 1e-3
    elif work == 'Tomczak+2014':
        z_sample = np.array(
            [0.35, 0.625, 0.875, 1.125, 1.375, 1.75, 2.25, 2.75])
        Mref = np.array([10.78, 10.70, 10.66, 10.54,
                        10.61, 10.74, 10.69, 10.74])
        a1 = np.array([-0.98, -0.39, -0.37, 0.30, -0.12, 0.04, 1.03, 1.62])
        Phi1 = 10**np.array([-2.54, -2.55, -2.56, -
                            2.72, -2.78, -3.05, -3.80, -4.54])
        a2 = np.array([-1.90, -1.53, -1.61, -1.45, -1.56, -1.49, -1.33, -1.57])
        Phi2 = 10**np.array([-4.29, -3.15, -3.39, -
                            3.17, -3.43, -3.38, -3.26, -3.69])

    def doubleSchecter(Ms, Mref, PhiRef1, alpha1, PhiRef2, alpha2):
        return np.log(10) * np.exp(-10**(Ms - Mref)) * 10**(Ms - Mref) * (PhiRef1 * 10**((Ms - Mref) * alpha1) + PhiRef2 * 10**((Ms - Mref) * alpha2))

    SMF_Cat = np.array([doubleSchecter(Mvals, Mref[iz], Phi1[iz],
                       a1[iz], Phi2[iz], a2[iz]) for iz in range(z_sample.size)])

    for im in range(Mvals.size):
        SMF[:, im] = interp1d(z_sample, np.log10(
            SMF_Cat[:, im]), bounds_error=False, fill_value='extrapolate')(zvals)

    return SMF


def abundance_matching(xvals, xdist, xarr, ydist, yarr, scatter=0.2, fillVal=-66):
    ''' #### abundance_matching ####
    FUNCTION :
        Abundance Matching, Applying equation 37 of Aversa 2015
    INPUT :
        xvals      - The values of the x quantity of the catalogue (e.g. Mhalo / HAR / sHAR) - 1D array (nhalo,)
        xarr       - Array of value corresponding to the values of the x distribution - 1D array (nx,)
        xdist      - x distribution function - 1D array (nx,)
        yarr       - Array of value corresponding to the values of the y distribution - 1D array (ny,)
        ydist      - y distribution function - 1D array (ny,)
        scatter    - scatter in the relation (0.15 for SMHM, 0.3 for HAR-SFR and sHAR-sSFR) - float
        delay      - Bool
        fill_value - fill value for the interpolation
    OUTPUT :
        yvals - The values of the y quantity corresponding to the xvals of the catalogue (e.g. Mstar / SFR / sSFR) - 1D array (nhalo,)
        y_AM  - The values of the abundance matching corresponding to the input xarr - 1D array (nx,)
    '''
    def get_cumDist(dist, vals, scatter=None):
        if scatter is None:
            cumDist = trapezoid(dist, vals) - \
                cumulative_trapezoid(dist, vals, initial=0)
        else:
            cumDist = np.array([trapezoid(
                dist / 2 * erfc((vals[i] - vals) / (np.sqrt(2) * scatter)), vals) for i in range(vals.size)])
        return cumDist

    Cumulative_xdist = np.log10(get_cumDist(xdist, xarr, scatter=scatter))
    Cumulative_ydist = np.log10(get_cumDist(ydist, yarr, scatter=None))
    Cumulative_xdist[np.invert(np.isfinite(Cumulative_xdist))] = -66.
    Cumulative_ydist[np.invert(np.isfinite(Cumulative_ydist))] = -66.

    y_AM = interp1d(np.flip(Cumulative_ydist), np.flip(yarr),
                    bounds_error=False, fill_value=fillVal)(Cumulative_xdist)
    yvals = interp1d(xarr, y_AM, fill_value='extrapolate')(xvals)
    return yvals, y_AM

    plt.savefig('plots/old/SMF.png')


def D_z_white(omegam, z):
    # Constants
    omegal = 1.0 - omegam
    omegak = 0.0

    Ez = (omegal + omegak * (1.0 + z) ** 2.0 +
          omegam * (1.0 + z) ** 3.0) ** 0.5

    omegamz = omegam * (1.0 + z) ** 3.0 / Ez ** 2.0
    omegalz = omegal / Ez ** 2.0

    gz = 2.5 * omegamz * (omegamz ** (4.0 / 7.0) - omegalz +
                          (1.0 + omegamz / 2.0) * (1.0 + omegalz / 70.0)) ** -1.0

    gz0 = 2.5 * omegam * (omegam ** (4.0 / 7.0) - omegal +
                          (1.0 + omegam / 2.0) * (1.0 + omegal / 70.0)) ** -1.0

    dz = (gz / gz0) / (1.0 + z)

    return dz


def fitvartreu(Mlog, omegam, h):
    # Constants-must be equal to those in wpr and HMF!
    hubble = h
    omega_matter = omegam
    omega_baryon = 0.044
    sigma_8 = 0.80
    spectral_index = 1.0

    Mh = np.linspace(5, 17, 100)  # Create an array of halo masses in log10
    log_m_halo = Mh
    gammam = omega_matter * hubble * \
        np.exp(-omega_baryon * (1.0 + np.sqrt(2.0 * hubble) / omega_matter))

    c = 3.804e-4
    x = (c * gammam * (10 ** (log_m_halo / 3.0))) / \
        (omega_matter ** (1.0 / 3.0))
    g1 = 64.087 * ((1.0 + 1.074 * (x ** 0.3) - 1.581 * (x ** 0.4) +
                   0.954 * (x ** 0.5) - 0.185 * (x ** 0.6)) ** -10.0)
    x = (32.0 * gammam)
    g2 = 64.087 * ((1.0 + 1.074 * (x ** 0.3) - 1.581 * (x ** 0.4) +
                   0.954 * (x ** 0.5) - 0.185 * (x ** 0.6)) ** -10.0)
    f = (g1 * g1) / (g2 * g2)
    s = np.sqrt(f * sigma_8 * sigma_8)
    sig = s * (10 ** ((log_m_halo - 14.09) * (1.00 - spectral_index) / 9.2)
               ) / (1.0 + (1.00 - spectral_index) / 9.2)

    res = np.interp(Mlog, Mh, sig)
    return res


def biastreu(omegam, Mhb, zp, h, Shen=False, Tinkbias=False):
    # need to convert back Mhalo into Msun/h for the variance
    Mh1 = Mhb + np.log10(h)
    sig = fitvartreu(Mh1, omegam, h)

    omega_mzp = omegam * (1.0 + zp) ** 3.0 / \
        (1.0 - omegam + omegam * (1.0 + zp) ** 3.0)
    delta_cp = (3.0 / 20.0) * (12.0 * np.pi) ** (2.0 / 3.0) * \
        (1.0 + 0.0123 * np.log10(omega_mzp))
    Dz = D_z_white(omegam, zp)
    delta_czp = delta_cp / Dz

    if Tinkbias:
        a = 0.707
        p = 0.35
        c = 0.8
        nu = delta_czp / sig
        nup = np.sqrt(a) * nu
        frac = nup ** (2.0 * c) / np.sqrt(a) / \
            (nup ** (2.0 * c) + p * (1.0 - c) * (1.0 - c / 2.0))
        pare = (nup ** 2.0 + p * nup ** (2.0 * (1.0 - c)) - frac)
        b = 1.0 + 1.0 / delta_cp * pare

    if Shen:
        neff = -3.0 - 6.0 * np.gradient(np.log10(sig), Mh1)
        b = (1.0 + 1.0 / delta_cp * (delta_czp ** 2.0 / sig ** 2.0 - 1.0)) * \
            (sig ** 4.0 / 2.0 / delta_czp ** 4.0 + 1.0) ** (0.06 - 0.02 * neff)

    beff = b
    return beff


def wpr(logRp, logR, xir):
    # Projected correlation function

    nr = len(logRp)
    wp = np.zeros(nr)

    rmax = np.log10(50.)

    logs = np.linspace(np.min(logR), rmax, 100)
    ss = 10.**logs
    ss2 = ss**2

    Rp = 10.**logRp
    Rp2 = Rp**2

    for k in range(nr):
        logrx = np.log10(np.sqrt(ss2 + Rp2[k]))
        interpol_func = interp1d(logR, np.log10(xir), fill_value="extrapolate")
        xiv = 10.**interpol_func(logrx)
        num = 2. * xiv * ss * np.log(10.)
        wp[k] = simps(num, logs)

    return wp


if __name__ == '__main__':
    if platform == "linux" or platform == "linux2":
        matplotlib.use("gtk4agg")
    # Redshift of observation
    z = np.array([0])
    # Get the HMF
    MhaloArray = np.linspace(11, 16, 251)
    HMF = get_HMF(z, MhaloArray)[0]  # The returned HMF is logged
    # Get the SMF
    MstarArray = np.linspace(8, 13, 276)
    SMF = get_SMF(z, MstarArray)[0]  # The returned SMF is logged

    file_path = 'data/TNG50_1_Dark_Centrals.hdf5'
    with h5py.File(file_path, 'r') as f:
        MhaloCatalogue = np.array(f['Centrals']['M'])
        galaxy_positions = np.array(f['Centrals']['pos']) / 1e3  # cMpc/h

    # Filter both MhaloCatalogue and galaxy_positions to ensure matching dimensions
    valid_indices = MhaloCatalogue >= 9
    MhaloCatalogue = MhaloCatalogue[valid_indices]
    galaxy_positions = galaxy_positions[valid_indices]

    nhalo = MhaloCatalogue.size
    vol = 50**3  # cMpc^3

    # Abundance match between the HMF and SMF to get the SMHM relation and the Stellar mass Catalogue
    MstarCatalogue, Mstar_AbundanceMatched = abundance_matching(
        MhaloCatalogue, 10**HMF, MhaloArray, 10**SMF, MstarArray, scatter=0.15, fillVal='extrapolate')
    # Add the scatter in the relation
    MstarCatalogue += np.random.normal(0, 0.15, nhalo)

    # Plot a simple histogram of halo masses
    plt.figure(figsize=(10, 6))
    plt.hist(MhaloCatalogue, bins=50, color='purple', alpha=0.7, range=(9, 14))
    plt.xlabel(r'$\log_{10}(M_{\mathrm{Halo}}/\mathrm{M}_{\odot})$')
    plt.ylabel('Number of Halos')
    plt.title('Histogram of Halo Masses')
    plt.grid(True)
    plt.savefig('plots/old/HaloMasses.png')

    # Plot a histogram of matched galaxy masses
    plt.figure(figsize=(10, 6))
    plt.hist(MstarCatalogue, bins=50, color='orange', alpha=0.7)
    plt.xlabel(r'$\log_{10}(M_{\mathrm{Star}}/\mathrm{M}_{\odot})$')
    plt.ylabel('Number of Galaxies')
    plt.title('Histogram of Matched Galaxy Masses')
    plt.grid(True)

    # generate the halo mass function of central haloes (no subhaloes)
    # msun/h
    mina = 11.
    maxa = 15.1
    binsize = 0.1
    z = 0.
    Vol = 500.**3.  # Mpc/h
    M = 10**np.arange(mina, maxa, binsize)
    mfunc = mass_function.massFunction(
        M, z, mdef='200m', model='tinker08', q_out='dndlnM')*np.log(10.)
    # plt.plot(np.log10(M), np.log10(mfunc))

    # convert to Msun units
    # cosmo = cosmology.getCurrent()
    # compute the cumulative halo mass function and extract the haloes
    a = mfunc[::-1]
    a = np.cumsum(a*binsize)
    a = a[::-1]
    vecint = np.arange(int(np.max(a*Vol)))
    b = a[::-1]
    cumh = np.log10(M[::-1])
    Mh = np.interp(vecint, b*Vol, cumh)  # Msun/h
    # plt.plot(np.log10(M), np.log10(a*Vol))

    # convert to Msun units
    # cosmo = cosmology.getCurrent()
    h = 0.7
    Mhalo = Mh-np.log10(h)

    import ScalingRelations as Sca

    # assign galaxies to haloes
    Mgal = Sca.Grylls19(z, Mhalo, 0.1)  # Mhalo must be in Msun

    omegam = 0.3
    # bias=biastreu(omegam, Mh, z, h, Shen=True)# Tinkbias=True)

    # mask=(Mgal > 10.1) & (Mgal <10.3)
    mask = (Mgal > 11.1) & (Mgal < 11.3)
    # bb=bias[mask]

    # data_file = "D:\progPython\wpLINz0.0.txt"
    # rp, wp= np.loadtxt(data_file,\
    #                             unpack=True, \
    #                             delimiter=None,\
    #                             dtype=float)

    logR = np.linspace(-2.0, 2.0, 100)
    R = 10.**logR

    cosmo = cosmology.getCurrent()
    params = {'flat': True, 'H0': 70., 'Om0': 0.30,
              'Ob0': 0.044, 'sigma8': 0.80, 'ns': 1.00}
    cosmo1 = cosmology.setCosmology('myCosmo', **params)

    corr = cosmo1.correlationFunction(R, z=z)

    logRp = np.linspace(0., 1.4, 30)
    wpc = wpr(logRp, logR, corr)
    # plt.plot(np.log10(rp),np.log10(wpc),marker='d')
    # plt.show()

    Dz = D_z_white(omegam, z)

    M = 10.**Mh
    nu = peaks.peakHeight(M, z)
    biass = bias.haloBiasFromNu(nu, model='sheth01')
    biast = bias.haloBias(M, model='tinker10', z=z, mdef='vir')

    bb = biass[mask]

    wpr = Dz**2*np.mean(bb)**2*wpc

    plt.figure()
    plt.xlabel(r"log r [Mpc/h]")
    plt.ylabel(r"log wp(r) [Mpc/h]")
    plt.title("Projected Correlation Function vs Seperation ")
    plt.plot(logRp, np.log10(wpr))

    # compare with data:
    # data_file = "D:\progPython\DataClusteringSDSScentrals10.19.txt"
    data_file = "data/DataClusteringSDSScentrals11.12.txt"
    rpd, wpd = np.loadtxt(data_file,
                          unpack=True,
                          delimiter=None,
                          dtype=float)
    plt.plot(np.log10(rpd), np.log10(wpd/rpd), marker="d")
    plt.savefig('plots/old/ProjectedCorrelationFunction.png')
