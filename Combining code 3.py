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


def abundance_matching2(xvals, xarr, xdist, yarr, ydist, scatter=0.2):
    """Abundance Matching using CDFs."""
    cum_xdist = np.cumsum(xdist[::-1])[::-1]
    cum_ydist = np.cumsum(ydist[::-1])[::-1]

    y_matched = np.interp(
        np.interp(xvals, xarr, cum_xdist), cum_ydist, yarr[::-1])
    scatter_array = np.random.normal(0, scatter, len(xvals))
    return y_matched + scatter_array


def plot_histograms(halo_masses, galaxy_masses):
    """Plot histograms of halo and galaxy masses."""
    plt.figure(figsize=(10, 6))
    plt.hist(halo_masses, bins=50, alpha=0.7, label='Halo Mass')
    plt.xlabel('log(M_halo) [M_sun]')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('plots/halo_masses.png')

    plt.figure(figsize=(10, 6))
    plt.hist(galaxy_masses, bins=50, alpha=0.7,
             label='Galaxy Mass')
    plt.xlabel('log(M_star) [M_sun]')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('plots/galaxy_masses.png')


def compute_2pcf(logRp, logR, correlation_function, bias_values, Dz):
    """Compute the 2-point correlation function (2PCF)."""
    wp = []
    for Rp in 10**logRp:
        integral = np.trapz(correlation_function /
                            np.sqrt(Rp**2 + 10**(2 * logR)), 10**logR)
        wp.append(2 * integral)
    wp = np.array(wp)
    wp_projected = Dz**2 * np.mean(bias_values)**2 * wp
    return wp_projected


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


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
    # Load data
    file_path = 'data/TNG50_1_Dark_Centrals.hdf5'  # Update this path if necessary
    with h5py.File(file_path, 'r') as f:
        MhaloCatalogue = np.array(f['Centrals']['M'])
        galaxy_positions = np.array(
            f['Centrals']['pos']) / 1e3  # Convert to cMpc/h

    # Define mass arrays
    MhaloArray = np.linspace(11, 16, 251)
    MstarArray = np.linspace(8, 13, 276)

    # Compute HMF and SMF
    HMF = get_HMF(np.array([0]), MhaloArray)[0]  # Redshift z = 0
    SMF = get_SMF(np.array([0]), MstarArray)[0]

    # Perform abundance matching
    MstarCatalogue = abundance_matching2(
        MhaloCatalogue, MhaloArray, HMF, MstarArray, SMF, scatter=0.2)

    # Plot histograms
    plot_histograms(MhaloCatalogue, MstarCatalogue)

    # Compute and plot 2PCF for selected mass bins
    logR = np.linspace(-2, 2, 100)
    R = 10.**logR
    correlation_function = cosmo.correlationFunction(R, z=0)

    bins_for_MhaloCatalogue = 6
    # Define stellar mass bins
    MstarCatalogue.sort()
    split_MstarCatalogue = list(split(MstarCatalogue, bins_for_MhaloCatalogue))
    bins = []
    for i in range(len(split_MstarCatalogue)):
        bins.append((split_MstarCatalogue[i][0], split_MstarCatalogue[i][-1]))

    # bins = [(10.5, 11.0), (11.0, 11.5), (11.5, 12.0)]
    Dz = cosmo.growthFactor(0)

    for i, (mass_min_unrounded, mass_max_unrounded) in enumerate(bins):
        print(mass_min_unrounded, mass_max_unrounded)
        mass_min = round(mass_min_unrounded, 2)
        mass_max = round(mass_max_unrounded, 2)
        mask = (MstarCatalogue >= mass_min) & (MstarCatalogue <= mass_max)
        logRp = np.linspace(0, 1.4, 30)

        # Compute bias
        try:
            M = 10.**MhaloCatalogue[mask]
            M = 10.**split_MstarCatalogue[i]
            nu = peaks.peakHeight(M, 0)
            halo_bias = bias.haloBiasFromNu(nu, model='sheth01')

            # Compute 2PCF
            wp_projected = compute_2pcf(
                logRp, logR, correlation_function, halo_bias, Dz)

            # Plot results
            plt.figure()
            plt.plot(logRp, np.log10(wp_projected),
                     label=f"{mass_min}-{mass_max}")
            plt.xlabel("log(r_p) [Mpc/h]")
            plt.ylabel("log(w_p(r_p)) [Mpc/h]")
            plt.title("2PCF for Galaxy Mass Bins")
            plt.legend()
            plt.savefig(f"plots/2PCF_{mass_min}-{mass_max}.png")
        except Exception as e:
            print(f"Error computing 2PCF for mass bin {
                  mass_min}-{mass_max}: {e}")
