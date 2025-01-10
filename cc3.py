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
from scipy.interpolate import interp1d
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from colossus.lss import peaks
from colossus.lss import bias
from scipy.integrate import simpson as simps
from cc2 import get_HMF, get_SMF
# plt.rcParams.update({'font.size': 18})

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


if __name__ == '__main__':
    if platform == "linux" or platform == "linux2":
        matplotlib.use("gtk4agg")
    # Load data
    file_path = 'data/TNG50_1_Dark_Centrals.hdf5'  # Update this path if necessary
    with h5py.File(file_path, 'r') as f:
        MhaloCatalogue = np.array(f['Centrals']['M'])
        galaxy_positions = np.array(
            f['Centrals']['pos']) / 1e3  # Convert to cMpc/h

    # Filter both MhaloCatalogue and galaxy_positions to ensure matching dimensions
    valid_indices = MhaloCatalogue >= 9
    MhaloCatalogue = MhaloCatalogue[valid_indices]
    galaxy_positions = galaxy_positions[valid_indices]

    # Define mass arrays
    MhaloArray = np.linspace(11, 16, 251)
    MstarArray = np.linspace(8, 13, 276)

    # Compute HMF and SMF
    HMF = get_HMF(np.array([0]), MhaloArray)[0]  # Redshift z = 0
    SMF = get_SMF(np.array([0]), MstarArray)[0]

    # Perform abundance matching
    MstarCatalogue = abundance_matching2(
        MhaloCatalogue, MhaloArray, 10**HMF, MstarArray, 10**SMF, scatter=0.2)

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
        mass_min = round(mass_min_unrounded, 2)
        mass_max = round(mass_max_unrounded, 2)
        mask = (MstarCatalogue >= mass_min) & (MstarCatalogue <= mass_max)
        logRp = np.linspace(0, 1.4, 30)

        # Compute bias
        try:
            M = 10.**MhaloCatalogue[mask]
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
