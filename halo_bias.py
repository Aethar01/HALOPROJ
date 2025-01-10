# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:02:15 2024

@author: fs1y12
"""

from colossus.cosmology import cosmology
from scipy.interpolate import interp1d
from colossus.lss import bias
from colossus.lss import peaks
import ScalingRelations as Sca
from colossus.lss import mass_function
from scipy.integrate import simps
import numpy as np
from matplotlib import pyplot as plt


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


cosmology.setCosmology('planck15')

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
plt.plot(logRp, np.log10(wpr))

# compare with data:
# data_file = "D:\progPython\DataClusteringSDSScentrals10.19.txt"
data_file = "E:/University/Y3/BSc Project - Galaxy Clusters/Intro/Coding/DataClusteringSDSScentrals11.12.txt"
rpd, wpd = np.loadtxt(data_file,
                      unpack=True,
                      delimiter=None,
                      dtype=float)
plt.plot(np.log10(rpd), np.log10(wpd/rpd), marker="d")
plt.show()
