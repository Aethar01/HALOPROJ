import numpy as np


def Grylls19(z, Mh, sigm):
    # sigm=0.15
    M10, N10, beta10, gamma10 = 11.95, 0.032, 1.61, 0.54
    M11, N11, beta11, gamma11 = 0.4, -0.02, -0.6, -0.1

    N = N10 + N11*(z/(z+1.))
    beta = beta10 + beta11*(z/(z+1.))
    gamma = gamma10 + gamma11*(z/(z+1.))
    M1 = M10 + M11*(z/(z+1.))

    # Mhv=Mh
    Mss = np.log10(2.)+np.log10(N)+Mh-np.log10(10.**(-beta*(Mh-M1)) +
                                               10.**(gamma*(Mh-M1)))+np.random.normal(0., sigm, len(Mh))

    return Mss


def Shankar17(Mh, sigm):
    # sigm=0.15
    Ms0, Mh0, alpha, beta = 10.68, 11.80, 2.13, 1.68  # SerExp
    # Ms0, Mh0, alpha, beta = 10.63, 11.80, 2.17, 1.77 #deVauc
    Mss = Ms0+alpha*np.log10(10.**(Mh-Mh0))-np.log10(1.+10. **
                                                     (beta*(Mh-Mh0)))+np.random.normal(0., sigm, len(Mh))

    return Mss


def AssignBHMass_EQ3(M_star):
    logBHMass = 8.54 + 1.18 * (M_star - 11)
    scatter = np.random.normal(0, 0.3, len(M_star))
    logBHMass += scatter
    return logBHMass


def AssignBHMass_EQ4(M_star):
    logBHMass = 8.35 + 1.31 * (M_star - 11)
    scatter = np.random.normal(0, 0.4, len(M_star))
    logBHMass += scatter
    return logBHMass


def AssignBHMass_EQ5(M_star):
    logBHMass = 7.547 + 1.946 * \
        (M_star - 11) - 0.306 * (M_star - 11)**2 - 0.011 * (M_star - 11)**3
    delta = 0.32 - 0.1 * (M_star - 12)
    scatter = np.random.normal(0, delta, len(M_star))
    logBHMass += scatter
    return logBHMass
