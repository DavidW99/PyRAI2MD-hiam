######################################################
#
# PyRAI2MD 2 module for trajectory surface hopping helper
#
# Author Jingbai Li
# Sep 7 2021
#
######################################################

import numpy as np

def avoid_singularity(energy_i, energy_j, state_i, state_j):
    """ Fixing singularity of state energy gap

        Parameters:          Type:
            energy_i         float	 energy of state i
            energy_j         float       energy of state j                 
            state_i          int         state i
            state_j          int         state j

        Return:              Type:
            diff             float	 energy different between i and j

    """

    cutoff = 1e-16
    gap = np.abs(energy_i - energy_j)

    if state_i < state_j:
        sign = -1.0
    else:
        sign = 1.0

    if energy_i == energy_j:
        diff = sign * cutoff
    elif energy_i != energy_j and gap < cutoff:
        diff = sign * cutoff
    elif energy_i != energy_j and gap >= cutoff:
        diff = sign * gap
    else:
        diff = cutoff

    return diff


def reflect_velo(velo, nac, reflect):
    """ Reflecting velocity at frustrated hopping

        Parameters:          Type:
            velo             ndarray	 nuclear velocity
            nac              ndarray     nonadiabatic coupling
            reflect          int         velocity reflection option

        Return:              Type:
            velo             ndarray     nuclear velocity

    """

    if reflect == 1:
        new_velo = -velo
    elif reflect == 2:
        new_velo = velo - 2 * np.sum(velo * nac) / np.sum(nac * nac) * nac

    return new_velo


def adjust_velo(energy_old, energy_new, velo, mass, nac, adjust, reflect):
    """ Adjusting velocity at surface hopping

        Parameters:          Type:
            energy_old       float       energy of old state
            energy_new       float       energy of new state 
            velo             ndarray     nuclear velocity
            nac              ndarray     nonadiabatic coupling
            adjust           int         velocity adjustment option
            reflect          int         velocity reflection option

        Return:              Type:
            velo             ndarray     nuclear velocity
            frustrated       int         surface hopping decision

    """

    kinetic = np.sum(0.5 * mass * velo ** 2)
    frustrated = 0

    if adjust == 0:
        del_kinetic = energy_old - energy_new + kinetic

        if del_kinetic < 0:
            new_velo = reflect_velo(velo, nac, reflect)
            frustrated = 1
        else:
            new_velo = velo

    elif adjust == 1:
        del_kinetic = energy_old - energy_new + kinetic
        if del_kinetic >= 0:
            f = (del_kinetic / kinetic) ** 0.5
            new_velo = velo * f
        else:
            new_velo = reflect_velo(velo, nac, reflect)
            frustrated = 1

    elif adjust == 2:
        a = np.sum(nac * nac / (2 * mass))
        b = np.sum(velo * nac)
        del_kinetic = energy_old - energy_new
        del_kinetic = 4 * a * del_kinetic + b ** 2
        if del_kinetic >= 0:
            if b < 0:
                f = (b + del_kinetic ** 0.5) / (2 * a)
            else:
                f = (b - del_kinetic ** 0.5) / (2 * a)
            new_velo = velo - f * nac / mass
        else:
            new_velo = reflect_velo(velo, nac, reflect)
            frustrated = 1
    else:
        new_velo = velo

    return new_velo, frustrated
