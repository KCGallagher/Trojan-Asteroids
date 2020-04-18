import math
import matplotlib.pyplot as plt
import numpy as np

import orbits

# from wander import wander
from constants import M_P, M_S, ORBIT_NUM, PRECISION, G, R  # User defined constants
from constants import (
    solar_rad,
    planet_rad,
    omega,
    time_span,
    lagrange,
)  # Derived constants

# Initial Conditions
initial_cond_rot = np.array((lagrange[0], lagrange[1], 0, 0, 0, 0))  # in rotating frame
initial_cond_stat = np.array(
    (lagrange[0], lagrange[1], 0, -omega * lagrange[1], omega * lagrange[0], 0)
)

greek_theta = np.arctan((R * math.sqrt(3) / 2) / (R / 2 - solar_rad))
cos = math.cos(greek_theta)
sin = math.sin(greek_theta)


def wander(pertubation, samples=1, pertubation_type="position"):
    """Returns maximum wander over orbit in the rotating frame for given initial point

    Wander is the maximum deviation from the initial point (not the lagrange point) over this timespan
    position gives the spatial pertubation from the lagrange point
    Samples denotes the number of random pertubations sampled within that point for gives perturbation size
    """
    if pertubation_type == "position":
        initial_cond = initial_cond_rot + np.array(
            (pertubation[0], pertubation[1], 0, 0, 0, 0)
        )  # add pertubation
    else:
        initial_cond = initial_cond_rot + np.array(
            (0, 0, 0, pertubation[0], pertubation[1], 0)
        )  # add pertubation

    orbit = orbits.rotating_frame(initial_cond)
    wander_t = np.zeros((len(orbit.t)))
    for i in range(len(orbit.t)):
        wander_t[i] = np.linalg.norm(
            orbit.y[0:3, i]
            - initial_cond[0:3]
            # - lagrange[0:3]
        )  # deviation in pos only
    return np.max(wander_t)
