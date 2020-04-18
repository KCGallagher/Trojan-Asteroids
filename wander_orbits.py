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


# ROTATING FRAME
# orbit_sol = orbits.rotating_frame(
#     (initial_cond_rot + (0, 0, 0, 0, 0, 0))
# )  # no perturbation
# orbit_sol = orbits.rotating_frame(
#     (initial_cond_rot + 0.01 * np.array((cos, sin, 0, 0, 0, 0)))
# )  # radial perturbation GIVES TADPOLES
orbit_sol = orbits.rotating_frame(
    (initial_cond_rot + 0.01 * np.array((sin, -cos, 0, 0, 0, 0)))
)  # tangential perturbation   GIVES ORBITS

# DEVIATION FROM LAGRANGE POINT
wndr = np.zeros_like(time_span)
for i in range(len(time_span)):
    wndr[i] = np.linalg.norm(orbit_sol.y[0:3, i] - lagrange[0:3])
plt.plot(time_span, wndr)
# plt.yscale("log", basey=10)
plt.ticklabel_format(axis="y", style="", scilimits=None)
plt.ylabel("Magnitude of Deviation /AU")
plt.xlabel("Time /years")
plt.title("Asteroid Deviation from Lagrange point")
plt.show()


plt.plot(time_span, orbit_sol.y[0, :] - lagrange[0])
# plt.yscale("log", basey=10)
plt.ticklabel_format(axis="y", style="", scilimits=None)
plt.ylabel("x Deviation /AU")
plt.xlabel("Time /years")
plt.title("Asteroid x Deviation over time")
plt.show()

plt.plot(time_span, orbit_sol.y[1, :] - lagrange[1])
# plt.yscale("log", basey=10)
plt.ticklabel_format(axis="y", style="", scilimits=None)
plt.ylabel("y Deviation /AU")
plt.xlabel("Time /years")
plt.title("Asteroid y Deviation over time")
plt.show()


plt.plot(orbit_sol.y[0, :], orbit_sol.y[1, :], label="Greeks")
plt.title("Rotating Frame just Greeks")
plt.show()
