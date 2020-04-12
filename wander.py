import math
import matplotlib.pyplot as plt
import numpy as np

import orbits
from constants import M_P, M_S, ORBIT_NUM, PRECISION, G, R  # User defined constants
from constants import (
    omega,
    time_span,
    lagrange,
)  # Derived constants


def perturb(data, max_pertubation_size=0.01):
    """ Returns perturbed version of initial conditions array

    data is the initial conditions, submitted as a numpy array
    pertubation_size is the relative magnitude of the perturbation, measured as 
    the percentage deivation from the lagrange point (for position)
    Note that this will give a different value each time it is run
    """
    return data + data * (
        max_pertubation_size * np.random.uniform(-1, 1, (np.shape(data)))
    )  # random betweeon -1 and 1


def max_wander(max_pertubation_size=0.01, samples=3):
    """Returns maximum deviation over orbit in the rotating frame for given number of samples

    This calculates the distange from the initial point (not the lagrange point)
    pertubation_size determies the relative magnitude of the perturbation as in perturb()
    Samples denotes the number of random pertubations sampled for gives perturbation size
    """
    wander = np.zeros((samples, 2))  # size of pertubation; size of wander
    for n in range(samples):
        init_cond = perturb(init_cond_rot, max_pertubation_size)
        orbit = orbits.rotating_frame(init_cond)
        sample_wander = np.zeros((len(orbit.t)))
        for i in range(len(sample_wander)):
            sample_wander[i] = np.linalg.norm(
                orbit.y[0:3, i]
                - init_cond[0:3]
                # orbit.y[0:3, i]
                # - lagrange[0:3]
            )  # deviation in pos only
        wander[n, 0] = np.linalg.norm(
            init_cond[0:3] - lagrange[0:3]
        )  # size of pertubation
        wander[n, 1] = np.max(sample_wander)  # size of wander
    return wander


init_cond_rot = np.array((lagrange[0], lagrange[1], 0, 0, 0, 0))  # in rotating frame
init_cond_stat = np.array(
    (lagrange[0], lagrange[1], 0, -omega * lagrange[1], omega * lagrange[0], 0)
)


wander_data = max_wander(max_pertubation_size=0.001, samples=100)
plt.plot(wander_data[:, 0], wander_data[:, 1], linestyle="None", marker="x")
plt.title("Wander against Pertubation size")
plt.ylabel("Wander")
plt.xlabel("Pertubation Size")
plt.savefig("wanderagainstsmallpertubation.png")
plt.show()


# for i in range(3):
#     plt.plot(perturb(init_cond_rot)[0], perturb(init_cond_rot)[1], marker="o")
# plt.show()

# then run and compare max wander (with different definitions??)
# 2d colour meshes over position and velocity phase space to give max deviation from lagrange point??
#
# need other way to define stability as other stable orbits away from lagrange point may form


# STATIONARY FRAME
for i in range(2):
    orbit_sol2 = orbits.stationary_frame(perturb(init_cond_stat, 0.3))
    plt.plot(
        orbit_sol2.y[0, :], orbit_sol2.y[1, :], linewidth=1, linestyle="dotted",
    )

plt.plot(0, 0, label="CoM", color="black", marker="x", linestyle="None")
plt.plot(
    init_cond_stat[0],
    init_cond_stat[1],
    label="Lagrange Point",
    color="black",
    marker="+",
    linestyle="None",
)
plt.plot(
    orbits.solar_pos(time_span)[0],
    orbits.solar_pos(time_span)[1],
    label="Sun",
    color="yellow",
    # markersize=12,
    # marker="o",
)
plt.plot(
    orbits.planet_pos(time_span)[0],
    orbits.planet_pos(time_span)[1],
    label="Jupiter",
    color="red",
    # markersize=8,
    # marker="o",
)

plt.title("Stationary Frame")
plt.legend()
plt.savefig("drift5e-1.png")
plt.show()
