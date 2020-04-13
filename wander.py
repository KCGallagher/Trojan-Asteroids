import math
import matplotlib.pyplot as plt
import numpy as np

import orbits
from constants import M_P, M_S, ORBIT_NUM, PRECISION, G, R  # User defined constants
from constants import (
    solar_rad,
    omega,
    time_span,
    lagrange,
)  # Derived constants

# Initial Conditions
init_cond_rot = np.array((lagrange[0], lagrange[1], 0, 0, 0, 0))  # in rotating frame
init_cond_stat = np.array(
    (lagrange[0], lagrange[1], 0, -omega * lagrange[1], omega * lagrange[0], 0)
)

greek_theta = np.arctan((R * math.sqrt(3) / 2) / (R / 2 - solar_rad))
cos, sin = np.cos(greek_theta), np.sin(greek_theta)

# Defined Functions
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
    It returns a samples*3 array, giving the x and y components
    of the pertubation, and then the maximum wander over this timespan

    pertubation_size determies the relative magnitude of the perturbation as in perturb()
    Samples denotes the number of random pertubations sampled for gives perturbation size
    """
    wander = np.zeros((samples, 3))  # size of pertubation; size of wander
    for n in range(samples):
        init_cond = perturb(init_cond_rot, max_pertubation_size)
        orbit = orbits.rotating_frame(init_cond)
        sample_wander = np.zeros((len(orbit.t)))
        for i in range(len(sample_wander)):
            sample_wander[i] = np.linalg.norm(
                orbit.y[0:3, i]
                - init_cond[0:3]
                # - lagrange[0:3]
            )  # deviation in pos only
        # wander[n, 0] = np.linalg.norm(
        #     init_cond[0:3] - lagrange[0:3]
        # )  # magnitude of pertubation
        wander[n, 0] = np.linalg.norm((init_cond[0:3] - lagrange[0:3])) * np.cos(
            greek_theta - np.arctan(init_cond[1] / init_cond[0])
        )  # radial component of pertubation
        wander[n, 1] = np.linalg.norm((init_cond[0:3] - lagrange[0:3])) * np.sin(
            greek_theta - np.arctan(init_cond[1] / init_cond[0])
        )  # tangential component of pertubation
        wander[n, 2] = np.max(sample_wander)  # size of wander
    return wander


def wander(position, samples=1):
    """Returns maximum deviation over orbit in the rotating frame for given initial point

    It returns the maximum wander from the initial point (not the lagrange point) over this timespan
    pertubation_size determies the relative magnitude of the perturbation as in perturb()
    Samples denotes the number of random pertubations sampled within that point for gives perturbation size
    """
    init_cond = init_cond_rot
    init_cond[0] += position[0]
    init_cond[1] += position[1]
    orbit = orbits.rotating_frame(init_cond)
    wander_t = np.zeros((len(orbit.t)))
    for i in range(len(orbit.t)):
        wander_t[i] = np.linalg.norm(
            orbit.y[0:3, i]
            - init_cond[0:3]
            # - lagrange[0:3]
        )  # deviation in pos only
    return np.max(wander_t)


# wander_data = max_wander(max_pertubation_size=0.001, samples=100)
# plt.plot(wander_data[:, 0], wander_data[:, 2], linestyle="None", marker="x")
# plt.title("Wander against Pertubation size")
# plt.ylabel("Wander")
# plt.xlabel("Radial Pertubation Size")
# # plt.savefig("wanderagainstradialpertubation.png")
# plt.show()

# plt.plot(wander_data[:, 1], wander_data[:, 2], linestyle="None", marker="x")
# plt.title("Wander against Pertubation size")
# plt.ylabel("Wander")
# plt.xlabel("Tangential Pertubation Size")
# # plt.savefig("wanderagainsttangentialpertubation.png")
# plt.show()


# for i in range(3):
#     plt.plot(perturb(init_cond_rot)[0], perturb(init_cond_rot)[1], marker="o")
# plt.show()

# then run and compare max wander (with different definitions??)
# 2d colour meshes over position and velocity phase space to give max deviation from lagrange point??
#
# need other way to define stability as other stable orbits away from lagrange point may form


import matplotlib
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


grid_size = 0.01
sampling_points = 8
# make these smaller to increase the resolution
dx, dy = grid_size / sampling_points, grid_size / sampling_points

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[
    slice(-grid_size, grid_size + dy, dy), slice(-grid_size, grid_size + dx, dx)
]

print(wander((x[0, 12], y[0, 12])))
print(wander((x[0, 16], y[0, 16])))
print(wander((x[3, 7], y[3, 7])))
print(wander((x[3, 8], y[3, 8])))

import time

# z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
z = np.zeros_like(x)
print(np.shape(z))
for i, j in np.ndindex(z.shape):
    start = time.time()
    z[i, j] = wander((x[i, j], y[i, j]))
    end = time.time()
    print(str((i, j)) + " in time " + str(end - start) + " s")


# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = z[:-1, :-1]
levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())


# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
cmap = plt.get_cmap("PiYG")
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

# fig, (ax0, ax1) = plt.subplots(nrows=2)

fig = plt.figure()
ax0 = fig.add_subplot()

im = ax0.pcolormesh(x, y, z, cmap=cmap, norm=norm)
fig.colorbar(im, ax=ax0)
ax0.set_title("Wander over spatial coordinates in rotating frame")


# # contours are *point* based plots, so convert our bound into point
# # centers
# cf = ax1.contourf(
#     x[:-1, :-1] + dx / 2.0, y[:-1, :-1] + dy / 2.0, z, levels=levels, cmap=cmap
# )
# fig.colorbar(cf, ax=ax1)
# ax1.set_title("contourf with levels")

# adjust spacing between subplots so `ax1` title and `ax0` tick labels
# don't overlap
fig.tight_layout()
plt.savefig("testcolourmesh3.png")
# plt.ticklabel_format(axis="both", style="", scilimits=None)
plt.show()


# STATIONARY FRAME
# for i in range(2):
#     orbit_sol2 = orbits.stationary_frame(perturb(init_cond_stat, 0.3))
#     plt.plot(
#         orbit_sol2.y[0, :], orbit_sol2.y[1, :], linewidth=1, linestyle="dotted",
#     )

# plt.plot(0, 0, label="CoM", color="black", marker="x", linestyle="None")
# plt.plot(
#     init_cond_stat[0],
#     init_cond_stat[1],
#     label="Lagrange Point",
#     color="black",
#     marker="+",
#     linestyle="None",
# )
# plt.plot(
#     orbits.solar_pos(time_span)[0],
#     orbits.solar_pos(time_span)[1],
#     label="Sun",
#     color="yellow",
#     # markersize=12,
#     # marker="o",
# )
# plt.plot(
#     orbits.planet_pos(time_span)[0],
#     orbits.planet_pos(time_span)[1],
#     label="Jupiter",
#     color="red",
#     # markersize=8,
#     # marker="o",
# )

# plt.title("Stationary Frame")
# plt.legend()
# # plt.savefig("drift5e-1.png")
# plt.show()