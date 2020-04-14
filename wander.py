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
initial_cond_rot = np.array((lagrange[0], lagrange[1], 0, 0, 0, 0))  # in rotating frame
initial_cond_stat = np.array(
    (lagrange[0], lagrange[1], 0, -omega * lagrange[1], omega * lagrange[0], 0)
)

greek_theta = np.arctan((R * math.sqrt(3) / 2) / (R / 2 - solar_rad))


# Defined Functions
def perturb(initial_cond, max_pertubation_size=0.01):
    """ Returns perturbed version of initial conditions array

    init_cond is the initial conditions, submitted as a numpy array
    pertubation_size is the relative magnitude of the perturbation, measured as 
    the percentage deivation from the lagrange point (for position)
    Note that this will give a different value each time it is run
    """
    # return data + data * (
    #     max_pertubation_size * np.random.uniform(-1, 1, (np.shape(data)))
    # )  # random between -1 and 1, perturbs position only for initial_cond_rot
    # theta = np.random.uniform(0, 2 * math.pi)
    # return data + data * (
    #     max_pertubation_size
    #     * np.random.rand()
    #     * np.array((math.cos(theta), math.sin(theta), 0, 0, 0, 0))
    # )  # random betweeon -1 and 1, perturbs position only for initial_cond_rot
    rand_array = np.random.uniform(-1, 1, (np.shape(initial_cond)))
    while np.linalg.norm(rand_array[0:2]) > 1:
        rand_array = np.random.uniform(-1, 1, (np.shape(initial_cond)))
    return initial_cond + initial_cond * (
        max_pertubation_size * rand_array
    )  # random between -1 and 1, selects points in circle of radius 1


def max_wander(max_pertubation_size, samples):
    """Returns maximum deviation over orbit in the rotating frame for given number of samples

    This calculates the distange from the initial point (not the lagrange point)
    It returns a samples*3 array, giving the radial and tangential components
    of the pertubation, and then the maximum wander over this timespan

    pertubation_size determies the relative magnitude of the perturbation as in perturb()
    Samples denotes the number of random pertubations sampled for gives perturbation size
    """
    output = np.zeros((samples, 3))  # size of pertubation; size of wander
    for n in range(samples):
        initial_cond = perturb(initial_cond_rot, max_pertubation_size)
        orbit = orbits.rotating_frame(initial_cond)
        sample_wander = np.zeros((len(orbit.t)))

        perturb_theta = np.arctan(
            (initial_cond[1] - initial_cond_rot[1])
            / (initial_cond[0] - initial_cond_rot[0])
        )

        for i in range(len(sample_wander)):
            sample_wander[i] = np.linalg.norm(
                orbit.y[0:3, i]
                - initial_cond[0:3]
                # - lagrange[0:3]
            )  # deviation in pos only
        output[n, 0] = np.linalg.norm((initial_cond[0:3] - lagrange[0:3])) * np.abs(
            np.cos(greek_theta - perturb_theta)
        )  # radial component of pertubation
        output[n, 1] = np.linalg.norm((initial_cond[0:3] - lagrange[0:3])) * np.abs(
            np.sin(greek_theta - perturb_theta)
        )  # tangential component of pertubation
        output[n, 2] = np.max(sample_wander)  # size of wander
    return output


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


wander_data = max_wander(max_pertubation_size=0.001, samples=200)

fig = plt.figure()
ax = fig.add_subplot()

coeff = np.polyfit(wander_data[:, 0], wander_data[:, 2], 1)
plt.plot(
    [wander_data[:, 0].min(), wander_data[:, 0].max()],
    [
        wander_data[:, 0].min() * coeff[0] + coeff[1],
        wander_data[:, 0].max() * coeff[0] + coeff[1],
    ],
    linewidth=0.75,
)
plt.plot(
    wander_data[:, 0], wander_data[:, 2], linestyle="None", marker="+", color="navy"
)

plt.title("Wander against Pertubation size")
plt.ylabel("Wander /AU")
plt.xlabel("Radial Pertubation Size /AU")
plt.text(
    0.02,
    0.95,
    "Best fit coefficient: " + str("{0:.2f}".format(coeff[0])),
    transform=ax.transAxes,
    fontsize=9.5,
)
plt.text(
    0.02,
    0.9,
    "Pearson correlation coefficient: "
    + str("{0:.6f}".format(np.corrcoef(wander_data[:, 0], wander_data[:, 2])[0, 1])),
    transform=ax.transAxes,
    fontsize=9.5,
)
plt.savefig("wanderagainstradialpertubation.png")
plt.show()


plt.plot(wander_data[:, 1], wander_data[:, 2], linestyle="None", marker="x")
plt.title("Wander against Pertubation size")
plt.ylabel("Wander /AU")
plt.xlabel("Tangential Pertubation Size /AU")
# plt.text(
#     0.02,
#     0.95,
#     "Pearson correlation coefficient: "
#     + str("{0:.6f}".format(np.corrcoef(wander_data[:, 1], wander_data[:, 2])[0, 1])),
#     transform=ax.transAxes,
#     fontsize=9.5,
# )
plt.savefig("wanderagainsttangentialpertubation.png")
plt.show()


# for i in range(3):
#     plt.plot(perturb(initial_cond_rot)[0], perturb(initial_cond_rot)[1], marker="o")
# plt.show()

# then run and compare max wander (with different definitions??)
# 2d colour meshes over position and velocity phase space to give max deviation from lagrange point??
#
# need other way to define stability as other stable orbits away from lagrange point may form


# # START OF MESH GRID PLOTS

# grid_size = 0.04
# sampling_points = 30
# # make these smaller to increase the resolution
# dx, dy = grid_size / sampling_points, grid_size / sampling_points

# # generate 2 2d grids for the x & y bounds
# y, x = np.mgrid[
#     slice(-grid_size, grid_size + dy, dy), slice(-grid_size, grid_size + dx, dx)
# ]

# import time

# z = np.zeros_like(x)
# for i, j in np.ndindex(z.shape):
#     start = time.time()
#     z[i, j] = wander((x[i, j], y[i, j]), pertubation_type="velocity")
#     end = time.time()
#     print(str((i, j)) + " in time " + str(end - start) + " s")


# # x and y are bounds, so z should be the value *inside* those bounds.
# # Therefore, remove the last value from the z array.
# z = z[:-1, :-1]

# # pick the desired colormap
# # cmap = plt.get_cmap("PiYG")

# fig = plt.figure()
# ax0 = fig.add_subplot()

# im = ax0.pcolormesh(x, y, z, label="x")  # cmap=cmap)  # , norm=norm)
# cbar = fig.colorbar(im)
# cbar.ax.set_ylabel("Wander /AU")

# ax0.set_title("Wander from starting point in the rotating frame")
# ax0.set_xlabel("Deviation from Lagrange velocity in x direction (AU/year)")
# ax0.set_ylabel("Deviation from Lagrange velocity in y direction (AU/year)")

# plt.savefig("testvelocitymesh.png")
# plt.show()

# import matplotlib
# from matplotlib.colors import BoundaryNorm
# from matplotlib.ticker import MaxNLocator

# fig = plt.figure()
# ax1 = fig.add_subplot()

# levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())

# # pick the desired colormap, sensible levels, and define a normalization
# # instance which takes data values and translates those into levels.#

# cmap = plt.get_cmap("PiYG")
# norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

# # contours are *point* based plots, so convert our bound into point
# # centers
# cf = ax1.contourf(
#     x[:-1, :-1] + dx / 2.0, y[:-1, :-1] + dy / 2.0, z, levels=levels, cmap=cmap
# )

# cbar = fig.colorbar(cf, ax=ax1)
# cbar.ax.set_ylabel("Wander /AU")
# ax1.set_title("Wander from Starting Point in the Rotating Frame")
# ax1.set_xlabel("Deviation from Lagrange velocity in x direction (AU/year)")
# ax1.set_ylabel("Deviation from Lagrange velocity in y direction (AU/year)")
# plt.savefig("testvelocitymeshb.png")
# plt.show()


# STATIONARY FRAME

# for i in range(2):
#     orbit_sol2 = orbits.stationary_frame(perturb(initial_cond_stat, 0.3))
#     plt.plot(
#         orbit_sol2.y[0, :], orbit_sol2.y[1, :], linewidth=1, linestyle="dotted",
#     )

# plt.plot(0, 0, label="CoM", color="black", marker="x", linestyle="None")
# plt.plot(
#     initial_cond_stat[0],
#     initial_cond_stat[1],
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
