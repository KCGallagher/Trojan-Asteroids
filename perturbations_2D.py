import math
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

import orbits
import wander
from constants import R, PRECISION  # User defined constants
from constants import (
    solar_rad,
    omega,
    period,
    time_span,
    lagrange,
)  # Derived constants

# INITIAL CONDITIONS
initial_cond_rot = np.array((lagrange[0], lagrange[1], 0, 0, 0, 0))  # in rotating frame
initial_cond_stat = np.array(
    (lagrange[0], lagrange[1], 0, -omega * lagrange[1], omega * lagrange[0], 0)
)

greek_theta = np.arctan((R * math.sqrt(3) / 2) / (R / 2 - solar_rad))
cos = math.cos(greek_theta)
sin = math.sin(greek_theta)

# RANDOM SAMPLING OF WANDER

wander_data = wander.rand_sample(max_pertubation_size=0.01, samples=20)

fig = plt.figure()
ax = fig.add_subplot()

coeff = np.polyfit(
    wander_data[:, 0], wander_data[:, 2], 2
)  # for quadratic line of best fit
best_fit = np.poly1d(coeff)
order = np.argsort(wander_data[:, 0])
plt.plot(
    wander_data[order, 0], best_fit(wander_data[order, 0]), linewidth=0.75,
)
plt.plot(
    wander_data[:, 0], wander_data[:, 2], linestyle="None", marker="+", color="navy"
)

plt.title("Wander under Radial Pertubations")
plt.ylabel("Wander /AU")
plt.xlabel("Radial Pertubation Size /AU")
plt.text(
    0.02,
    0.95,
    "Line of Best fit: "
    + str("{0:.2f}".format(coeff[0]))
    + "x\u00b2 "
    + str("{0:+.2f}".format(coeff[1]))
    + "x "
    + str("{0:+.2f}".format(coeff[2])),
    transform=ax.transAxes,
    fontsize=9.5,
)
plt.savefig("wanderagainstradialpertubation2.png")
plt.show()


plt.plot(
    wander_data[:, 1], wander_data[:, 2], linestyle="None", marker="x", color="navy"
)
plt.title("Wander under Tangential Pertubations")
plt.ylabel("Wander /AU")
plt.xlabel("Tangential Pertubation Size /AU")
plt.savefig("wanderagainsttangentialpertubation2.png")
plt.show()


# # # START OF MESH GRID PLOTS

grid_size = 0.04
sampling_points = 5
# make these smaller to increase the resolution
dx, dy = grid_size / sampling_points, grid_size / sampling_points

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[
    slice(-grid_size, grid_size + dy, dy), slice(-grid_size, grid_size + dx, dx)
]

import time

z = np.zeros_like(x)
for i, j in np.ndindex(z.shape):
    start = time.time()
    z[i, j] = wander.initial_point((x[i, j], y[i, j], 0), pertubation_type="velocity")
    end = time.time()
    print(str((i, j)) + " in time " + str(end - start) + " s")


# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = z[:-1, :-1]

# pick the desired colormap
# cmap = plt.get_cmap("PiYG")

fig = plt.figure()
ax0 = fig.add_subplot()

im = ax0.pcolormesh(x, y, z)  # label="x")  # cmap=cmap)  # , norm=norm)
cbar = fig.colorbar(im)
cbar.ax.set_ylabel("Wander /AU")

ax0.set_title("Wander from starting point in the rotating frame")
ax0.set_xlabel("Deviation from Lagrange velocity in x direction (AU/year)")
ax0.set_ylabel("Deviation from Lagrange velocity in y direction (AU/year)")

# plt.savefig("testvelocitymesh.png")
plt.show()


# PLOT OF WANDER WITHIN STATIONARY FRAME

for i in range(2):
    orbit_sol = orbits.stationary_frame(wander.perturb(initial_cond_stat, 0.3))
    plt.plot(
        orbit_sol.y[0, :], orbit_sol.y[1, :], linewidth=1, linestyle="dotted",
    )

plt.plot(0, 0, label="CoM", color="black", marker="x", linestyle="None")
plt.plot(
    initial_cond_stat[0],
    initial_cond_stat[1],
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
# plt.savefig("drift5e-1.png")
plt.show()


# WANDER WITHIN ROTATING FRAME
orbit_sol = orbits.rotating_frame(
    (initial_cond_rot + (0, 0, 0, 0, 0, 0))
)  # no perturbation
orbit_sol_rad = orbits.rotating_frame(
    (initial_cond_rot + 0.01 * np.array((cos, sin, 0, 0, 0, 0)))
)  # radial perturbation
orbit_sol_large_rad = orbits.rotating_frame(
    (initial_cond_rot + 0.07 * np.array((cos, sin, 0, 0, 0, 0)))
)  # large radial perturbation
orbit_sol_tan = orbits.rotating_frame(
    (initial_cond_rot + 0.01 * np.array((sin, -cos, 0, 0, 0, 0)))
)  # tangential perturbation

# PLOT OF TADPOLE AND HORSESHOE ORBITS
plt.plot(orbit_sol_large_rad.y[0, :], orbit_sol_large_rad.y[1, :], label="Horseshoe")
plt.plot(orbit_sol_rad.y[0, :], orbit_sol_rad.y[1, :], label="Tadpole")

plt.plot(
    lagrange[0],
    lagrange[1],
    marker="+",
    markersize=8,
    color="black",
    linestyle="",
    label="Lagrange",
)

plt.plot(
    lagrange[0], -lagrange[1], marker="+", markersize=8, color="black", linestyle="",
)

plt.plot(
    orbits.solar_pos(0)[0],
    0,
    label="Sun",
    color="yellow",
    markersize=18,
    marker="o",
    linestyle="None",
)
plt.plot(
    orbits.planet_pos(0)[0],
    0,
    label="Jupiter",
    color="red",
    markersize=9,
    marker="o",
    linestyle="None",
)
plt.title("Tadpole and Horseshoe Orbits in the Rotating Frame")
plt.xlabel("X Position /AU")
plt.ylabel("Y Position /AU")
plt.legend(bbox_to_anchor=(0, 0, 1, 0.2), loc="lower left", mode="expand", ncol=5)
plt.savefig("horseshoe.png")
plt.show()


# FOURIER ANALYSIS OF POLAR ANGLE
# Consider polar angle from the centre of mass

angle = np.zeros_like(time_span)
for i in range(len(time_span)):
    angle[i] = np.arctan(orbit_sol_rad.y[1, i] / orbit_sol_rad.y[0, i])
plt.plot(time_span, angle)
plt.ticklabel_format(axis="y", style="", scilimits=None)
plt.ylabel("Polar Angle from CoM /rad")
plt.xlabel("Time /years")
plt.title("Asteroid Deviation from Lagrange point")
plt.show()

sampling_rate = PRECISION / period
freqs = np.fft.fftfreq(len(angle)) * sampling_rate
fourier = np.abs(np.fft.fft(angle))

plt.plot(freqs, fourier)
plt.title("Fourier Transform of Angle")
plt.ylabel("'Frequency Domain (Spectrum) Magnitude'")
plt.xlabel("Frequency (1/year)")
plt.xlim(0, sampling_rate / 500)  # select low freq region of data
plt.ylim(0, 1000)  # Gives rough view of relevant peaks
plt.show()

peaks, properties = scipy.signal.find_peaks(fourier, prominence=(300,), width=(0,))

peak_freq = np.zeros(len(peaks))
for i in range(len(peaks)):
    peak_freq[i] = 1 / freqs[peaks[i]]
print("Period (in years) of primary frequency components in Fourier spectrum: ")
print(np.unique(np.abs(peak_freq)).round(2))
# Evaluated for 1000 points per orbit, 100 orbits

# DEVIATION FROM LAGRANGE POINT

plt.rcParams.update({"font.size": 15})  # larger font for these plots
plt.xticks([2.585, 2.59, 2.595, 2.60, 2.605])
plt.xlim([2.585, 2.605])
plt.plot(orbit_sol_tan.y[0, :], orbit_sol_tan.y[1, :], label="Greeks")
plt.plot(
    initial_cond_rot[0] + 0.01 * sin,
    initial_cond_rot[1] - 0.01 * cos,
    marker="o",
    markersize=4,
    color="purple",
    linestyle="",
    label="Initial Point",
)
plt.plot(
    lagrange[0],
    lagrange[1],
    marker="+",
    markersize=8,
    color="gray",
    linestyle="",
    label="Lagrange Point",
)

# plt.title("Orbit under Tangential Perturbation in the Rotating Frame")
plt.xlabel("X Position /AU")
plt.ylabel("Y Position /AU")
plt.savefig("tangentialp_orbits.png")
plt.show()

plt.plot(orbit_sol_rad.y[0, :], orbit_sol_rad.y[1, :], label="Greeks")
plt.plot(
    initial_cond_rot[0] + 0.01 * cos,
    initial_cond_rot[1] + 0.01 * sin,
    marker="o",
    markersize=4,
    color="purple",
    linestyle="",
    label="Initial Point",
)
plt.plot(
    lagrange[0],
    lagrange[1],
    marker="+",
    markersize=8,
    color="gray",
    linestyle="",
    label="Lagrange Point",
)

# plt.title("Orbit under Radial Perturbation in the Rotating Frame")
plt.xlabel("X Position /AU")
plt.ylabel("Y Position /AU")
plt.legend()
plt.savefig("radialp_orbits.png")
plt.show()
