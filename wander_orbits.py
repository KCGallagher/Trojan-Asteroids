import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

import orbits

# from wander import wander
from constants import M_P, M_S, ORBIT_NUM, PRECISION, G, R  # User defined constants
from constants import (
    solar_rad,
    planet_rad,
    omega,
    period,
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
orbit_sol = orbits.rotating_frame(
    (initial_cond_rot + (0, 0, 0, 0, 0, 0))
)  # no perturbation
orbit_sol_rad = orbits.rotating_frame(
    (initial_cond_rot + 0.01 * np.array((cos, sin, 0, 0, 0, 0)))
)  # radial perturbation GIVES TADPOLES
orbit_sol_tan = orbits.rotating_frame(
    (initial_cond_rot + 0.01 * np.array((sin, -cos, 0, 0, 0, 0)))
)  # tangential perturbation   GIVES ORBITS

orbit_sol_rad2 = orbits.rotating_frame(
    (initial_cond_rot + 0.07 * np.array((cos, sin, 0, 0, 0, 0)))
)  # radial perturbation GIVES TADPOLES


# FOURIER ANALYSIS OF POLAR ANGLE
# angle = np.zeros_like(time_span)
# for i in range(len(time_span)):
#     angle[i] = np.arctan(orbit_sol_rad.y[1, i] / orbit_sol_rad.y[0, i])
# plt.plot(time_span, angle)
# plt.ticklabel_format(axis="y", style="", scilimits=None)
# plt.ylabel("Polar Angle from CoM /rad")
# plt.xlabel("Time /years")
# plt.title("Asteroid Deviation from Lagrange point")
# plt.show()

# sampling_rate = PRECISION / period
# freqs = np.fft.fftfreq(len(angle)) * sampling_rate
# fourier = np.abs(np.fft.fft(angle))

# plt.plot(freqs, fourier)
# plt.title("Fourier Transform of Angle")
# plt.ylabel("'Frequency Domain (Spectrum) Magnitude'")
# plt.xlabel("Frequency (1/year)")
# plt.xlim(0, sampling_rate / 500)  # select low freq region of data
# plt.ylim(0, 1000)
# plt.show()

# peaks, properties = scipy.signal.find_peaks(fourier, prominence=(300,), width=(0,))

# print(properties)
# peak_freq = np.zeros(len(peaks))
# for i in range(len(peaks)):
#     peak_freq[i] = 1 / freqs[peaks[i]]
# print("Period (in years) of primary frequency components in Fourier spectrum: ")
# print(np.abs(np.unique(peak_freq)).round(2))
# # Evaluated for 1000 points per orbit, 100 orbits

# DEVIATION FROM LAGRANGE POINT

# # plt.rcParams.update({"font.size": 15})
# plt.xticks([2.585, 2.59, 2.595, 2.60, 2.605])
# plt.xlim([2.585, 2.605])
# plt.plot(orbit_sol_tan.y[0, :], orbit_sol_tan.y[1, :], label="Greeks")
# plt.plot(
#     initial_cond_rot[0] + 0.01 * sin,
#     initial_cond_rot[1] - 0.01 * cos,
#     marker="o",
#     markersize=4,
#     color="purple",
#     linestyle="",
#     label="Initial Point",
# )
# plt.plot(
#     lagrange[0],
#     lagrange[1],
#     marker="+",
#     markersize=8,
#     color="gray",
#     linestyle="",
#     label="Lagrange Point",
# )

# # plt.title("Orbit under Tangential Perturbation in the Rotating Frame")
# plt.xlabel("X Position /AU")
# plt.ylabel("Y Position /AU")
# # plt.legend()
# plt.savefig("tangentialp_orbits.png")
# plt.show()

plt.plot(orbit_sol_rad2.y[0, :], orbit_sol_rad2.y[1, :], label="Horseshoe")
plt.plot(orbit_sol_rad.y[0, :], orbit_sol_rad.y[1, :], label="Tadpole")
plt.plot(
    orbit_sol_rad2.y[0, -1],
    orbit_sol_rad2.y[1, -1],
    marker="o",
    markersize=10,
    color="pink",
    linestyle="",
)

# plt.plot(
#     initial_cond_rot[0] + 0.01 * cos,
#     initial_cond_rot[1] + 0.01 * sin,
#     marker="o",
#     markersize=4,
#     color="purple",
#     linestyle="",
#     label="Initial Point",
# )
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
# plt.legend()
plt.legend(bbox_to_anchor=(0, 0, 1, 0.2), loc="lower left", mode="expand", ncol=5)
plt.savefig("radialp_orbits.png")
plt.show()
