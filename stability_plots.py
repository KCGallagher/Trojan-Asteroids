import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

import orbits
from constants import M_P, M_S, ORBIT_NUM, PRECISION, G, R  # User defined constants
from constants import (
    solar_rad,
    planet_rad,
    period,
    greek_theta,
    time_span,
    lagrange,
)  # Derived constants

sin = math.sin(greek_theta)
cos = math.cos(greek_theta)

# ROTATING FRAME
orbit_sol = orbits.rotating_frame()


# DEVIATION FROM LAGRANGE POINT
plt.plot(time_span, ((lagrange[0] - orbit_sol.y[0, :])))
# plt.yscale("log", basey=10)
plt.ticklabel_format(axis="y", style="sci", scilimits=None)
plt.ylabel("Magnitude of Deviation in x direction /AU")
plt.xlabel("Time /years")
plt.title("Asteroid Deviation from Lagrange point")
# plt.savefig("asteroid_deviation_linear.png")
plt.show()


orbit_values = orbits.rotating_frame()

plt.plot(orbit_values.y[0, :] - lagrange[0], orbit_values.y[1, :] - lagrange[1])
plt.ticklabel_format(axis="both", style="sci", scilimits=None, useOffset=False)
plt.title("Rotating Frame just Greeks")
plt.show()


wander = np.zeros((len(orbit_values.t), 2))

for i in range(len(wander)):
    theta = np.arctan(
        (orbit_values.y[1, i] - lagrange[1]) / (orbit_values.y[0, i] - lagrange[0])
    )
    wander[i, 0] = np.linalg.norm((orbit_values.y[0:3, i] - lagrange[0:3])) * np.abs(
        np.cos(greek_theta - theta)
    )  # radial component of wander
    wander[i, 1] = np.linalg.norm((orbit_values.y[0:3, i] - lagrange[0:3])) * np.abs(
        np.sin(greek_theta - theta)
    )  # tangential component of wander

plt.plot(time_span, wander[:, 0])
# plt.yscale("log", basey=10)
plt.ticklabel_format(axis="y", style="", scilimits=None)
plt.ylabel("Radial Deviation /AU")
plt.xlabel("Time /years")
plt.title("Asteroid Deviation from Lagrange point")
# plt.savefig("asteroid_deviation_linear.png")
plt.show()

plt.plot(time_span, wander[:, 1])
# plt.yscale("log", basey=10)
plt.ticklabel_format(axis="y", style="", scilimits=None)
plt.ylabel("Tangential Deviation /AU")
plt.xlabel("Time /years")
plt.title("Asteroid Deviation from Lagrange point")
# plt.savefig("asteroid_deviation_linear.png")
plt.show()


#   polars

wander = np.zeros((len(orbit_values.t), 2))  # radius, angle

for i in range(len(orbit_values.t)):
    wander[i, 0] = np.linalg.norm(
        (orbit_values.y[0:3, i] - lagrange[0:3])
    )  # deviation in pos only
    wander[i, 1] = np.arctan(
        (orbit_values.y[1, i] - lagrange[1]) / (orbit_values.y[0, i] - lagrange[0])
    )  # in xy plane only

plt.plot(time_span, wander[:, 0])
# plt.yscale("log", basey=10)
plt.ticklabel_format(axis="y", style="", scilimits=None)
plt.ylabel("Radius /AU")
plt.xlabel("Time /years")
plt.title("Asteroid Deviation from Lagrange point")
# plt.savefig("asteroid_deviation_linear.png")
plt.show()

plt.plot(time_span, wander[:, 1])
# plt.yscale("log", basey=10)
plt.ticklabel_format(axis="y", style="", scilimits=None)
plt.ylabel("Angle /rad")
plt.xlabel("Time /years")
plt.title("Asteroid Deviation from Lagrange point")
# plt.savefig("asteroid_deviation_linear.png")
plt.show()


wander = np.zeros((len(orbit_values.t), 2))  # radius, angle

for i in range(len(orbit_values.t)):
    wander[i, 0] = np.linalg.norm(
        (orbit_values.y[0:3, i])  # - lagrange[0:3])
    )  # deviation in pos only
    wander[i, 1] = np.arctan(
        (orbit_values.y[1, i] / orbit_values.y[0, i])
    )  # in xy plane only

plt.plot(time_span, wander[:, 0])
# plt.yscale("log", basey=10)
plt.ticklabel_format(axis="y", style="", scilimits=None)
plt.ylabel("Radius /AU")
plt.xlabel("Time /years")
plt.title("Absolute Asteroid Position")
# plt.savefig("asteroid_deviation_linear.png")
plt.show()

plt.plot(time_span, wander[:, 1])
# plt.yscale("log", basey=10)
plt.ticklabel_format(axis="y", style="", scilimits=None)
plt.ylabel("Angle /rad")
plt.xlabel("Time /years")
plt.title("Absolute Asteroid Position")
# plt.savefig("asteroid_deviation_linear.png")
plt.show()


# for chaneg in orbit number

# orbit_number = np.arange(10, 2010, 500)  # range of planetary masses
# max_wander = np.zeros_like(orbit_number)
# for n in range(len(orbit_number)):
#     import constants

#     constants.time_span = np.linspace(
#         0, orbit_number * constants.period, int(orbit_number[n] * constants.PRECISION)
#     )

#     import orbits

#     orbit = orbits.rotating_frame()
#     wander_t = np.zeros((len(orbit.t)))
#     for i in range(len(orbit.t)):
#         wander_t[i] = np.linalg.norm(
#             orbit.y[0:3, i] - lagrange[0:3]
#         )  # deviation in pos only
#     print("*" + str(np.amax(wander_t)))
#     max_wander[n] = np.amax(wander_t)
#     # max_wander[n] = wander_t.max()
#     print("**" + str(max_wander[n]))
# print(max_wander)
# plt.plot(orbit_number, max_wander)
# plt.title("Change in wander with different orbit numbers")
# plt.xlabel("Orbit Num")
# plt.ylabel("Wander /AU")
# # plt.savefig("wanderwithorbitnum.png")
# plt.show()
