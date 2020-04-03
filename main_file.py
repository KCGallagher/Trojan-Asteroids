import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

import orbits
from constants import M_P, M_S, ORBIT_NUM, PRECISION, G, R

orbit_sol = orbits.rotating_frame()
for i in range(len(orbit_sol.t)):
    print(orbit_sol.t[i], orbit_sol.y[0, i], orbit_sol.y[1, i])

solar_rad = R * M_P / (M_S + M_P)  # distance from origin to sun in CoM frame
planet_rad = R * M_S / (M_S + M_P)
period = math.sqrt(R ** 3 / (M_S + M_P))
time_span = np.linspace(0, ORBIT_NUM * period, int(ORBIT_NUM * PRECISION))

lagrange_x = planet_rad - R / 2
plt.plot(time_span, (np.abs(lagrange_x - orbit_sol.y[0, :])))
plt.yscale("log", basey=10)
plt.ylabel("absolute error on x_pos")
plt.xlabel("time")
plt.show()

plt.plot(
    solar_rad,
    0,
    label="Sun",
    color="yellow",
    markersize=12,
    marker="o",
    linestyle="None",
)
plt.plot(
    planet_rad,
    0,
    label="Jupiter",
    color="red",
    markersize=8,
    marker="o",
    linestyle="None",
)

plt.plot(orbit_sol.y[0, :], orbit_sol.y[1, :], label="Greeks")
plt.legend()
plt.title("Rotating Frame")
plt.show()

plt.plot(orbit_sol.y[0, :], orbit_sol.y[1, :], label="Greeks")
plt.title("Rotating Frame just Greeks")
plt.show()


orbit_sol2 = orbits.stationary_frame(
    # y0_stat=(2.597396111680207, 4.507828683528303, -1.3769877532043, 2.3897875502501607)
)
for i in range(len(orbit_sol2.t)):
    print(orbit_sol2.t[i], orbit_sol2.y[0, i], orbit_sol.y[1, i])

time_span = np.linspace(0, ORBIT_NUM * period, int(ORBIT_NUM * PRECISION))

plt.plot(0, 0, label="CoM", color="black", marker="x", linestyle="None")
plt.plot(
    orbits.solar_pos(time_span)[0],
    orbits.solar_pos(time_span)[1],
    label="Sun",
    color="yellow",
    # markersize=12,
    # marker="o",
    linestyle="dotted",
)
plt.plot(
    orbits.planet_pos(time_span)[0],
    orbits.planet_pos(time_span)[1],
    label="Jupiter",
    color="red",
    # markersize=8,
    # marker="o",
    linestyle="dotted",
)
plt.plot(orbit_sol2.y[0, :], orbit_sol2.y[1, :], label="Greeks")
plt.title("Stationary Frame")
plt.legend()
plt.show()
