import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import orbits
from constants import M_S, ORBIT_NUM, PRECISION, R  # User defined constants
from constants import greek_theta  # Derived constants

cos = math.cos(greek_theta)
sin = math.sin(greek_theta)
# For radial and tangential perturbation directions

mass = np.linspace(1.0e-05, 1e-3, 150)  # range of planetary masses
max_wander = np.zeros_like(mass)
for n in range(len(mass)):
    import constants

    constants.solar_rad = R * mass[n] / (M_S + mass[n])
    constants.planet_rad = R * M_S / (M_S + mass[n])
    constants.period = math.sqrt(R ** 3 / (M_S + mass[n]))
    constants.lagrange = (constants.planet_rad - R / 2, R * math.sqrt(3) / 2, 0)
    constants.greek_theta = np.arctan(
        (R * math.sqrt(3) / 2) / (R / 2 - constants.solar_rad)
    )

    cos = math.cos(constants.greek_theta)
    sin = math.sin(constants.greek_theta)

    import orbits

    orbits.M_P = mass[n]
    orbits.solar_rad = R * mass[n] / (M_S + mass[n])
    orbits.planet_rad = R * M_S / (M_S + mass[n])
    orbits.period = math.sqrt(R ** 3 / (M_S + mass[n]))
    orbits.time_span = np.linspace(
        0, ORBIT_NUM * orbits.period, int(ORBIT_NUM * PRECISION)
    )
    orbits.omega = 2 * np.pi / constants.period  # angular velocity of frame

    initial_cond = np.array((constants.lagrange[0], constants.lagrange[1], 0, 0, 0, 0))
    orbit = orbits.rotating_frame(
        (initial_cond + 1e-3 * np.array((cos, sin, 0, sin, -cos, 0)))
    )  # radial perturbation
    wander_t = np.zeros((len(orbit.t)))
    for i in range(len(orbit.t)):
        wander_t[i] = np.linalg.norm(
            orbit.y[0:3, i] - initial_cond[0:3]
        )  # deviation in pos only
    max_wander[n] = np.max(wander_t)
    print(str(n) + ": " + str(orbits.M_P))


fig = plt.figure()
ax = fig.add_subplot()

coeff = np.polyfit(np.reciprocal(np.sqrt(mass)), max_wander, 1)  # for line of best fit
order = np.argsort(max_wander)

plt.plot(mass, max_wander, linestyle="None", marker="x", color="navy")
plt.plot(
    mass[order],
    coeff[0] * np.reciprocal(np.sqrt(mass[order])) + coeff[1],
    linewidth=0.75,
)  # Only plot best fit line in small mass region
plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
plt.text(
    0.465,
    0.95,
    "Line of Best Fit: "
    + str("{0:.2e}".format(coeff[0]))
    + "/\u221AM"
    + str("{0:+.2e}".format(coeff[1])),
    transform=ax.transAxes,
    fontsize=9.5,
)
plt.title("Variation of Wander with Planetary Mass")
plt.xlabel("Planet Mass /Solar Masses")
plt.ylabel("Wander /AU")
plt.savefig("wanderwithplanetmass.png")
plt.show()
