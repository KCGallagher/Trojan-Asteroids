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

mass = np.linspace(0.001, 0.02, 100)  # range of planetary masses
max_wander = np.zeros_like(mass)
for n in range(len(mass)):
    import constants

    constants.M_P = mass[n]
    constants.solar_rad = R * mass[n] / (M_S + mass[n])
    constants.planet_rad = R * M_S / (M_S + mass[n])
    constants.period = math.sqrt(R ** 3 / (M_S + mass[n]))
    constants.omega = 2 * np.pi / constants.period  # angular velocity of frame
    constants.lagrange = (constants.planet_rad - R / 2, R * math.sqrt(3) / 2, 0)

    import orbits

    initial_cond = np.array((constants.lagrange[0], constants.lagrange[1], 0, 0, 0, 0))
    orbit = orbits.rotating_frame(initial_cond)
    wander_t = np.zeros((len(orbit.t)))
    for i in range(len(orbit.t)):
        wander_t[i] = np.linalg.norm(
            orbit.y[0:3, i] - initial_cond[0:3]
        )  # deviation in pos only
    max_wander[n] = np.max(wander_t)
plt.plot(mass, max_wander)
plt.title("Change in wander with different planet mass")
plt.xlabel("Planet Mass /Solar Sasses")
plt.ylabel("Wander /AU")
plt.savefig("wanderwithplanetmass.png")
plt.show()
