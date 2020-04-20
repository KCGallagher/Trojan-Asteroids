import numpy as np
import matplotlib.pyplot as plt
import timeit

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

time_repetitions = 100
time = timeit.timeit(
    "orbits.rotating_frame()", globals=globals(), number=time_repetitions
)
print(
    "Evaluated ODE system in "
    + str("{0:.4g}".format(time / time_repetitions))
    + " seconds"
)

orbit_values = orbits.rotating_frame()
wander = np.zeros((len(orbit_values.t)))

for i in range(len(orbit_values.t)):
    wander[i] = np.linalg.norm(
        (orbit_values.y[0:3, i] - lagrange[0:3])
    )  # magnitude of distance from lagrange point

print(
    "The maximum wander from the Lagrange point over "
    + str("{0:.0f}".format(ORBIT_NUM * period))
    + " years is: "
    + str("{0:.3g}".format(np.max(wander)))
    + " AU"
)
