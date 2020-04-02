import math
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

import orbits
from constants import G, M_S, M_P, R, ORBIT_NUM, PRECISION


orbit_sol = orbits.rotating_frame()
for i in range(len(orbit_sol.t)):
    print(orbit_sol.t[i], orbit_sol.y[0, i], orbit_sol.y[1, i])


plt.plot(orbit_sol.y[0, :], orbit_sol.y[1, :], label="Greeks")
plt.legend()
plt.show()


orbit_sol2 = orbits.stationary_frame()
for i in range(len(orbit_sol2.t)):
    print(orbit_sol2.t[i], orbit_sol2.y[0, i], orbit_sol.y[1, i])


plt.plot(orbit_sol2.y[0, :], orbit_sol2.y[1, :], label="Greeks")
plt.legend()
plt.show()
