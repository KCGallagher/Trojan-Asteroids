import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import orbits
from constants import M_P, M_S, ORBIT_NUM, PRECISION, G, R  # User defined constants
from constants import lagrange, omega, time_span  # Derived constants

# Set up orbits for each body
orbit_greeks = orbits.stationary_frame(
    (lagrange[0], lagrange[1], 0, -omega * lagrange[1], omega * lagrange[0], 0)
)
orbit_trojans = orbits.stationary_frame(
    (lagrange[0], -lagrange[1], 0, -omega * lagrange[1], -omega * lagrange[0], 0)
)
orbit_list = [orbit_greeks, orbit_trojans]


def data_gen(orbit_data):
    gen_list = (
        [orbit_data.y[0, t], orbit_data.y[1, t]]
        for t in np.arange(
            0, ORBIT_NUM * PRECISION, 100
        )  # step size of 100 pulls out each 100th point to speed up animation
    )
    return gen_list


def init_g():
    return point_g


def run_g(data):
    t, y = data
    point_g.set_data(t, y)
    return point_g


fig = plt.figure()
ax = plt.axes(xlim=(-8, 8), ylim=(-8, 8))
ax.set_aspect(aspect=1)

(point_g,) = ax.plot([0], [0], "go", label="Greeks")
point_g.set_data(0, 0)

ani = animation.FuncAnimation(
    fig, run_g, data_gen(orbit_greeks), init_func=init_g, interval=1
)
plt.title("Asteroid orbit in static frame")
plt.xlabel("X distance/ AU")
plt.ylabel("Y distance/ AU")
plt.legend()
plt.show()

# ani.save("asteroid_orbit.gif", writer="imagemagick")
