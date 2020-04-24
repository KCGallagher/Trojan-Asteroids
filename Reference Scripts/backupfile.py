# CURRENT VERSION OF ORBITS.PY (03/04/2020, 20;36)
# RUNS, HOWEVER GIVES CRESENT MOON SHAPE IN STAT FRAME

import math
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

from constants import G, M_S, M_P, R, ORBIT_NUM, PRECISION  # User defined constants
from constants import (
    solar_rad,
    planet_rad,
    period,
    omega,
    time_span,
)  # Derived constants

# DERIVED QUANTITIES
# actually refers to greeks in this case, symmetric to trojans
greek_theta = np.arctan((R * math.sqrt(3) / 2) / (R / 2 - solar_rad))
greek_rad = math.sqrt(R ** 2 - solar_rad * planet_rad)
greek_v = omega * (greek_rad)

lagrange = (planet_rad - R / 2, R * math.sqrt(3) / 2, 0)
# are these necessary as well as initial conditions?

# Initial conditions for Greeks, given as equillibrium state but can be changed in XXXXX
cos, sin = np.cos(greek_theta), np.sin(greek_theta)
rcos, rsin = greek_rad * cos, greek_rad * sin
vcos, vsin = greek_v * cos, greek_v * sin

# Positions of massize bodies
def solar_pos(t=0):
    return np.array([-solar_rad * np.cos(omega * t), -solar_rad * np.sin(omega * t), 0])


def planet_pos(t=0):
    return np.array([planet_rad * np.cos(omega * t), planet_rad * np.sin(omega * t), 0])


# assuming massive bodies do not deviate from xy plane


# DEFINED FUNCTIONS IN THE ROTATING FRAME
# def rot_acceleration(position, velocity, t):
#     # factors defined for convenience
#     solar_dr3 = np.linalg.norm(solar_pos(0) - position) ** 3  # should this be zero?
#     planet_dr3 = np.linalg.norm(planet_pos(0) - position) ** 3

#     virtual_force_x = position[0] * omega ** 2 + 2 * omega * velocity[1]
#     virtual_force_y = position[1] * omega ** 2 - 2 * omega * velocity[0]
#     virtual_force_z = 0  # change this!!

#     return (
#         -G
#         * (
#             M_S * (position[0] + solar_pos(0)[0]) / solar_dr3
#             + M_P * (position[0] - planet_pos(0)[0]) / planet_dr3
#         )
#         + virtual_force_x,
#         -G * (M_S * position[1] / solar_dr3 + M_P * position[1] / planet_dr3)
#         + virtual_force_y,
#         -G * (M_S * position[2] / solar_dr3 + M_P * position[2] / planet_dr3)
#         + virtual_force_z,
#     )


def rot_derivatives(t, y):
    position, velocity = np.array(y[0:3]), np.array(y[3:6])

    print(position)
    print(velocity)
    print(y)
    solar_dr3 = np.linalg.norm(solar_pos(0) - position) ** 3  # should this be zero?
    planet_dr3 = np.linalg.norm(planet_pos(0) - position) ** 3

    virtual_force_x = position[0] * omega ** 2 + 2 * omega * velocity[1]
    virtual_force_y = position[1] * omega ** 2 - 2 * omega * velocity[0]
    virtual_force_z = 0  # change this!!

    acceleration = (
        -G
        * (
            M_S * (position[0] - solar_pos(0)[0]) / solar_dr3
            + M_P * (position[0] - planet_pos(0)[0]) / planet_dr3
        )
        + virtual_force_x,
        -G * (M_S * position[1] / solar_dr3 + M_P * position[1] / planet_dr3)
        + virtual_force_y,
        -G * (M_S * position[2] / solar_dr3 + M_P * position[2] / planet_dr3)
        + virtual_force_z,
    )

    return np.concatenate((velocity, acceleration))


print(rot_derivatives(0, (1, 1, 0, 0, 0, 0)))


def rotating_frame(y0_rot=(rcos, rsin, 0, 0, 0, 0)):
    return integrate.solve_ivp(
        fun=rot_derivatives,
        method="LSODA",  ###on adams advice
        t_span=(0, ORBIT_NUM * period),
        y0=y0_rot,
        t_eval=time_span,  # selects points for storage
    )


# DEFINED FUNCTIONS IN THE STATIONARY FRAME
def stat_acceleration(position, t):
    # factors defined for convenience
    solar_dr3 = np.linalg.norm(solar_pos(t) - position) ** 3
    planet_dr3 = np.linalg.norm(planet_pos(t) - position) ** 3

    return (
        -G
        * (
            M_S * (position[0] - solar_pos(t)[0]) / solar_dr3
            + M_P * (position[0] - planet_pos(t)[0]) / planet_dr3
        ),
        -G
        * (
            M_S * (position[1] - solar_pos(t)[1]) / solar_dr3
            + M_P * (position[1] - planet_pos(t)[1]) / planet_dr3
        ),
        -G * (M_S * position[2] / solar_dr3 + M_P * position[2] / planet_dr3),
    )


def stat_derivatives(t, y):
    position, velocity = np.array(y[0:3]), np.array(y[3:6])
    return np.concatenate((velocity, stat_acceleration(position, t)))


def stationary_frame(y0_stat=(rcos, rsin, 0, -vsin, vcos, 0)):
    return integrate.solve_ivp(
        fun=stat_derivatives,
        method="LSODA",  ##on adams advice
        t_span=(0, ORBIT_NUM * period),
        y0=y0_stat,
        t_eval=time_span,
    )


# CURRENT VERSION OF STAT_FRAME_VIDEO BEFORE IMPLEMEMNTING LOOP 06/04/2020 15;22

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


# VERSION OF PERTURBATIONS_3D.PY MADE AT 10:47 ON 24/04/2020

import math

import matplotlib.pyplot as plt
import numpy as np

import orbits
from constants import M_P, M_S, ORBIT_NUM, PRECISION, G, R  # User defined constants
from constants import (
    solar_rad,
    planet_rad,
    period,
    time_span,
    lagrange,
)  # Derived constants

greek_theta = np.arctan((R * math.sqrt(3) / 2) / (R / 2 - solar_rad))
cos = math.cos(greek_theta)
sin = math.sin(greek_theta)

# ROTATING FRAME
initial_cond_rot = np.array((lagrange[0], lagrange[1], 0, 0, 0, 0))
perturb = 0.01 * np.array((cos, sin, 0, 0, 0, 0))
perturb_z = 0.01 * np.array((cos, sin, 10, 0, 0, 0))
orbit_sol = orbits.rotating_frame(initial_cond_rot + perturb)
orbit_solz = orbits.rotating_frame(initial_cond_rot + perturb_z)


# wander

z_perturbations = np.linspace(0, 0.1, 30)
wander_data = np.zeros((len(z_perturbations), 2))
for n in range(len(z_perturbations)):
    orbit_solz = orbits.rotating_frame(
        # initial_cond_rot + np.array((0, 0, z_perturbations[n], 0, 0, 0))
        initial_cond_rot
        + np.array((0.02 * cos, 0.02 * sin, z_perturbations[n], 0, 0, 0))
    )
    wander_t = np.zeros((len(orbit_solz.t)))
    wander_z = np.zeros((len(orbit_solz.t)))
    for i in range(len(orbit_solz.t)):
        wander_t[i] = np.linalg.norm(
            orbit_solz.y[0:3, i] - initial_cond_rot[0:3] - perturb_z[0:3]
        )  # deviation in pos magnitude
        wander_z[i] = orbit_solz.y[2, i] - perturb_z[2]  # deviation in z pos only
    wander_data[n, 0] = np.max(wander_t)
    wander_data[n, 1] = np.max(wander_z)
    print(n)

fig = plt.figure()
ax = fig.add_subplot()

coeff = np.polyfit(
    z_perturbations, wander_data[:, 0], 2
)  # for quadratic line of best fit
best_fit = np.poly1d(coeff)
order = np.argsort(z_perturbations)

quad_label = str(
    "Quadratic Best Fit: "
    + str("{0:.2f}".format(coeff[0]))
    + "x\u00b2 "
    + str("{0:+.2f}".format(coeff[1]))
    + "x "
    + str("{0:+.2f}".format(coeff[2]))
)

plt.plot(z_perturbations, best_fit(z_perturbations), linewidth=0.75, label=quad_label)

plt.plot(
    z_perturbations,
    wander_data[:, 0],
    label="Magnitude",
    linestyle="None",
    marker="+",
    color="navy",
)
# plt.plot(z_perturbations, np.sqrt(1 + z_perturbations ** 2), label="Prediction")
# plt.plot(
#     z_perturbations,
#     (wander_data[-1, 0] / np.sqrt(wander_data[0, 0] ** 2 + z_perturbations[-1] ** 2))
#     * np.sqrt(wander_data[0, 0] ** 2 + z_perturbations ** 2),
#     label="Prediction2",
# )
# plt.plot(z_perturbations, wander_data[:, 1], label="Z component")


plt.title("Wander under z- perturbations")
plt.legend()
plt.show()


# DEVIATION FROM LAGRANGE POINT
plt.plot(time_span, orbit_solz.y[2, :])
# plt.yscale("log", basey=10)
plt.ticklabel_format(axis="y", style="", scilimits=None)
plt.ylabel("Magnitude of Deviation in z-direction /AU")
plt.xlabel("Time /years")
plt.title("Asteroid Deviation from Lagrange point")
# plt.savefig("asteroid_zdeviation_linear.png")
plt.show()

# MOTION IN ROTATING FRAME
plt.plot(
    orbits.solar_pos(0)[0],
    0,
    label="Sun",
    color="yellow",
    markersize=12,
    marker="o",
    linestyle="None",
)
plt.plot(
    orbits.planet_pos(0)[0],
    0,
    label="Jupiter",
    color="red",
    markersize=8,
    marker="o",
    linestyle="None",
)

plt.plot(orbit_sol.y[0, :], orbit_sol.y[1, :], label="Greeks", color="blue")
plt.legend()
plt.title("Rotating Frame")
plt.show()

plt.plot(orbit_solz.y[0, :], orbit_solz.y[1, :], label="Z")
plt.plot(
    orbit_sol.y[0, :],
    orbit_sol.y[1, :],
    label="No Z",
    linewidth=0.75,
    linestyle="dashed",
    color="black",
)
plt.title("Rotating Frame just Greeks")
plt.show()
