import math
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

from constants import G, M_S, M_P, R, ORBIT_NUM, PRECISION

# DERIVED QUANTITIES
solar_rad = R * M_P / (M_S + M_P)  # distance from origin to sun in CoM frame
planet_rad = R * M_S / (M_S + M_P)

period = math.sqrt(R ** 3 / (M_S + M_P))
omega = 2 * np.pi / period  # angular velocity of frame

# actually refers to greeks in this case, symmetric to trojans
greek_theta = np.arctan((R * math.sqrt(3) / 2) / (R / 2 - solar_rad))
greek_rad = math.sqrt(R ** 2 - solar_rad * planet_rad)
greek_v = omega * (greek_rad)

lagrange_x = planet_rad - R / 2  # co-ordinates of lagrange point from CoM
lagrange_y = R * math.sqrt(3) / 2
# are these necessary as well as initial conditions?

# Initial conditions for Greeks, given as equillibrium state but can be changed in XXXXX

cos, sin = np.cos(greek_theta), np.sin(greek_theta)
rcos, rsin = greek_rad * cos, greek_rad * sin
vcos, vsin = greek_v * cos, greek_v * sin


# DEFINED FUNCTIONS IN THE ROTATING FRAME
def rot_derivatives(t, y):
    (
        x_pos,
        y_pos,
        x_vel,
        y_vel,
    ) = y  # all position and velocity variables included in one array

    # factors defined for convenience
    solar_dr3 = ((solar_rad + x_pos) ** 2 + y_pos ** 2) ** 1.5
    # Note that solar radius has opposite orientation
    planet_dr3 = ((planet_rad - x_pos) ** 2 + y_pos ** 2) ** 1.5

    virtual_force_x = x_pos * omega ** 2 + 2 * omega * y_vel
    virtual_force_y = y_pos * omega ** 2 - 2 * omega * x_vel
    return (
        x_vel,
        y_vel,
        -G
        * (
            M_S * (x_pos + solar_rad) / solar_dr3
            + M_P * (x_pos - planet_rad) / planet_dr3
        )
        + virtual_force_x,
        -G * (M_S * y_pos / solar_dr3 + M_P * y_pos / planet_dr3) + virtual_force_y,
    )


def rotating_frame(y0_rot=(rcos, rsin, 0, 0)):
    return integrate.solve_ivp(
        fun=rot_derivatives,
        method="LSODA",  ###on adams advice
        t_span=(0, ORBIT_NUM * period),
        y0=y0_rot,
        t_eval=np.linspace(
            0, ORBIT_NUM * period, int(ORBIT_NUM * PRECISION)
        ),  # selects points for storage
    )


# DEFINED FUNCTIONS IN THE STATIONARY FRAME
def solar_pos(t):
    return np.array([-solar_rad * np.cos(omega * t), -solar_rad * np.sin(omega * t)])


def planet_pos(t):
    return np.array([planet_rad * np.cos(omega * t), planet_rad * np.sin(omega * t)])


def stat_acceleration(x_pos, y_pos, t):
    planet_x = planet_pos(t)[0]
    planet_y = planet_pos(t)[1]
    solar_x = solar_pos(t)[0]
    solar_y = solar_pos(t)[1]

    # factors defined for convenience
    solar_dr3 = ((solar_x - x_pos) ** 2 + (solar_y - y_pos) ** 2) ** 1.5
    planet_dr3 = ((planet_x - x_pos) ** 2 + (planet_y - y_pos) ** 2) ** 1.5
    return (
        -G
        * (M_S * (x_pos - solar_x) / solar_dr3 + M_P * (x_pos - planet_x) / planet_dr3),
        -G
        * (M_S * (y_pos - solar_y) / solar_dr3 + M_P * (y_pos - planet_y) / planet_dr3),
    )


# time_span = np.linspace(0, ORBIT_NUM * period, int(ORBIT_NUM * PRECISION))


def stat_derivatives(t, y):
    (
        x_pos,
        y_pos,
        x_vel,
        y_vel,
    ) = y  # all position and velocity variables included in one array

    return (
        x_vel,
        y_vel,
        stat_acceleration(x_pos, y_pos, t)[0],
        stat_acceleration(x_pos, y_pos, t)[1],
    )


initial_conditions = (
    rcos,
    rsin,
    -vsin,
    vcos,
)


def stationary_frame(y0_stat=initial_conditions):
    return integrate.solve_ivp(
        fun=stat_derivatives,
        method="LSODA",  ##on adams advice
        t_span=(0, ORBIT_NUM * period),
        y0=y0_stat,
        t_eval=np.linspace(
            0, ORBIT_NUM * period, int(ORBIT_NUM * PRECISION)
        ),  # selects points for storage
    )
