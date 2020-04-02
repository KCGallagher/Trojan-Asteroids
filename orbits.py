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
greek_rad = math.sqrt(R ** 2 + solar_rad * planet_rad)
greek_v = omega * (greek_rad)

lagrange_x = planet_rad - R / 2  # co-ordinates of lagrange point from CoM
lagrange_y = R * math.sqrt(3) / 2

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

    virtual_force_x = 2 * omega * y_vel + x_pos * omega ** 2
    virtual_force_y = -2 * omega * x_vel + y_pos * omega ** 2

    return (
        x_vel,
        y_vel,
        -G
        * (
            M_S * (x_pos + solar_rad) / solar_dr3
            + M_P * (x_pos - planet_rad) / planet_dr3
        )
        + virtual_force_x,
        -G * (M_S * y_pos / solar_dr3 + M_P * y_pos / planet_dr3 + virtual_force_y),
    )


def rotating_frame(yo_rot=(rcos, rsin, -vcos, vsin)):
    return integrate.solve_ivp(
        fun=rot_derivatives,
        t_span=(0, ORBIT_NUM * period),
        y0=yo_rot,
        t_eval=np.linspace(
            0, ORBIT_NUM * period, ORBIT_NUM * PRECISION
        ),  # selects points for storage
    )


# DEFINED FUNCTIONS IN THE STATIONARY FRAME
def stat_acceleration(pos, state):
    pos = x_pos, y_pos
    state[2] = planet_x, planet_y
    state[4] = solar_x, solar_y

    # factors defined for convenience
    solar_dr3 = ((solar_x - x_pos) ** 2 + (solar_y - y_pos) ** 2) ** 1.5
    planet_dr3 = ((planet_x - x_pos) ** 2 + (planet_y - y_pos) ** 2) ** 1.5

    return (
        -G
        * (M_S * (x_pos - solar_x) / solar_dr3 + M_P * (x_pos - planet_x) / planet_dr3),
        -G
        * (M_S * (x_pos - solar_y) / solar_dr3 + M_P * (x_pos - planet_y) / planet_dr3),
    )


np.linspace(
            0, ORBIT_NUM * period, ORBIT_NUM * PRECISION

def stat_derivatives(t, y):
    (
        greek_pos,
        greek_vel,
        planet_pos,
        planet_vel,
        solar_pos,
        solar_vel,
    ) = y  # all position and velocity variables included in one array

    return (
        greek_vel,
        stat_acceleration(greek_pos, y),
        planet_vel,
        stat_force(planet_pos, y),
        solar_vel,
        stat_force(solar_pos, y),
    )


initial_conditions = (
    rcos,
    rsin,
    -vcos,
    vsin,
    planet_rad,
    0,
    0,
    omega * planet_rad,
    -solar_rad,
    0,
    0,
    -omega * solar_rad,
)


def stationary_frame(y0_stat=initial_conditions):
    return integrate.solve_ivp(
        fun=stat_derivatives,
        t_span=(0, ORBIT_NUM * period),
        y0=initial_conditions,
        t_eval=np.linspace(
            0, ORBIT_NUM * period, ORBIT_NUM * PRECISION
        ),  # selects points for storage
    )
