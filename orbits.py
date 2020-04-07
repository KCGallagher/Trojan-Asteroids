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
# decide later which are most useful to import etc

# Initial conditions for Greeks, given as equillibrium state but can be changed in XXXXX
cos, sin = np.cos(greek_theta), np.sin(greek_theta)
rcos, rsin = greek_rad * cos, greek_rad * sin
vcos, vsin = greek_v * cos, greek_v * sin

# Positions of massize bodies
def solar_pos(t=0):
    """Position of sun in 3-dimensions at time t

    Note that z coordinate is fixed to 0, 
    so movement is constrained to xy plane in this exact solution
    """
    return np.array([-solar_rad * np.cos(omega * t), -solar_rad * np.sin(omega * t), 0])


def planet_pos(t=0):
    """Position of planet in 3-dimensions at time t

    Note that z coordinate is fixed to 0, 
    so movement is constrained to xy plane in this exact solution"""
    return np.array([planet_rad * np.cos(omega * t), planet_rad * np.sin(omega * t), 0])


print(planet_pos(time_span)[0])


def lagrange_pos(t=0):
    return np.array(
        [
            greek_rad * np.cos(omega * t + greek_theta),
            greek_rad * np.sin(omega * t + greek_theta),
            0,
        ]
    )


# assuming massive bodies do not deviate from xy plane


def rot_derivatives(t, y):
    """Gives derivative of each term of y at time t in the rotating frame

    y should be of the form (x_pos, y_pos, z_pos, x_vel, y_vel, z_vel)
    """

    position, velocity = np.array(y[0:3]), np.array(y[3:6])

    solar_dr3 = np.linalg.norm(solar_pos(0) - position) ** 3  # should this be zero?
    planet_dr3 = np.linalg.norm(planet_pos(0) - position) ** 3

    virtual_force = (
        position[0] * omega ** 2 + 2 * omega * velocity[1],
        position[1] * omega ** 2 - 2 * omega * velocity[0],
        0,
    )

    acceleration = (
        -G
        * (
            M_S * (position[0] - solar_pos(0)[0]) / solar_dr3
            + M_P * (position[0] - planet_pos(0)[0]) / planet_dr3
        )
        + virtual_force[0],
        -G * (M_S * position[1] / solar_dr3 + M_P * position[1] / planet_dr3)
        + virtual_force[1],
        -G * (M_S * position[2] / solar_dr3 + M_P * position[2] / planet_dr3)
        + virtual_force[2],
    )

    return np.concatenate((velocity, acceleration))


def rotating_frame(y0_rot=(rcos, rsin, 0, 0, 0, 0)):
    """Gives position and vlocity of asteroids in the rotating frame

    Uses scipy solve_ivp method with LSODA, taking an input in the form
    of (x_pos, y_pos, z_pos, x_vel, y_vel, z_vel) for the initial state.
    This has a default value of the equillibrium position if not given
    """

    return integrate.solve_ivp(
        fun=rot_derivatives,
        method="LSODA",  ###on adams advice
        t_span=(0, ORBIT_NUM * period),
        y0=y0_rot,
        t_eval=time_span,  # selects points for storage
    )


# DEFINED FUNCTIONS IN THE STATIONARY FRAME
def stat_acceleration(position, t):
    """Gives acceleration of asteroids in stationary frame at a given position and time"""
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
    """Gives derivative of each term of y at time t in the stationary frame

    y should be of the form (x_pos, y_pos, z_pos, x_vel, y_vel, z_vel)
    """
    position, velocity = np.array(y[0:3]), np.array(y[3:6])
    return np.concatenate((velocity, stat_acceleration(position, t)))


def stationary_frame(y0_stat=(rcos, rsin, 0, -vsin, vcos, 0)):
    """Gives position and vlocity of asteroids in the stationary frame

    Uses scipy solve_ivp method with LSODA, taking an input in the form
    of (x_pos, y_pos, z_pos, x_vel, y_vel, z_vel) for the initial state.
    This has a default value of the equillibrium state if not given.
    """
    return integrate.solve_ivp(
        fun=stat_derivatives,
        method="LSODA",  ##on adams advice
        t_span=(0, ORBIT_NUM * period),
        y0=y0_stat,
        t_eval=time_span,
    )
