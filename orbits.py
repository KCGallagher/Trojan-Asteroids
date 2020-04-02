import math
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

from constants import G, M_S, M_P, R, ORBIT_NUM, PRECISION


def rot_derivatives(t, y, rotating_frame=False):
    (
        x_pos,
        y_pos,
        x_vel,
        y_vel,
    ) = y  # all position and velocity variables included in one array

    # factors defined for convenience
    solar_dr3 = ((solar_rad + x_pos) ** 2 + y_pos ** 2) ** 1.5
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


def rotating_frame(initial_conditions=(rcos, rsin, -vcos, vsin)):
    return integrate.solve_ivp(
        fun=rot_derivatives,
        t_span=(0, ORBIT_NUM * period),
        y0=initial_conditions,
        t_eval=np.linspace(
            0, ORBIT_NUM * period, ORBIT_NUM * PRECISION
        ),  # selects points for storage
    )
