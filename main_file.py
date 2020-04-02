import math
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

import orbits
from constants import G, M_S, M_P, R, ORBIT_NUM, PRECISION


# DERIVED QUANTITIES
solar_rad = R * M_P / (M_S + M_P)  # distance from origin to sun in CoM frame
planet_rad = R * M_S / (M_S + M_P)

period = math.sqrt(R ** 3 / (M_S + M_P))
omega = 2 * np.pi / period  # angular velocity of frame

# actually refers to greeks in this case, symmetric to trojans
trojan_theta = np.arctan((R * math.sqrt(3) / 2) / (R / 2 - solar_rad))
trojan_rad = math.sqrt(R ** 2 + solar_rad * planet_rad)
trojan_v = omega * (trojan_rad)

lagrange_x = planet_rad - R / 2  # co-ordinates of lagrange point from CoM
lagrange_y = R * math.sqrt(3) / 2

# Initial conditions, given as equillibrium state but can be changed in XXXXX

cos, sin = np.cos(trojan_theta), np.sin(trojan_theta)
rcos, rsin = trojan_rad * cos, trojan_rad * sin
vcos, vsin = trojan_v * cos, trojan_v * sin


def derivatives(t, y, rotating_frame=False):
    (
        x_pos,
        y_pos,
        x_vel,
        y_vel,
    ) = y  # all position and velocity variables included in one array

    # factors defined for convenience
    solar_dr3 = ((solar_rad + x_pos) ** 2 + y_pos ** 2) ** 1.5
    planet_dr3 = ((planet_rad - x_pos) ** 2 + y_pos ** 2) ** 1.5

    if rotating_frame:
        virtual_force_x = 2 * omega * y_vel + x_pos * omega ** 2
        virtual_force_y = -2 * omega * x_vel + y_pos * omega ** 2
    else:
        virtual_force_x, virtual_force_y = 0, 0

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


rotating_frame = False


def orbit(initial_conditions=(rcos, rsin, -vcos, vsin)):
    return integrate.solve_ivp(
        fun=derivatives,
        t_span=(0, ORBIT_NUM * period),
        y0=initial_conditions,
        t_eval=np.linspace(
            0, ORBIT_NUM * period, ORBIT_NUM * PRECISION
        ),  # selects points for storage
    )


orbit_sol = orbit()
for i in range(len(orbit_sol.t)):
    print(orbit_sol.t[i], orbit_sol.y[0, i], orbit_sol.y[1, i])


plt.plot(orbit_sol.y[0, :], orbit_sol.y[1, :], label="Greeks")
plt.legend()
plt.show()
