import math
import numpy as np

# USER DEFINED CONSTANTS
G = 4 * np.pi ** 2  # gravitational constant in astronomical units
M_S = 1  # solar mass
M_P = 0.001  # planetary mass
R = 5.2  # average planetary radius from sun


PRECISION = 100  # evaluation points per orbit
ORBIT_NUM = 100  # number of orbits


# DERIVED CONSTANTS
solar_rad = R * M_P / (M_S + M_P)  # distance from origin to sun in CoM frame
planet_rad = R * M_S / (M_S + M_P)
period = math.sqrt(R ** 3 / (M_S + M_P))
omega = 2 * np.pi / period  # angular velocity of frame
time_span = np.linspace(0, ORBIT_NUM * period, int(ORBIT_NUM * PRECISION))
lagrange = (planet_rad - R / 2, R * math.sqrt(3) / 2, 0)
greek_theta = np.arctan((R * math.sqrt(3) / 2) / (R / 2 - solar_rad))
