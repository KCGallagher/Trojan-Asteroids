import math
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# USER DEFINED CONSTANTS
G = 4 * np.pi ** 2  # gravitational constant in astronomical units
M_s = 1  # solar mass
M_p = 0.001  # planetary mass
R = 5.2  # average planetary radius from sun
# T = R ** 1.5  # jupiter -> 11.8618 years

precision = 100  # evaluation points per orbit

# derived quantities
solar_position = R * M_p / (M_s + M_p)  # distance from origin to sun in CoM frame
planetary_postion = R * M_s / (M_s + M_p)
w = 2 * np.pi / T  # angular velocity of frame

# co - ordinates of lagrange point
lx = rp - R / 2
ly = math.sqrt(3) * R / 2

