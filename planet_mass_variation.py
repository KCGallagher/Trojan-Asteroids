import math
import matplotlib.pyplot as plt
import numpy as np

import orbits

# from wander import wander
from constants import M_P, M_S, ORBIT_NUM, PRECISION, G, R  # User defined constants
from constants import solar_rad  # Derived constants

# import wander

greek_theta = np.arctan((R * math.sqrt(3) / 2) / (R / 2 - solar_rad))
cos = math.cos(greek_theta)
sin = math.sin(greek_theta)

mass = np.linspace(0.002, 0.05, 100)  # range of planetary masses
max_wander = np.zeros_like(mass)
for n in range(len(mass)):
    import constants

    constants.M_P = mass[n]
    constants.solar_rad = R * mass[n] / (M_S + mass[n])
    constants.planet_rad = R * M_S / (M_S + mass[n])
    constants.period = math.sqrt(R ** 3 / (M_S + mass[n]))
    constants.omega = 2 * np.pi / constants.period  # angular velocity of frame
    constants.lagrange = (constants.planet_rad - R / 2, R * math.sqrt(3) / 2, 0)

    import orbits

    initial_cond = np.array((constants.lagrange[0], constants.lagrange[1], 0, 0, 0, 0))
    # orbit = orbits.rotating_frame(initial_cond)
    orbit = orbits.rotating_frame(
        (initial_cond + 0.001 * np.array((cos, sin, 0, sin, -cos, 0)))
    )  # radial perturbation
    wander_t = np.zeros((len(orbit.t)))
    for i in range(len(orbit.t)):
        wander_t[i] = np.linalg.norm(
            orbit.y[0:3, i] - initial_cond[0:3]
        )  # deviation in pos only
    max_wander[n] = np.max(wander_t)
    print(n)

fig = plt.figure()
ax = fig.add_subplot()
plt.plot(mass, max_wander, marker="x", linestyle="")

coeff_quad = np.polyfit(
    mass[0:50], max_wander[0:50], 2
)  # for quadratic line of best fit
best_fit_quad = np.poly1d(coeff_quad)
best_fit_lin = np.poly1d(coeff_quad[1:])
# without quadratic term for comparison
order = np.argsort(mass)

quad_label = str(
    "Quadratic Best Fit: "
    + str("{0:.2f}".format(coeff_quad[0]))
    + "x\u00b2 "
    + str("{0:+.2f}".format(coeff_quad[1]))
    + "x "
    + str("{0:+.2f}".format(coeff_quad[2]))
)

plt.plot(
    mass[0:50],
    best_fit_quad(mass)[0:50],
    linewidth=0.75,
    color="navy",
    label=quad_label,
)
# plt.plot(
#     mass[0:55],
#     best_fit_lin(mass)[0:55],
#     linewidth=0.5,
#     color="gray",
#     linestyle="dashed",
#     label="Linear Comparison",
# )

plt.title("Variation of wander with planet mass")
plt.xlabel("Planet Mass /Solar Sasses")
plt.ylabel("Wander /AU")
plt.legend()
# plt.savefig("wanderwithplanetmass_p5e.png")
plt.show()
