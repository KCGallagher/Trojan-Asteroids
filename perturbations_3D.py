import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

import orbits
import wander
from constants import ORBIT_NUM, PRECISION, R  # User defined constants
from constants import (
    solar_rad,
    period,
    time_span,
    lagrange,
)  # Derived constants

greek_theta = np.arctan((R * math.sqrt(3) / 2) / (R / 2 - solar_rad))
cos = math.cos(greek_theta)
sin = math.sin(greek_theta)

initial_cond_rot = np.array((lagrange[0], lagrange[1], 0, 0, 0, 0))


# # DEVIATION FROM LAGRANGE POINT
perturb_z = np.array((0, 0, 0.2, 0, 0, 0))
orbit_sol_z = orbits.rotating_frame(initial_cond_rot + perturb_z)

plt.plot(time_span, orbit_sol_z.y[2, :])
plt.ylabel("Magnitude of Deviation in z-direction /AU")
plt.xlabel("Time /years")
plt.title("Asteroid Deviation from Lagrange point")
# plt.savefig("asteroid_zdeviation_linear.png")
plt.show()


sampling_rate = PRECISION / period
freqs = np.fft.fftfreq(len(orbit_sol_z.y[2, :])) * sampling_rate
fourier = np.abs(np.fft.fft(orbit_sol_z.y[2, :]))

peaks, properties = scipy.signal.find_peaks(fourier, prominence=(100), width=(0,))

peak_freq = np.zeros(len(peaks))
for i in range(len(peaks)):
    peak_freq[i] = 1 / freqs[peaks[i]]
print("Period (in years) of primary frequency components in Fourier spectrum: ")
print(np.unique(np.abs(peak_freq)).round(2))
# Evaluated for 100 points per orbit, 100 orbits


# WANDER PLOTS

z_perturbations = np.linspace(0, 0.5, 30)
wander_data = np.zeros(
    (len(z_perturbations), 2)
)  # 2 columns for perturbations with and without xy plane
for n in range(len(z_perturbations)):
    perturbation_z = np.array((0, 0, z_perturbations[n], 0, 0, 0))  # no xy components
    perturbation_xyz = np.array(
        (0.002 * cos, 0.002 * sin, z_perturbations[n], 0, 0, 0)
    )  # radial xy components
    wander_data[n, 0] = wander.initial_point(perturbation_z, "position")
    wander_data[n, 1] = wander.initial_point(perturbation_xyz, "position")
    print(n)  # tracking

fig = plt.figure()
ax = fig.add_subplot()

coeff = np.polyfit(
    z_perturbations, wander_data[:, 0], 2
)  # for quadratic line of best fit
best_fit = np.poly1d(coeff)  # best fit line plotting function
order = np.argsort(z_perturbations)  # order arguments for plotting

quad_label = str(
    "Best Fit: "
    + str("{0:.2f}".format(coeff[0]))
    + "x\u00b2 "
    + str("{0:+.2f}".format(coeff[1]))
    + "x "
    + str("{0:+.2f}".format(coeff[2]))
)

plt.plot(z_perturbations, best_fit(z_perturbations), linewidth=0.75, label=quad_label)


coeff2 = np.polyfit(
    z_perturbations, wander_data[:, 1], 2
)  # for quadratic line of best fit
best_fit2 = np.poly1d(coeff2)
order = np.argsort(z_perturbations)

quad_label2 = str(
    "Best Fit: "
    + str("{0:.2f}".format(coeff2[0]))
    + "x\u00b2 "
    + str("{0:+.2f}".format(coeff2[1]))
    + "x "
    + str("{0:+.2f}".format(coeff2[2]))
)

plt.plot(
    z_perturbations,
    best_fit2(z_perturbations),
    linewidth=0.75,
    label=quad_label2,
    color="pink",
)

plt.plot(
    z_perturbations,
    wander_data[:, 0],
    label="No Perturbation",
    linestyle="None",
    marker="+",
    color="navy",
)


plt.plot(
    z_perturbations,
    wander_data[:, 1],
    label="0.002 AU Perturbation",
    linestyle="None",
    marker="+",
    color="purple",
)

plt.title("Wander under Z Perturbations")
plt.xlabel("Z Perturbation /AU")
plt.ylabel("Magnitude of Wander /AU")
plt.legend()
# plt.savefig("z_perturbations5c.png")
plt.show()


# # MOTION IN ROTATING FRAME
plt.plot(orbit_sol_z.y[0, :], orbit_sol_z.y[1, :], label="Greek Orbit")
plt.plot(
    lagrange[0],
    lagrange[1],
    label="Lagrange Point",
    linestyle="",
    marker="+",
    color="black",
)
plt.ylabel("Y Position /AU")
plt.xlabel("X Position /AU")
plt.title("Orbit following Z Perturbation in the Rotating Frame")
plt.legend()
# plt.savefig("z_perturb_orbit.png")
plt.show()

# WANDER IN Z DIRECTION ONLY AGAINST PERTURBATION SIZE

z_perturbations = np.linspace(0, 0.5, 30)
wander_data = np.zeros(len(z_perturbations))
for n in range(len(z_perturbations)):
    orbit_solz = orbits.rotating_frame(
        initial_cond_rot
        + np.array((0.01 * cos, 0.01 * sin, z_perturbations[n], 0, 0, 0))
    )
    wander_z = np.zeros((len(orbit_solz.t)))
    for i in range(len(orbit_solz.t)):
        wander_z[i] = np.abs(
            orbit_solz.y[2, i] - z_perturbations[n]
        )  # deviation in z pos only
    wander_data[n] = np.max(wander_z)
    print(n)  # tracking

fig = plt.figure()
ax = fig.add_subplot()

coeff_lin = np.polyfit(
    z_perturbations, wander_data[:], 1
)  # for linear line of best fit
best_fit_lin = np.poly1d(coeff_lin)
order = np.argsort(z_perturbations)

lin_label = str(
    "Best Fit: "
    + str("{0:.2f}".format(coeff_lin[0]))
    + "x "
    + str("{0:+.2f}".format(coeff_lin[1]))
)  # Theoretical prediction is y = 2x

plt.plot(
    z_perturbations, best_fit_lin(z_perturbations), linewidth=0.75, label=lin_label,
)

plt.plot(
    z_perturbations,
    wander_data[:],
    label="Z component of Wander",
    linestyle="",
    marker="x",
    color="navy",
)

plt.ylabel("Max Wander /AU")
plt.xlabel("Z perturbation /AU")
plt.title("Wander under Z-perturbations")
# plt.savefig("zdirection_components.png")
plt.legend()
plt.show()
