import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

import orbits
import wander
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


# DEVIATION FROM LAGRANGE POINT
plt.plot(time_span, orbit_solz.y[2, :])
# plt.yscale("log", basey=10)
plt.ticklabel_format(axis="y", style="", scilimits=None)
plt.ylabel("Magnitude of Deviation in z-direction /AU")
plt.xlabel("Time /years")
plt.title("Asteroid Deviation from Lagrange point")
# plt.savefig("asteroid_zdeviation_linear.png")
plt.show()


sampling_rate = PRECISION / period
freqs = np.fft.fftfreq(len(orbit_solz.y[2, :])) * sampling_rate
fourier = np.abs(np.fft.fft(orbit_solz.y[2, :]))

peaks, properties = scipy.signal.find_peaks(fourier, prominence=(100), width=(0,))

peak_freq = np.zeros(len(peaks))
for i in range(len(peaks)):
    peak_freq[i] = 1 / freqs[peaks[i]]
print("Period (in years) of primary frequency components in Fourier spectrum: ")
print(np.unique(np.abs(peak_freq)).round(2))
# Evaluated for 100 points per orbit, 100 orbits


# wander

z_perturbations = np.linspace(0, 0.2, 1)
wander_data = np.zeros((len(z_perturbations), 2))
for n in range(len(z_perturbations)):
    perturbation = np.array((0.01 * cos, 0.01 * sin, z_perturbations[n], 0, 0, 0))
    perturbation2 = np.array((0.0102 * cos, 0.0102 * sin, z_perturbations[n], 0, 0, 0))
    wander_data[n, 0] = wander.initial_point(perturbation, "position")
    wander_data[n, 1] = wander.initial_point(perturbation2, "position")
    print(n)

fig = plt.figure()
ax = fig.add_subplot()

coeff = np.polyfit(
    z_perturbations, wander_data[:, 0], 2
)  # for quadratic line of best fit
best_fit = np.poly1d(coeff)
order = np.argsort(z_perturbations)

quad_label = str(
    "Best Fit: "
    + str("{0:.3f}".format(coeff[0]))
    + "x\u00b2 "
    + str("{0:+.3f}".format(coeff[1]))
    + "x "
    + str("{0:+.3f}".format(coeff[2]))
)

plt.plot(z_perturbations, best_fit(z_perturbations), linewidth=0.75, label=quad_label)


coeff = np.polyfit(
    z_perturbations, wander_data[:, 1], 2
)  # for quadratic line of best fit
best_fit = np.poly1d(coeff)
order = np.argsort(z_perturbations)

quad_label = str(
    "Best Fit: "
    + str("{0:.3f}".format(coeff[0]))
    + "x\u00b2 "
    + str("{0:+.3f}".format(coeff[1]))
    + "x "
    + str("{0:+.3f}".format(coeff[2]))
)

plt.plot(
    z_perturbations,
    best_fit(z_perturbations),
    linewidth=0.75,
    label=quad_label,
    color="pink",
)

plt.plot(
    z_perturbations,
    wander_data[:, 0],
    label="0.01 AU Perturbation",
    linestyle="None",
    marker="+",
    color="navy",
)


plt.plot(
    z_perturbations,
    wander_data[:, 1],
    label="0.0102 AU Perturbation",
    linestyle="None",
    marker="+",
    color="purple",
)

# plt.plot(z_perturbations, np.sqrt(1 + z_perturbations ** 2), label="Prediction")
# plt.plot(
#     z_perturbations,
#     (wander_data[-1, 0] / np.sqrt(wander_data[0, 0] ** 2 + z_perturbations[-1] ** 2))
#     * np.sqrt(wander_data[0, 0] ** 2 + z_perturbations ** 2),
#     label="Prediction2",
# )


plt.title("Wander under Z-Perturbations")
plt.legend()
# plt.savefig("z_perturbations3.png")
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


z_perturbations = np.linspace(0, 0.1, 30)
wander_data = np.zeros((len(z_perturbations), 2))
for n in range(len(z_perturbations)):
    orbit_solz = orbits.rotating_frame(
        # initial_cond_rot + np.array((0, 0, z_perturbations[n], 0, 0, 0))
        initial_cond_rot
        + np.array((0.0 * cos, 0.0 * sin, z_perturbations[n], 0, 0, 0))
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


coeff = np.polyfit(
    z_perturbations, wander_data[:, 1], 2
)  # for quadratic line of best fit
best_fit = np.poly1d(coeff)
order = np.argsort(z_perturbations)

quad_label = str(
    "Best Fit: "
    + str("{0:.3f}".format(coeff[0]))
    + "x\u00b2 "
    + str("{0:+.3f}".format(coeff[1]))
    + "x "
    + str("{0:+.3f}".format(coeff[2]))
)

plt.plot(
    z_perturbations,
    best_fit(z_perturbations),
    linewidth=0.75,
    label=quad_label,
    color="pink",
)

plt.plot(z_perturbations, wander_data[:, 1], label="Z component")


plt.title("Wander under Z-perturbations")
plt.legend()
plt.show()
