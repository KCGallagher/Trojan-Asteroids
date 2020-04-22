import numpy as np
import matplotlib.pyplot as plt
import timeit
import scipy.signal

import orbits
from constants import G, M_P, M_S, ORBIT_NUM, PRECISION, G, R  # User defined constants
from constants import (
    solar_rad,
    planet_rad,
    period,
    omega,
    greek_theta,
    time_span,
    lagrange,
)  # Derived constants

# ROTATING FRAME

time_repetitions = 100
time = timeit.timeit(
    "orbits.rotating_frame()", globals=globals(), number=time_repetitions
)
print(
    "Evaluated ODE system in "
    + str("{0:.4g}".format(time / time_repetitions))
    + " seconds"
)

orbit_values = orbits.rotating_frame()
wander = np.zeros((len(orbit_values.t)))

for i in range(len(orbit_values.t)):
    wander[i] = np.linalg.norm(
        (orbit_values.y[0:3, i] - lagrange[0:3])
    )  # magnitude of distance from lagrange point

print(
    "Maximum wander from the Lagrange point in the rotating frame over "
    + str("{0:.0f}".format(ORBIT_NUM * period))
    + " years: "
    + str("{0:.3g}".format(np.max(wander)))
    + " AU"
)

# # STATIONARY FRAME

orbit_values = orbits.stationary_frame()
wander = np.zeros((len(orbit_values.t)))

for i in range(len(orbit_values.t)):
    wander[i] = np.linalg.norm(
        (orbit_values.y[0:3, i] - orbits.lagrange_pos(orbit_values.t[i])[0:3])
    )  # magnitude of distance from lagrange point

print(
    "Maximum wander from the Lagrange point in the stationary frame over "
    + str("{0:.0f}".format(ORBIT_NUM * period))
    + " years: "
    + str("{0:.3g}".format(np.max(wander)))
    + " AU"
)

plt.plot(time_span, wander, linewidth=0.8)
plt.title("Wander of Greeks from Unperturbed State")
plt.ylabel("Deviation of Greeks from Lagrange Point /AU")
plt.xlabel("Time /years")
plt.savefig("greeks_deviation_stationary_frame.png")
plt.show()

sampling_rate = PRECISION / period
freqs = np.fft.fftfreq(len(wander)) * sampling_rate
fourier = np.abs(np.fft.fft(wander))

plt.plot(freqs, fourier)
plt.title("Fourier Transform of Wander")
plt.ylabel("'Frequency Domain (Spectrum) Magnitude'")
plt.xlabel("Frequency (1/year)")
plt.xlim(0, sampling_rate / 200)  # select low freq region of data
plt.show()

peaks, properties = scipy.signal.find_peaks(
    fourier, prominence=(5, 200), width=(0, 1000)
)
peak_freq = np.zeros(len(peaks))
for i in range(len(peaks)):
    peak_freq[i] = 1 / freqs[peaks[i]]
print("Period (in years) of primary frequency components in Fourier spectrum: ")
print(np.unique(np.abs(peak_freq)).round(2))
# Evaluated for 100 points per orbit, 100 orbits

# ENERGY CONSERVATION

energy = np.zeros(len(time_span))
for i in range(len(time_span)):
    energy[i] = orbits.specific_energy(time_span[i], orbit_values.y[:, i])

plt.plot(time_span, energy[:] - energy[0], linewidth=0.5)
plt.title("Energy Conservation in the Stationary Frame")
plt.xlabel("Time /years")
plt.ylabel("Specific Energy offset from initial value(J/kg)")
plt.show()

max_error = 100 * np.max(np.abs((energy - energy[0]) / energy[0]))
print(
    "Maximum deviation in asteroid energy from initial value: "
    + str("{0:.3f}".format(max_error))
    + " %"
)
