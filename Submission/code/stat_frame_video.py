import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import orbits
from constants import M_P, M_S, ORBIT_NUM, PRECISION, G, R  # User defined constants
from constants import lagrange, omega, time_span  # Derived constants

# Set up orbits for each body
orbit_sun = orbits.solar_pos(time_span)
orbit_jupiter = orbits.planet_pos(time_span)
orbit_greeks = orbits.stationary_frame(
    (lagrange[0], lagrange[1], 0, -omega * lagrange[1], omega * lagrange[0], 0)
)
orbit_trojans = orbits.stationary_frame(
    (lagrange[0], -lagrange[1], 0, omega * lagrange[1], omega * lagrange[0], 0)
)
orbit_list = [orbit_sun, orbit_jupiter, orbit_greeks, orbit_trojans]
bodies = ["Sun", "Jupiter", "Greeks", "Trojans"]
N = len(bodies)
# Setting up these lists allows us to iterate over all bodies when animating this video
# Therefore, additional bodies can be added for comparison very easily

orbit_data = np.zeros(
    (len(orbit_trojans.t), 2, int(N))
)  # select time coordinate, spatial coordinate (x/y), and index of body in list 'bodies'


for i in range(N):
    if type(orbit_list[i]) is np.ndarray:  # Different format of orbit results
        orbit_data[:, 0, i] = orbit_list[i][0]
        orbit_data[:, 1, i] = orbit_list[i][1]
    else:  # Account for solve_ivp ouput format
        orbit_data[:, 0, i] = orbit_list[i].y[0, :]
        orbit_data[:, 1, i] = orbit_list[i].y[1, :]


fig = plt.figure(figsize=(6, 6))
ax = plt.axes(xlim=(-8, 8), ylim=(-8, 8))
ax.set_aspect(aspect=1)  # So that circular orbits will not be distorted
time_marker = ax.text(0.02, 0.05, "", transform=ax.transAxes)  # To display current time

# These lists must also be upadated if further points are added
markers = [12, 8, 2, 2]
colours = ["yellow", "red", "blue", "green"]
points = []
for i in range(N):
    point = ax.plot(
        [],
        [],
        label=bodies[i],
        marker="o",
        linestyle="none",
        markersize=markers[i],
        color=colours[i],
    )
    points.append(point)
points = np.ndarray.tolist(np.squeeze(points))
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=4)


def init():
    """ Iterates over all bodies to initialize the data """
    for line in points:
        line.set_data([], [])
    return points


def animate(t):
    """Takes in the frame point in time(t) as the parameter and creates a function dependant on t for each body """
    for i in range(N):
        points[i].set_data([orbit_data[t, 0, i], orbit_data[t, 1, i]])
    time_marker.set_text(str("{:.1f}".format(orbit_trojans.t[t])) + " years")
    # return (points, time_marker)
    return points + [time_marker]


anim = animation.FuncAnimation(
    fig,
    animate,
    init_func=init,
    frames=ORBIT_NUM * int(PRECISION),
    interval=0.5,
    blit=True,
    repeat=False,
)

plt.title("Asteroid orbit in static frame")
plt.xlabel("X distance/ AU")
plt.ylabel("Y distance/ AU")


writer = animation.FFMpegWriter(
    fps=50, metadata=dict(artist="Candidate 6952R"), bitrate=1800
)
anim.save("movie.mp4", writer=writer)
plt.show()
