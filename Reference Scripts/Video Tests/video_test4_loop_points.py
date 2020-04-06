from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

fig = plt.figure()

ax = plt.axes(xlim=(-10, 10), ylim=(0, 100))

N = 4
points2 = ax.plot(*([[], []] * N), marker="o")  # old version


points = []
for i in range(N):
    point = ax.plot([], [], label=i, marker="o")
    points.append(point)
points = np.ndarray.tolist(np.squeeze(points))
ax.legend(loc=4)


def init():
    for line in points:
        line.set_data([], [])
    return points


def animate(i):
    points[0].set_data([0], [i])
    points[1].set_data([[1], [i + 1]])
    points[2].set_data([[2], [i + 2]])
    points[3].set_data([[3], [i + 3]])
    return points


anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=100, interval=20, blit=True
)

plt.show()
