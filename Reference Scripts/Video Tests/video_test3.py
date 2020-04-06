import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import sin, cos


def data_gen():
    gen_list = ([cos(t), sin(t)] for t in np.arange(0, 1000, 0.1))
    return gen_list


print(np.arange(0, 1000, 0.1))
print(data_gen())
print([cos(t), sin(t)] for t in np.arange(0, 1000, 0.1))


def init():
    ax.set_ylim(-6, 6)
    ax.set_xlim(-6, 6)
    return point


fig, ax = plt.subplots()
(point,) = ax.plot([0], [0], "go")
point.set_data(0, 0)


def run(data):

    t, y = data
    # xmin, xmax = ax.get_xlim()
    # ymin, ymax = ax.get_ylim()

    # if t >= xmax:
    #     ax.set_xlim(xmin, t + 1)
    #     ax.figure.canvas.draw()

    # if t <= xmin:
    #     ax.set_xlim(t - 1, xmax)
    #     ax.figure.canvas.draw()

    # if y >= ymax:
    #     ax.set_ylim(ymin, y + 1)
    #     ax.figure.canvas.draw()

    # if y <= ymin:
    #     ax.set_ylim(y - 1, ymax)
    #     ax.figure.canvas.draw()

    point.set_data(t, y)

    return point


ani = animation.FuncAnimation(fig, run, data_gen, init_func=init, interval=10)
plt.show()
