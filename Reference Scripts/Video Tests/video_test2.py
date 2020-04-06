import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Acc_11 = [1, 2, 3, 4, 6, 7]
Acc_12 = [2, 2, 2, 2, 2, 2]

# Scatter plot
fig = plt.figure(figsize=(5, 5))
axes = fig.add_subplot(111)
axes.set_xlim(min(Acc_11), max(Acc_11))
axes.set_ylim(min(Acc_12), max(Acc_12))

(point,) = axes.plot([Acc_11[0]], [Acc_12[0]], "go")


def ani(coords):
    point.set_data([coords[0]], [coords[1]])
    return point


def frames():
    for acc_11_pos, acc_12_pos in zip(Acc_11, Acc_12):
        yield acc_11_pos, acc_12_pos


ani = FuncAnimation(fig, ani, frames=frames, interval=1000)

plt.show()
