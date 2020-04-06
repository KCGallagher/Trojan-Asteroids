import matplotlib


from matplotlib import pyplot as plt
from matplotlib import animation

fig = plt.figure()

ax = plt.axes(xlim=(0, 2), ylim=(0, 100))

N = 4
points = [plt.plot() for _ in range(N)]  # points to animate


def init():
    # init points
    for point in points:
        point.set_data()
    return points


def animate(i):
    # animate points
    for j, point in enumerate(points):
        point.set_data(10 * j, i)

    return points


anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=100, interval=20, blit=True
)

plt.show()
