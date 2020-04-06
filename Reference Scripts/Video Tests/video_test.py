import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


plt.style.use("seaborn-pastel")


# We simply create a figure window with a single axis in the figure.
# Then we create our empty line object which is essentially the one to be modified in the animation.
# The line object will be populated with data later.
fig = plt.figure()
ax = plt.axes(xlim=(0, 4), ylim=(-2, 2))
(line,) = ax.plot([], [], lw=3)


def init():
    """ The init function initializes the data and also sets the axis limits."""
    line.set_data([], [])
    return (line,)


def animate(i):
    """Takes in the frame number(i) as the parameter and creates a function dependant on i 
    
    Returns a tuple of the plot objects which have been modified which 
    tells the animation framework what parts of the plot should be animated.
    """
    x = np.linspace(0, 4, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return (line,)


anim = FuncAnimation(fig, animate, init_func=init, frames=1000, interval=20, blit=True)
# create actual animation object, blit parameter ensures that only changed parts of plot are re-drawn

anim.save("sine_wave.gif", writer="imagemagick")
