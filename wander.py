import numpy as np

import orbits
from constants import omega, lagrange, greek_theta  # Derived constants

# Initial Conditions
initial_cond_rot = np.array((lagrange[0], lagrange[1], 0, 0, 0, 0))  # in rotating frame
initial_cond_stat = np.array(
    (lagrange[0], lagrange[1], 0, -omega * lagrange[1], omega * lagrange[0], 0)
)


# Defined Functions
def perturb(initial_cond, max_pertubation_size=0.01):
    """ Returns perturbed version of initial conditions array

    init_cond is the initial conditions, submitted as a numpy array
    pertubation_size is the relative magnitude of the perturbation, measured as 
    the percentage deivation from the lagrange point (for position)
    Note that this will give a different value each time it is run

    Perturbations are chosen randomly within a circle of radius one, 
    to give a more uniform radial distribution than from a square.
    """
    rand_array = np.random.uniform(-1, 1, (np.shape(initial_cond)))
    while np.linalg.norm(rand_array[0:2]) > 1:
        rand_array = np.random.uniform(-1, 1, (np.shape(initial_cond)))
    return initial_cond + initial_cond * (
        max_pertubation_size * rand_array
    )  # random between -1 and 1, selects points in circle of radius 1


def rand_sample(max_pertubation_size, samples):
    """Returns maximum deviation over orbit in the rotating frame for given number of random samples

    This calculates the distange from the initial point (not the lagrange point)
    It returns a samples*3 array, giving the radial and tangential components
    of the pertubation, and then the maximum wander over this timespan

    pertubation_size determies the relative magnitude of the perturbation as in perturb()
    Samples denotes the number of random pertubations sampled for given perturbation size
    """
    output = np.zeros((samples, 3))  # size of pertubation; size of wander
    for n in range(samples):
        initial_cond = perturb(initial_cond_rot, max_pertubation_size)
        orbit = orbits.rotating_frame(initial_cond)
        sample_wander = np.zeros((len(orbit.t)))

        perturb_theta = np.arctan(
            (initial_cond[1] - initial_cond_rot[1])
            / (initial_cond[0] - initial_cond_rot[0])
        )

        for i in range(len(sample_wander)):
            sample_wander[i] = np.linalg.norm(
                orbit.y[0:3, i]
                - initial_cond[0:3]
                # - lagrange[0:3]
            )  # deviation in pos only
        output[n, 0] = np.linalg.norm((initial_cond[0:3] - lagrange[0:3])) * np.abs(
            np.cos(greek_theta - perturb_theta)
        )  # radial component of pertubation
        output[n, 1] = np.linalg.norm((initial_cond[0:3] - lagrange[0:3])) * np.abs(
            np.sin(greek_theta - perturb_theta)
        )  # tangential component of pertubation
        output[n, 2] = np.max(sample_wander)  # size of wander
    return output


def initial_point(pertubation, samples=1, pertubation_type="position"):
    """Returns maximum wander over orbit in the rotating frame for given initial point 

    
    Wander is the maximum deviation from the initial point (not the lagrange point) over this timespan
    perturbation is the initial point in 2D position/velocity space
    samples denotes the number of random pertubations sampled within that point for gives perturbation size
    perturbation_type can be "position" or "velocity" 
    """
    if pertubation_type == "position":
        initial_cond = initial_cond_rot + np.array(
            (pertubation[0], pertubation[1], pertubation[2], 0, 0, 0)
        )  # add position pertubation
    elif pertubation_type == "velocity":
        initial_cond = initial_cond_rot + np.array(
            (0, 0, 0, pertubation[0], pertubation[1], pertubation[2])
        )  # add velocity pertubation
    else:
        raise ValueError(
            "Unknown perturbation_type: Please use 'position' or 'velocity'"
        )

    orbit = orbits.rotating_frame(initial_cond)
    wander_t = np.zeros((len(orbit.t)))
    for i in range(len(orbit.t)):
        wander_t[i] = np.linalg.norm(
            orbit.y[0:3, i]
            - initial_cond[0:3]
            # - lagrange[0:3]
        )  # deviation in pos only
    return np.max(wander_t)
