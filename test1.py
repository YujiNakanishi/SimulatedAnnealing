import numpy as np
import SimulatedAnnealing as SA
from SimulatedAnnealing.Sampling.realVector import Gauss
from SimulatedAnnealing import Temperature as SAT

def function(x):
    return ((1. - x[0])**2 + 100.*(x[1]-x[0])**2)

objective_function = function
sampler = Gauss()
x_init = np.zeros(2)

T = SAT.util.decideInitialTemperature(objective_function, sampler, x_init)