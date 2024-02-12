import numpy as np
from .. import SimulatedAnnealing as SA


def function(x):
    return ((1. - x[0])**2 + 100.*(x[1]-x[0])**2)