import numpy as np
from Ant import *
from Visualisations import *


class System:

    def __init__(self):

        self.grid = np.zeros((500,50)) # Grid of 50 width and 500 height initialized to all zeroes
        self.ants = []  # This stores all existing ants

        # Model parameters
        self.N = 10000  # Number of times to repeat
        self.rate = 0  # Rate at which an ant enters the system
        self.iteration = 0 # Current iteration
        self.reward_scores = {} # Maps iteration values to reward scores
        self.complete = False
