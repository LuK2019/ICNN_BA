import numpy as np

class random_generator:
    def __init__(self):
        pass

    def generate(self):
        pass

class random_generator_uniform(random_generator):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mean = (lower_bound + upper_bound)/2
    def generate(self):
        return np.random.uniform(self.lower_bound, self.upper_bound)