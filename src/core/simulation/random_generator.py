import numpy as np


class RandomGenerator:
    def __init__(self):
        pass

    def generate(self):
        pass


class RandomGeneratorUniform(RandomGenerator):
    def __init__(self, lower_bound, upper_bound):
        """The constructor of the random_generator uniform
        Args:
            lower_bound: lower bound of the uniform distribution
            upper_bound: upper bound of the uniform distribution
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mean = (lower_bound + upper_bound) / 2

    def generate(self):
        """Random value between [self.lower_bound, self.upper_bound] is generated."""
        return np.random.uniform(self.lower_bound, self.upper_bound)


class RandomGeneratorNormal(RandomGenerator):
    def __init__(self, std, mean, lower_bound=None, upper_bound=None):
        """Constructor:
        Args:
            std: standard deviation of the normal distribution
            mean: mean of the normal distribution
            lower_bound, upper_bound: if the generated value lies outside the 
            boundaries, a new value is drawn till it suffices the boundary 
            requirement
        """
        self.std = std
        self.mean = mean
        self.lower_bound = None
        self.upper_bound = None

    def generate(self):
        """ Generate a random value from a normal distribution ~ (self.mean, self.std).
        If necessary, the return value is discarded if it does not lie within the 
        interval [self.lower_bound, self.upper_bound]
        """
        if (self.lower_bound is None) and (self.upper_bound is None):
            return np.random.normal(loc=self.mean, scale=self.std)
        else:
            raise NotImplemented(
                "Boundaries for normal distribution not implemented yet!"
            )
