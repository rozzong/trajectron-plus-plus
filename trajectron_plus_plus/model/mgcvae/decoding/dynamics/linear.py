from ._dynamics import DynamicalModel


class Linear(DynamicalModel):

    def initialize(self, *args, **kwargs):
        pass

    def integrate_samples(self, v, x):
        return v

    def integrate_distribution(self, v_dist, x):
        return v_dist
