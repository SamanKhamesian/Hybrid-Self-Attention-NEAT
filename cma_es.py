import cma
import numpy as np


class CMAEvolutionStrategy:
    def __init__(self, population_size, init_sigma, init_params):
        self.__module = cma.CMAEvolutionStrategy(x0=init_params,
                                                 sigma0=init_sigma,
                                                 inopts={'popsize': population_size,
                                                         'randn': np.random.randn,
                                                         }, )

    def get_population(self):
        return self.__module.ask()

    def get_current_parameters(self):
        return self.__module.result.xfavorite

    def get_best_result(self):
        return self.__module.result.xbest, self.__module.result.fbest

    def evolve(self, fitness):
        self.__module.tell(self.get_population(), -fitness)
