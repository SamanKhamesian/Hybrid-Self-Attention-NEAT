import numpy as np

import neat.parallel


class ParallelEvaluator(neat.parallel.ParallelEvaluator):
    def __init__(self, num_workers, cmaes, eval_function, timeout=None):
        super().__init__(num_workers, eval_function, timeout)
        self.cmaes = cmaes

    def __del__(self):
        self.pool.close()
        self.pool.join()

    def evaluate_cmaes_for_attention(self, best_genome, config):
        candidate_fitness, jobs = [], []
        pops = self.cmaes.get_population()

        for candidate_params in pops:
            jobs.append(self.pool.apply_async(self.eval_function, (best_genome, config, candidate_params)))

        for job in jobs:
            candidate_fitness.append(job.get(timeout=self.timeout))

        self.cmaes.evolve(np.array(candidate_fitness))

    def evaluate_cmaes_for_weights(self, config):
        jobs, candidate_fitness = [], []
        pops = self.cmaes.get_population()
        for candidate_params in pops:
            jobs.append(self.pool.apply_async(self.eval_function, (config, candidate_params, False)))

        for job in jobs:
            candidate_fitness.append(job.get(timeout=self.timeout))

        self.cmaes.evolve(np.array(candidate_fitness))
