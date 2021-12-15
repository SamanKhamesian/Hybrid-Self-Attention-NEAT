import os
import pickle
import time

from experiment.runner import AttentionNEATModule
from base_tuner import BaseTunerModule
from cma_es import CMAEvolutionStrategy
from experiment.configs.config import *
from neat.nn.recurrent import RecurrentNetwork
from parallel import ParallelEvaluator
from utility import process_action


class TunerModule(BaseTunerModule):
    def __init__(self, best_genome):
        super(TunerModule, self).__init__(best_genome)

        self.best_fitness = []
        self.cmaes_model = CMAEvolutionStrategy(population_size=TunerConfig.POP_SIZE,
                                                init_sigma=TunerConfig.INIT_SIGMA,
                                                init_params=self.get_params())


def get_action(net, ob):
    top = runner.attention_model.get_output(ob)
    new_ob = runner.attention_model.normalize_patch_centers(top)
    new_ob = np.append(new_ob, [1.0])
    action = net.activate(new_ob)
    action = process_action(action)
    return action, top


def eval_fitness(config, candidate_params=None, test_mode=False):
    fitness = []
    if not test_mode:
        if candidate_params is None:
            candidate_params = tuner.cmaes_model.get_current_parameters()
        tuner.set_params(candidate_params)
    genome = tuner.get_genome()

    for _ in range(TunerConfig.TRIALS):
        net = RecurrentNetwork.create(genome, config)
        ob = env.reset()
        total_reward = 0
        step = 0
        done = False

        while not done:
            action, new_ob = get_action(net, ob)
            ob, reward, done, info = env.step(action)
            step += 1
            total_reward += reward

        fitness.append(total_reward)

    return np.array(fitness).mean()


def save_result():
    net = RecurrentNetwork.create(tuner.get_genome(), AttentionNEATConfig.NEAT_CONFIG)

    with open(BASE_DIR + 'tuned_net.pkl', 'wb') as net_output:
        pickle.dump(net, net_output, pickle.HIGHEST_PROTOCOL)

    with open(BASE_DIR + 'tuned_model.pkl', 'wb') as tuner_output:
        pickle.dump(tuner, tuner_output, pickle.HIGHEST_PROTOCOL)


def load(reset=True):
    with open(BASE_DIR + 'main_model.pkl', 'rb') as attention_neat_output:
        _runner = pickle.load(attention_neat_output)

    if reset or not os.path.isfile(BASE_DIR + 'tuned_model.pkl'):
        return TunerModule(_runner.population.best_genome), _runner
    else:
        with open(BASE_DIR + 'tuned_model.pkl', 'rb') as tuner_output:
            return pickle.load(tuner_output), _runner


def test():
    score_list, time_list = [], []
    for i in range(TunerConfig.TEST):
        start = time.time()
        bc, _ = tuner.cmaes_model.get_best_result()
        score = eval_fitness(AttentionNEATConfig.NEAT_CONFIG, bc, False)
        end = time.time()

        print('\n#################### Test Result #####################\n')
        print('Reward: {0}'.format(score))
        print('Execution time: {0:.3f} sec'.format(end - start))
        print('\n######################################################\n')
        score_list.append(score)
        time_list.append(end - start)

    print('Mean Execution time: {0:.3f} sec'.format(np.array(time_list).mean()))
    print('Mean Test Fitness: {0:.3f}'.format(np.array(score_list).mean()))


def run():
    best_fitness = tuner.best_fitness
    parallel_runner = ParallelEvaluator(CPU_COUNT,
                                        tuner.cmaes_model,
                                        eval_fitness, None)

    for g in range(TunerConfig.GENERATIONS):
        start = time.time()
        parallel_runner.evaluate_cmaes_for_weights(AttentionNEATConfig.NEAT_CONFIG)
        end = time.time()
        bc, bf = tuner.cmaes_model.get_best_result()
        best_fitness.append(-bf)
        tuner.set_params(bc)
        print('\n################### Generations {0} ####################\n'.format(len(best_fitness)))
        print('Reward: {0}'.format(-bf))
        print('Execution time: {0:.3f} sec'.format(end - start))
        print('\n######################################################\n')

        save_result()


if __name__ == '__main__':
    tuner, runner = load(False)
    run()
