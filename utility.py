import numpy as np

import neat


def initial_population(state, stats, config, output):
    population = neat.population.Population(config, [], [], state)
    if output:
        population.add_reporter(neat.reporting.StdOutReporter(True))
    population.add_reporter(stats)
    return population


def process_action(actions):
    action = np.argmax(actions)
    return action
