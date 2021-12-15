import logging
import sys
import warnings

import gym
import numpy as np
import pyvirtualdisplay

import neat

# _display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
# _display.start()

if not sys.warnoptions:
    warnings.simplefilter("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
env = gym.make('Asteroids-v0')

np.set_printoptions(linewidth=100)

BASE_DIR = 'experiment/'
CPU_COUNT = 8


class AttentionNEATConfig:
    ACTIVATION = 'sigmoid'
    GENERATIONS = 10
    TEST = 10
    TRIALS = 3
    NEAT_CONFIG = neat.config.Config(neat.genome.DefaultGenome,
                                     neat.reproduction.DefaultReproduction,
                                     neat.species.DefaultSpeciesSet,
                                     neat.stagnation.DefaultStagnation,
                                     BASE_DIR + 'configs/config_attention_neat')


class SelfAttentionConfig:
    IMAGE_SHAPE = (210, 160, 3)
    PATCH_SIZE = 10
    PATCH_STRIDE = 5
    TRANSFORMER_D = 4
    TOP_K = 10


class CMAESConfig:
    POP_SIZE = 32
    INIT_SIGMA = 0.1


class TunerConfig:
    GENERATIONS = 10
    TEST = 10
    TRIALS = 3
    POP_SIZE = 32
    INIT_SIGMA = 0.1
