# Hybrid-Self-Attention-NEAT

### Abstract

This repository contains the code to reproduce the results presented in the original [paper](https://arxiv.org/abs/2112.03670). <br/>
In this article, we present a “Hybrid Self-Attention NEAT” method to improve the original NeuroEvolution of Augmenting Topologies (NEAT) algorithm in high-dimensional inputs. Although the NEAT algorithm has shown a significant result in different challenging tasks, as input representations are high dimensional, it cannot create a well-tuned network. Our study addresses this limitation by using self-attention as an indirect encoding method to select the most important parts of the input. In addition, we improve its overall performance with the help of a hybrid method to evolve the final network weights. The main conclusion is that Hybrid Self-Attention NEAT can eliminate the restriction of the original NEAT. The results indicate that in comparison with evolutionary algorithms, our model can get comparable scores in Atari games with raw pixels input with a much lower number of parameters.

NOTE: The original implementation of self-attention for atari-games, and the NEAT algorithm can be found here:<br/>
Neuroevolution of Self-Interpretable Agents: https://github.com/google/brain-tokyo-workshop/tree/master/AttentionAgent <br/>
Pure python library for the NEAT and other variations: https://github.com/ukuleleplayer/pureples

### Execution

#### To use this work on your researches or projects you need:
* Python 3.7
* Python packages listed in `requirements.txt`

_NOTE: The following commands are based on Ubuntu 20.04_
###

#### To install Python:
_First, check if you already have it installed or not_.
~~~~
python3 --version
~~~~
_If you don't have python 3.7 in your computer you can use the code below_:
~~~~
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.7
sudo apt install python3.7-distutils
~~~~
###

_NOTE: To create a virtual environment, you can use the following link:_
<br/> Creation of virtual environment: https://docs.python.org/3.7/library/venv.html

#### To install packages via pip install:
~~~~
python3.7 -m pip install -r requirements.txt
~~~~
###

#### To run this project on Ubuntu server:
_You need to uncomment the following lines in_ `experiments/configs/configs.py`
~~~~
_display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
_display.start()
~~~~

_And also install some system dependencies as well_
~~~~
apt-get install -y xvfb x11-utils
~~~~
###

#### To train the model:
* First, check the configuration you need. The default ones are listed in `experiments/configs/`.
* We highly recommend increasing the number of population size, and the number of iterations to get better results.
* Check the working directory to be: `~/Hybrid_Self_Attention_NEAT/`
* Run the `runner.py` as below:
~~~~
python3.7 -m experiment.runner
~~~~
_NOTE: If you have limited resources (like RAM), you should decrease the number of iterations and instead use loops command_
~~~~
for i in {1..<n>}; do python3.7 -m experiment.runner; done
~~~~
###

#### To tune the model:
* First, check you trained the model, and the model successfully saved in `experiments/` as `main_model.pkl`
* Run the `tunner.py` as below:
~~~~
python3.7 -m experiment.tunner
~~~~
_NOTE: If you have limited resources (like RAM), you should decrease the number of iterations and instead use loops command_
~~~~
for i in {1..<n>}; do python3.7 -m experiment.tunner; done
~~~~

### Citation

#### For attribution in academic contexts, please cite this work as:
~~~~
@misc{khamesian2021hybrid,
    title           = {Hybrid Self-Attention NEAT: A novel evolutionary approach to improve the NEAT algorithm}, 
    author          = {Saman Khamesian and Hamed Malek},
    year            = {2021},
    eprint          = {2112.03670},
    archivePrefix   = {arXiv},
    primaryClass    = {cs.NE}
}
~~~~
