import numpy as np
import torch


class BaseAttentionRunnerModule:
    def __init__(self):
        self._layers = []

    @property
    def layers(self):
        return self._layers

    def set_params(self, params):
        offset = 0
        for index, layer in enumerate(self._layers):
            weights_to_set = {}
            weight_dict = layer.state_dict()
            for k in sorted(weight_dict.keys()):
                weight = weight_dict[k].numpy()
                weight_size = weight.size
                weights_to_set[k] = torch.from_numpy(
                    params[offset:(offset + weight_size)].reshape(weight.shape))
                offset += weight_size
            self._layers[index].load_state_dict(state_dict=weights_to_set)

    def get_params(self):
        params = []
        for layer in self._layers:
            weight_dict = layer.state_dict()
            for k in sorted(weight_dict.keys()):
                params.append(weight_dict[k].numpy().copy().ravel())
        return np.concatenate(params)

    def get_num_params_per_layer(self):
        num_params_per_layer = []
        for layer in self._layers:
            weight_dict = layer.state_dict()
            num_params = 0
            for k in sorted(weight_dict.keys()):
                weights = weight_dict[k].numpy()
                num_params += weights.size
            num_params_per_layer.append(num_params)
        return num_params_per_layer
