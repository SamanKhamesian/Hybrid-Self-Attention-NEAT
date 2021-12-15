class BaseTunerModule:
    def __init__(self, best_genome):
        self._genome = best_genome

        self._nodes = self._genome.nodes
        self._connections = self._genome.connections

        self._true_connections = dict()
        for key, connection in self._connections.items():
            if connection.enabled:
                self._true_connections[key] = connection

        self.cmaes_model = None

    def get_params(self):
        bias_list = []
        weight_list = []
        for _, node in self._nodes.items():
            bias_list.append(node.bias)

        for _, connection in self._connections.items():
            if connection.enabled:
                weight_list.append(connection.weight)

        return bias_list + weight_list

    def set_params(self, params):
        bias_list, weight_list = params[:len(self._nodes)], params[len(self._nodes):]

        for (key, bias) in zip(self._nodes, bias_list):
            self._nodes[key].bias = bias

        for (key, weight) in zip(self._true_connections, weight_list):
            self._true_connections[key].weight = weight

        self._connections.update(self._true_connections)

    def get_genome(self):
        return self._genome
