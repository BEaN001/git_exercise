import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from collections import Counter
import random
import numpy as np
from copy import deepcopy


seed = 1
random.seed(seed)
np.random.seed(seed)


def random_linear_small():
    return lambda x: (random.uniform(-10, 10)) * x +random.uniform(-10, 10)


def sigmoid(x):
    return 1 / (1 +np.exp(-x))


def tanh(x):
    return np.tanh(x)


def one_layer(input_x, nodes):
    test_x = input_x
    span_indices = sorted(random.sample(range(0, len(test_x)), len(nodes) - 1))
    span_indices = [0] + span_indices + [None]

    outputs = []
    for i, index in enumerate(span_indices[:-1]):
        now, next_ = index, span_indices[i + 1]
        sub_x = test_x[index:span_indices[i + 1]]
        sub_output = nodes[i](sub_x)
        outputs.append(sub_output)

    return np.concatenate(outputs)


def multiply_layers(input_x, layers):
    this_layers_output = one_layer(input_x, layers[0])

    if len(layers) == 1:
        return this_layers_output
    else:
        return multiply_layers(this_layers_output, layers[1:])


def is_connected(node1, node2, graph_define):
    for root, connects in graph_define.items():
        if node1 in root and node2 in list(connects):
            return True
    return False


def get_directed_connect(toplogical_order, node, graph_define):
    connected_node = []

    for i, n in enumerate(toplogical_order):
        begin, end = node, n

        if is_connected(begin, end, graph_define):
            connected_node.append(end)

    return connected_node


def get_color_from_visited_order(visit_order, index, graph, color_map):
    changed = visit_order[:index]
    before, after = color_map
    color_map = [after if c in changed else before for c in graph]
    return color_map


def toplogic(define_graph):
    inner_graph = deepcopy(define_graph)

    sorted_node = []

    while len(inner_graph) > 0:

        all_inputs = []
        all_outputs = []

        for n in inner_graph:
            all_inputs += inner_graph[n]
            all_outputs.append(n)

        all_inputs = set(all_inputs)
        all_outputs = set(all_outputs)

        need_remove = all_outputs - all_inputs  # which in all_inputs but not in all_outputs

        if len(need_remove) > 0:
            node = random.choice(list(need_remove))

            visited_next = [node]
            if len(inner_graph) == 1:  visited_next += inner_graph[node]

            inner_graph.pop(node)
            sorted_node += visited_next

            for _, links in inner_graph.items():
                if node in links: links.remove(node)
        else:
            break

    return sorted_node


if __name__ == "__main__":
    node_x, node_k1, node_b1 = 'x', 'k1', 'b1'
    node_k2, node_b2 = 'k2', 'b2'
    node_linear_01, node_linear_02, node_sigmoid = 'linear_01', 'linear_02', 'sigmoid'
    node_loss = 'loss'

    computing_graph = {  # represent model
        node_x: [node_linear_01],
        node_k1: [node_linear_01],
        node_b1: [node_linear_01],
        node_linear_01: [node_sigmoid],
        node_sigmoid: [node_linear_02],
        node_k2: [node_linear_02],
        node_b2: [node_linear_02],
        node_linear_02: [node_loss],
    }





