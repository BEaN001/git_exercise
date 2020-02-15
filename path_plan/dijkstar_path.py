from dijkstar import Graph, find_path


def cost_func(u, v, edge, prev_edge):
    length, name = edge
    if prev_edge:
     prev_name = prev_edge[1]
    else:
     prev_name = None
    cost = length
    if name != prev_name:
     cost += 10
    return cost


if __name__ == "__main__":
    #
    graph = Graph()
    graph.add_edge(1, 2, 110)
    graph.add_edge(2, 3, 125)
    graph.add_edge(3, 4, 108)

    print(find_path(graph, 1, 4))

    #
    graph = Graph()
    graph.add_edge(1, 2, (110, 'Main Street'))
    graph.add_edge(2, 3, (125, 'Main Street'))
    graph.add_edge(3, 4, (108, '1st Street'))

    print(find_path(graph, 1, 4, cost_func=cost_func))


