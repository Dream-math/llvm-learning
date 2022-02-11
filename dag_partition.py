'''
input graph 
                       1
                  /  |  \    \
                 /   3   \    \
                2    |    4    5
               / \   |   / \  /
              /   \  |  /   \/
             /       7       8
            /      /
           6 -----/
'''
init_graph = [[1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 2, 7], [7, 2, 3, 4], [8, 4, 5]]
init_output = [6, 7, 8]

def get_input(graph):
    input = []
    for node in graph:
        if len(node) == 1:
            input.append(node[0])
    return input

def get_list(graph, node):
    for it in graph:
        if it[0] == node:
            return it
    return None

def count_degree(graph):
    degrees = []
    for i in range(len(graph)):
        degrees.append([graph[i][0], 0])
    for it in graph:
        if len(it) > 1:
            for node in it:
                if node != it[0]:
                    degree = get_list(degrees, node)
                    degree[1] = degree[1] + 1
    return degrees



def deal(graph, output, graphs):
    # at last every node will be processed
    if not output:
        return

    input = get_input(graph)

    cross = []
    for i in range(len(graph)):
        cross.append([graph[i][0]])
    index = len(graph) + 1

    def visit(node, cross):
        cross_it = get_list(cross, node)
        if index not in cross_it:
            cross_it.append(index)
            # cross = update_list(cross, cross_it)
        if node not in input:
            graph_it = get_list(graph, node)
            for it in graph_it:
                if it != node:
                    cross_jt = get_list(cross, it)
                    if index not in cross_jt:
                        cross_jt.append(index)
                    visit(it, cross)


    def partition(node, subgraph):
        if node not in subgraph:
            subgraph.append(node)

        if node not in input:
            graph_it = get_list(graph, node)
            for it in graph_it:
                if it != node and it not in output:
                    degree = get_list(degrees, it)
                    cross_it = get_list(cross, it)
                    if degree[1] == 1 or len(cross_it) == 2:
                        partition(it, subgraph)

    def new_output(delete, graph):
        ne_output = []
        for node in delete:
            graph_it = get_list(graph, node)
            if len(graph_it) > 1:
                for jt in graph_it:
                    if jt != node and jt not in delete and jt not in ne_output:
                        ne_output.append(jt)
        return ne_output

    def new_graph(ne_output, graph):
        ne_graph = []
        def upper(ne_graph, node, graph):
            graph_it = get_list(graph, node)
            if graph_it not in ne_graph:
                ne_graph.append(graph_it)
            for jt in graph_it:
                if jt != node:
                    upper(ne_graph, jt, graph)

        for node in ne_output:
            upper(ne_graph, node, graph)
        return ne_graph




    for node in output:
        visit(node, cross)
        index = index + 1

    # print("cross information")
    # print(cross)
    degrees = count_degree(graph)
    # print("degree information")
    # print(degrees)

    delete = []
    for node in output:
        subgraph = []
        partition(node, subgraph)
        delete = delete + subgraph
        graphs.append(subgraph)

    # print("partition graphs")
    # print(graphs)
    # print("delete nodes")
    # print(delete)

    ne_output = new_output(delete, graph)
    # print("new output")
    # print(ne_output)

    ne_graph = new_graph(ne_output, graph)
    # print("new graph")
    # print(ne_graph)
    deal(ne_graph, ne_output, graphs)

graphs = []
deal(init_graph, init_output, graphs)
print(graphs)
