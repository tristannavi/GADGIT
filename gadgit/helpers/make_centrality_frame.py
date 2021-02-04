import pandas as pd
import networkx as nx

def make_centrality_frame(graph_path):
    """Take a simple graph text file returns an nx Graph and a DataFrame with two basic centrality measures: degree and betweenness.

    This function is to be used as a helper to get a user up and running.
    The input file is of the form:
        First line: number of nodes
        Subsequent lines which specify edges: N tab M

    Parameters
    -------
    graph_path: string
        Path to a text file containing a graph of the form specified above.

    Returns
    -------
    Frame: pandas DataFrame

    """

    # Below file handling is done line by line to avoid reading
    # entire file into memory. First line is only for node count
    nodes = -1
    edge_list = [] # List of pairs
    with open(graph_path, 'r') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                nodes = int(line)
            else:
                edge_list.append(tuple([int(x) for x in line.split('\t')]))

    # Build graph
    G = nx.Graph()
    G.add_nodes_from([0, nodes])
    G.add_edges_from(edge_list)

    # Pandas notation for grabbing 
    df = pd.DataFrame(dict(
        DEGREE = dict(G.degree),
        DEGREE_CENTRALITY = nx.degree_centrality(G),
        BETWEENNESS_CENTRALITY = nx.betweenness_centrality(G),
    ))
    # Quality of life things
    df.index.names = ['NODE']
    df = df.sort_index()

    return G, df
