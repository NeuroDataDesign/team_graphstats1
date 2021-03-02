# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from typing import Union
from networkx.classes.digraph import DiGraph
import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu
import networkx as nx

_VALID_TESTS = ["fisher", "mww"]


def homotopic_test(
    graph: Union[nx.Graph, nx.DiGraph, np.ndarray],
    test: str,
) -> float:
    """
    Returns test of homotopic affinity in a graph.
    """

    '''
    if not isinstance(graph, Union[nx.Graph, nx.DiGraph, np.ndarray]):
        msg = "Graph must be a nxGraph or ndarray, not {}".format(type(graph))
        raise TypeError(msg)

    if not isinstance(test, str):
        msg = "Test must be a str, not {}".format(type(test))
        raise TypeError(msg)
    elif test not in _VALID_TESTS:
        msg = "Unknown test {}. Valid tests are {}".format(test, _VALID_TESTS)
        raise ValueError(msg)

    if type(graph) in [nx.Graph, nx.DiGraph]:
        graph = nx.to_numpy_array(graph)

    '''

    m, m = graph.shape
    graph_size = m**2
    comm_size = int(0.5*m)

    c1_c2 = graph[:comm_size, comm_size:]
    c2_c1 = graph[comm_size:, :comm_size]

    c1_bilateral = np.diag(c1_c2)
    c2_bilateral = np.diag(c2_c1)

    c1_homotopic = len(c1_bilateral[c1_bilateral != 0])
    c2_homotopic = len(c2_bilateral[c2_bilateral != 0])

    homotopic = c1_homotopic + c2_homotopic
    heterotopic = len(graph[graph != 0]) - homotopic

    # Convert to probabilities
    homotopic /= (len(c1_bilateral) + len(c2_bilateral))
    heterotopic /= graph_size - (len(c1_bilateral) + len(c2_bilateral))

    # Contingency Table
    con_table = np.zeros((2, 2))

    con_table[0, 0] = 1 - homotopic
    con_table[1, 0] = 1 - heterotopic
    con_table[0, 1] = homotopic
    con_table[1, 1] = heterotopic

    if test == "fisher":
        _, pvalue = fisher_exact(con_table)

        return con_table, pvalue

    else:
        statistic, pvalue = mannwhitneyu(con_table)

        return con_table, statistic, pvalue


def homophilic_test(
    graph: Union[nx.Graph, nx.DiGraph],
    test: str,
) -> float:
    """
    Returns test of homophilia in a graph.
    """

    '''
    if not isinstance(graph, Union[nx.Graph, nx.DiGraph, np.ndarray]):
        msg = "Graph must be a nxGraph, not {}".format(type(graph))
        raise TypeError(msg)

    if not isinstance(test, str):
        msg = "Test must be a str, not {}".format(type(test))
        raise TypeError(msg)
    elif test not in _VALID_TESTS:
        msg = "Unknown test {}. Valid tests are {}".format(test, _VALID_TESTS)
        raise ValueError(msg)
    '''
    
    #graph_array = nx.to_numpy_array(graph)

    m, m = graph.shape
    comm_size = int(0.5*m)
    comm_edges = comm_size**2

    con_table = np.zeros((2, 2))

    c_1 = graph[:comm_size, :comm_size]
    c_2 = graph[comm_size:, comm_size:]
    c_1_c_2 = graph[:comm_size, comm_size:]
    c_2_c_1 = graph[comm_size:, :comm_size]

    c_1_edges = len(c_1[c_1 != 0])
    c_2_edges = len(c_2[c_2 != 0])
    c_1_c_2_edges = len(c_1_c_2[c_1_c_2 != 0])
    c_2_c_1_edges = len(c_2_c_1[c_2_c_1 != 0])

    edges_within = (c_1_edges + c_2_edges) / (2*comm_edges)
    edges_between = (c_1_c_2_edges + c_2_c_1_edges) / (2*comm_edges)

    con_table[0, 0] = 1 - edges_within
    con_table[1, 0] = 1 - edges_between
    con_table[0, 1] = edges_within
    con_table[1, 1] = edges_between

    if test == "fisher":
        _, pvalue = fisher_exact(con_table)

        return con_table, pvalue

    else:
        statistic, pvalue = mannwhitneyu(con_table)

        return con_table, statistic, pvalue
