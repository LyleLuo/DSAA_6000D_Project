import struct
from scipy.sparse import csr_matrix
import networkx as nx
import time
import os
import numpy as np
import adds

def read_gr(filename, fmt='I'):
    """Read a graph from a file in Galois .gr format."""
    infh = open(filename, 'rb')

    meta_bin = infh.read(8*4)
    version, sizeEdgeTy, numNodes, numEdges = struct.unpack("<4Q", meta_bin)

    outIdx = (0,) + struct.unpack("<{}Q".format(numNodes), infh.read(8*numNodes))
    outs = struct.unpack("<{}L".format(numEdges), infh.read(4*numEdges))

    if numEdges % 2 != 0:
        infh.read(4)

    edgeData = struct.unpack("<{}{}".format(numEdges, fmt) ,infh.read(4*numEdges))

    infh.close()

    return outIdx, outs, edgeData, numNodes, numEdges

def get_csr_from_gr(filename, fmt='I'):
    row_ptrs, column, value, num_nodes, num_edges = read_gr(filename, fmt)
    graph_csr = csr_matrix((value, column, row_ptrs), shape=(num_nodes, num_nodes))
    return num_nodes, num_edges, graph_csr



if __name__ == '__main__':

    # warm up
    num_nodes, num_edges, graph_csr = get_csr_from_gr("./ads_int/input/rmat22.gr")
    result = adds.sssp_from_csr(num_nodes, num_edges, graph_csr.indptr, graph_csr.indices, graph_csr.data)


    time_adds, time_nx_dj, time_nx_bf = [], [], []
    for filename in os.listdir('./ads_int/input'):
        num_nodes, num_edges, graph_csr = get_csr_from_gr(os.path.join('./ads_int/input', filename))
        print(filename)
        adds_start_time = time.time()
        result = adds.sssp_from_csr(num_nodes, num_edges, graph_csr.indptr, graph_csr.indices, graph_csr.data)
        # print(result)
        adds_end_time = time.time()
        print('adds time: ', adds_end_time - adds_start_time)
        time_adds.append(adds_end_time - adds_start_time)

        nx_dj_start_time = time.time()
        graph_nx = nx.from_scipy_sparse_array(graph_csr, create_using=nx.DiGraph)
        # print(nx.density(graph_nx))
        sssp_dj = nx.shortest_path_length(graph_nx, source=0, weight='weight')
        nx_dj_end_time = time.time()
        print('NetworkX Dijkstra time: ', nx_dj_end_time - nx_dj_start_time)
        time_nx_dj.append(nx_dj_end_time - nx_dj_start_time)


        # nx_bf_start_time = time.time()
        # graph_nx = nx.from_scipy_sparse_array(graph_csr, create_using=nx.DiGraph)
        # sssp_bf = nx.shortest_path_length(graph_nx, source=0, weight='weight', method='bellman-ford')
        # nx_bf_end_time = time.time()
        # print('NetworkX BF time: ', nx_bf_end_time - nx_bf_start_time)
        # time_nx_bf.append(nx_bf_end_time - nx_bf_start_time)

        time_compare_dj, time_compare_bf = [], []
        for i in range(len(time_adds)):
            time_compare_dj.append(time_nx_dj[i]/time_adds[i])
            # time_compare_bf.append(time_nx_bf[i]/time_adds[i])
        print(' ')

    print('adds vs dj: ', np.mean(time_compare_dj))
    # print('adds vs bf: ', np.mean(time_compare_bf))

    np_dj = np.array(time_compare_dj)
    np.save('adds_vs_dj.npy', np_dj)

    # np_bf = np.array(time_compare_bf)
    # np.save('adds_vs_bf.npy', np_bf)
