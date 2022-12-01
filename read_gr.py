import struct
from scipy.sparse import csr_matrix
import networkx as nx
import matplotlib.pyplot as plt

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
    return graph_csr


if __name__ == '__main__':
    graph_csr = get_csr_from_gr("./al2010.gr")
    graph_nx = nx.from_scipy_sparse_matrix(graph_csr, create_using=nx.DiGraph)
    print(nx.density(graph_nx))
    sssp = nx.shortest_path_length(graph_nx, source=0, weight='weight')
    print(sssp)
