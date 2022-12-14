/*
 csr_graph.h

 Implements a CSR Graph. Part of the GGC source code.
 Interface derived from LonestarGPU.

 Copyright (C) 2014--2016, The University of Texas at Austin

 See LICENSE.TXT for copyright license.

 Author: Sreepathi Pai <sreepai@ices.utexas.edu>
 */

#ifndef LSG_CSR_GRAPH
#define LSG_CSR_GRAPH

#include <cstdint>
#include <fstream>
#include <stdint.h>
#include <float.h>
typedef unsigned index_type; // should be size_t, but GPU chokes on size_t
typedef int edge_data_type;
typedef int node_data_type;
#define INF (INT32_MAX)

// very simple implementation
struct CSRGraph {
	unsigned read(char file[]);
	void copy_to_gpu(struct CSRGraph &copygraph);
	void copy_to_cpu(struct CSRGraph &copygraph);

	void init_from_array_to_gpu(uint64_t num_nodes, uint64_t num_edges, index_type *row_ptr, index_type *column, edge_data_type *edgeData);
	void copy_result_to_numpy(edge_data_type *result);

	CSRGraph();

	unsigned init();
	unsigned allocOnHost();
	unsigned allocOnDevice();
	void progressPrint(unsigned maxii, unsigned ii);
	unsigned readFromGR(char file[]);

	unsigned deallocOnHost();
	unsigned deallocOnDevice();
	void dealloc();

	__device__ __host__ bool valid_node(index_type node) {
		return (node < nnodes);
	}

	__device__ __host__ bool valid_edge(index_type edge) {
		return (edge < nedges);
	}

	__device__  __host__ index_type getOutDegree(unsigned src) {
		return row_start[src + 1] - row_start[src];
	}
	;

	__device__  __host__ index_type getDestination(unsigned src, unsigned edge) {
		index_type abs_edge = row_start[src] + edge;

		return edge_dst[abs_edge];
	}
	;

	__device__  __host__ index_type getAbsDestination(unsigned abs_edge) {

		return edge_dst[abs_edge];
	}
	;

	__device__  __host__ index_type getFirstEdge(unsigned src) {
		return row_start[src];
	}
	;

	__device__  __host__ edge_data_type getWeight(unsigned src, unsigned edge) {

		index_type abs_edge = row_start[src] + edge;

		return edge_data[abs_edge];

	}
	;

	__device__  __host__ edge_data_type getAbsWeight(unsigned abs_edge) {

		return edge_data[abs_edge];

	}
	;

	index_type nnodes, nedges;
	index_type *row_start; // row_start[node] points into edge_dst, node starts at 0, row_start[nnodes] = nedges
	index_type *edge_dst;
	edge_data_type *edge_data;
	node_data_type *node_data;
	bool device_graph;
	char file_name[256];

};

struct CSRGraphTex: CSRGraph {
	cudaTextureObject_t edge_dst_tx;
	cudaTextureObject_t row_start_tx;
	cudaTextureObject_t node_data_tx;

	void copy_to_gpu(struct CSRGraphTex &copygraph);
	unsigned allocOnDevice();

	__device__  __host__ index_type getOutDegree(unsigned src) {
//#ifdef __CUDA_ARCH__
#if 0
		return tex1Dfetch<index_type>(row_start_tx, src+1) -
		tex1Dfetch<index_type>(row_start_tx, src);
#else
		return CSRGraph::getOutDegree(src);
#endif 
	}
	;

	__device__ node_data_type node_data_ro(index_type node) {
		return tex1Dfetch < node_data_type > (node_data_tx, node);
	}

	__device__  __host__ index_type getDestination(unsigned src, unsigned edge) {
//#ifdef __CUDA_ARCH__
#if 0

		index_type abs_edge = tex1Dfetch<index_type>(row_start_tx, src + edge);

		return tex1Dfetch<index_type>(edge_dst_tx, abs_edge);
#else
		return CSRGraph::getDestination(src, edge);
#endif 

	}
	;

	__device__  __host__ index_type getAbsDestination(unsigned abs_edge) {
//#ifdef __CUDA_ARCH__
#if 0

		return tex1Dfetch<index_type>(edge_dst_tx, abs_edge);
#else
		return CSRGraph::getAbsDestination(abs_edge);
#endif 
	}
	;

	__device__  __host__ index_type getFirstEdge(unsigned src) {
//#ifdef __CUDA_ARCH__
#if 0
		return tex1Dfetch<index_type>(row_start_tx, src);
#else
		return CSRGraph::getFirstEdge(src);
#endif 
	}
	;
};

#ifdef CSRG_TEX
typedef CSRGraphTex CSRGraphTy;
#else
typedef CSRGraph CSRGraphTy;
#endif

#endif
