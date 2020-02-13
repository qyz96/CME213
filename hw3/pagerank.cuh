#ifndef _PAGERANK_CUH
#define _PAGERANK_CUH

#include "util.cuh"

/* 
 * Each kernel handles the update of one pagerank score. In other
 * words, each kernel handles one row of the update:
 *
 *      pi(t+1) = (1 / 2) A pi(t) + (1 / (2N))
 *
 * You may assume that num_nodes <= blockDim.x * 65535
 *
 */
__global__ void device_graph_propagate(
    const uint *graph_indices,
    const uint *graph_edges,
    const float *graph_nodes_in,
    float *graph_nodes_out,
    const float *inv_edges_per_node,
    int num_nodes
) {

    const uint i = (uint)(blockIdx.x * blockDim.x + threadIdx.x);
    float sum=0;
    if (i<num_nodes) {
        for (uint k=graph_indices[i]; k<graph_indices[k+1]; k++) {
            sum+=graph_nodes_in[graph_edges[k]];
        }
        sum*=inv_edges_per_node[i];
        graph_nodes_out[i]=0.5*sum+1/(2*(float)(num_nodes));
    }

    // TODO: fill in the kernel code here
}

/* 
 * This function executes a specified number of iterations of the
 * pagerank algorithm. The variables are:
 *
 * h_graph_indices, h_graph_edges:
 *     These arrays describe the indices of the neighbors of node i.
 *     Specifically, node i is adjacent to all nodes in the range
 *     h_graph_edges[h_graph_indices[i] ... h_graph_indices[i+1]].
 *
 * h_node_values_input:
 *     An initial guess of pi(0).
 *
 * h_gpu_node_values_output:
 *     Output array for the pagerank vector.
 *
 * h_inv_edges_per_node:
 *     The i'th element in this array is the reciprocal of the
 *     out degree of the i'th node.
 *
 * nr_iterations:
 *     The number of iterations to run the pagerank algorithm for.
 *
 * num_nodes:
 *     The number of nodes in the whole graph (ie N).
 *
 * avg_edges:
 *     The average number of edges in the graph. You are guaranteed
 *     that the whole graph has num_nodes * avg_edges edges.
 */
double device_graph_iterate(
    const uint *h_graph_indices,
    const uint *h_graph_edges,
    const float *h_node_values_input,
    float  *h_gpu_node_values_output,
    const float *h_inv_edges_per_node,
    int nr_iterations,
    int num_nodes,
    int avg_edges
) {
    const int num_bytes_alloc = (num_nodes) * sizeof(float);
    float *device_input_array  = nullptr;
    float *device_output_array = nullptr;

    // TODO: allocate GPU memory
    cudaMalloc((void **) &device_input_array,  num_bytes_alloc);
    cudaMalloc((void **) &device_output_array, num_bytes_alloc);
    cudaMemset(device_input_array + num_nodes, 0, num_bytes_alloc - num_nodes);
    
    // TODO: check for allocation failure
    if (!device_input_array || !device_output_array) 
     {
         std::cerr << "Couldn't allocate memory!" << std::endl;
         return 1;
     }
    // TODO: copy data to the GPU
    {
         cudaMemcpy(device_input_array, h_node_values_input, num_nodes, cudaMemcpyHostToDevice);
         check_launch("copy to gpu");

     }
    event_pair timer;
    start_timer(&timer);

    const int block_size = 192;

    int numBlocks = (num_nodes + block_size - 1) / block_size;

    // TODO: launch your kernels the appropriate number of iterations
    for(int iter = 0; iter < 5; iter++) 
    {
        device_graph_propagate<<<numBlocks, block_size>>>(h_graph_indices, h_graph_edges, h_node_values_input, h_gpu_node_values_output,
                             h_inv_edges_per_node, num_nodes);
        device_graph_propagate<<<numBlocks, block_size>>>(h_graph_indices, h_graph_edges, h_gpu_node_values_output, h_node_values_input,
                             h_inv_edges_per_node, num_nodes);
    }
    device_graph_propagate<<<numBlocks, block_size>>>(h_graph_indices, h_graph_edges, h_node_values_input, h_gpu_node_values_output,
                             h_inv_edges_per_node, num_nodes);
    check_launch("gpu graph propagate");
    double gpu_elapsed_time = stop_timer(&timer);

    // TODO: copy final data back to the host for correctness checking
    cudaMemcpy(h_gpu_node_values_output, device_output_array, num_nodes,
                cudaMemcpyDeviceToHost);
     check_launch("copy from gpu");
    // TODO: free the memory you allocated!
    cudaFree(device_input_array);
    cudaFree(device_output_array);
    return gpu_elapsed_time;
}

/**
 * This function computes the number of bytes read from and written to
 * global memory by the pagerank algorithm.
 * 
 * nodes:
 *      The number of nodes in the graph
 *
 * edges: 
 *      The average number of edges in the graph
 *
 * iterations:
 *      The number of iterations the pagerank algorithm was run
 */
uint get_total_bytes(uint nodes, uint edges, uint iterations)
{
    // TODO
    return 0; 
}

#endif
