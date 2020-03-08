/**
 * @file dns.h
 * 
 * In this homework, you are asked to implement the Dekel-Nassimi-Sahni (DNS)
 * matrix multiplication algorithm. This implementation follows the algorithm
 * described in the book "Introduction to Parallel Computing" by Grama, et. al
 * in Chapter 8.2.3 and 8.2.4. We will be implementing the block version 
 * described in 8.2.4.
 * 
 * The DNS algorithm assumes that the matrices A and B are already
 * evenly distributed across all ranks in the k = 0 plane. Instead, we will
 * assume that the root rank (rank 0) will have A and B. When running the DNS
 * algorithm, we would like to distribute necessary data to the appropriate
 * ranks as described in the DNS algorithm. Once the computation is complete, 
 * we would like rank 0 to have the output stored into C.
 * 
 * Your job is to implement the scattering of A and B to the other ranks,
 * the DNS algorithm itself, and the gathering of the sub-outputs back to the 
 * root rank.
 */

#ifndef _DNS_H
#define _DNS_H

#include "util.h"
#include "mpi.h"


#define MPI_SAFE_CALL( call ) do {                               \
    int err = call;                                              \
    if (err != MPI_SUCCESS) {                                    \
        fprintf(stderr, "MPI error %d in file '%s' at line %i",  \
               err, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    } } while(0)

/** Index for the i dimension when passing coordinates to MPI_Cart_rank */
constexpr int I_DIM = 0;

/** Index for the j dimension when passing coordinates to MPI_Cart_rank */
constexpr int J_DIM = 1;

/** Index for the k dimension when passing coordinates to MPI_Cart_rank */
constexpr int K_DIM = 2;

/**
 * @struct mesh_info
 * Each rank needs to know its rank but also its location in the 3D topology.
 * However, what makes a cartesian topology extremely versatile is the ability
 * to define subplanes and communicate within those subplanes. For this problem,
 * each rank can talk to ranks in the following planes: 
 *  - the entire 3D topology (comm_3d)
 *  - the ij plane (comm_ij)
 *  - those with the same i (comm_i), j (comm_j), and k (comm_k). 
 * 
 * Do not modify this struct.
 */
struct mesh_info
{
  MPI_Comm comm_3d;   // can talk to all ranks in 3D grid
  MPI_Comm comm_ij;   // can talk to all ranks in same ij plane

  MPI_Comm comm_i;    // can talk to all ranks with same i
  MPI_Comm comm_j;    // can talk to all ranks with same j
  MPI_Comm comm_k;    // can talk to all ranks with same k

  int blockdim;       // block size of submatrix processed by this rank
  int myrank;         // this rank's original rank #
  int my3drank;       // this rank's new rank after 3D grid assignment
  int mycoords[3];    // this rank's coordinates in the 3D topology
  int num_procs;      // the total number of ranks in this program
};

/**
 * Get all communicators, ranks, and 3D coordinates for a given rank. In other
 * words, initialize every field of the mesh_info struct.
 * 
 * @param[in] blockdim  The square submatrix dimension
 * @param[in] q         The topology dimension (q x q x q = # processes)
 * 
 * @return a fully initialized mesh_info struct.
 */
struct mesh_info initialize_topology(int blockdim, int q)
{
  // TODO: initialize every field of the mesh_info struct
  return mesh_info();
}

/**
 * Free all communicators associated with the mesh_info struct.
 * @param[inout] mesh_info  mesh_info struct containing communicators to free
 */
void mesh_info_free(struct mesh_info& mesh_info)
{
  // TODO: free all communicators allocated in initialize_topology
}

/**
 * Runs the DNS matrix multiplication algorithm.
 * 
 * @param[in]  mesh_info    This rank's topology information
 * @param[in]  a            The original A matrix at root, nullptr otherwise
 * @param[in]  b            The original B matrix at root, nullptr otherwise
 * @param[out] c            The output C matrix at root, nullptr otherwise
 * @param[in]  q            The topology dimension
 * @param[in]  matrix_dim   The original matrix size
 */
void dns_multiply(const struct mesh_info& mesh_info, const float *a, 
                  const float *b, float *c, int q, int matrix_dim)
{
  // TODO: allocate submatrices that this rank needs to work on 

  /*
   * Prepare to scatterv A and B matrices from root to ranks in the k = 0 plane.
   * 
   * Since we need to send (non-contiguous) submatrices, we need new types.
   * 
   * We approach this by first making blocktype as a blockdim x blockdim block in the matrix.
   * However, we need to specify the memory location where the block starts.
   * The memory location is given by displs. displs is given in unit of the MPI data type. 
   * In principle, this is blocktype.
   * But in this case, we need to choose a different unit. We decided to set
   * the unit equal to blockdim. This simplifies the arithmetic for displs.
   */
  MPI_Datatype blocktype;        // MPI user data type corresponding to one block in the matrix
  MPI_Datatype blk_type_resized; // Extent of data type modified; used as the extent unit by displs

  // Data type for a block of size blockdim x blockdim
  MPI_Type_vector(mesh_info.blockdim, mesh_info.blockdim, matrix_dim, MPI_FLOAT, &blocktype);
  // Changing the extent and making it equal to blockdim
  MPI_Type_create_resized(blocktype, 0, sizeof(float) * mesh_info.blockdim, &blk_type_resized);
  MPI_Type_commit(&blk_type_resized);

  // Given this new datatype, we need to compute the start index of the
  // submatrix to send to each rank.
  int *displs = new int[mesh_info.num_procs];
  int *counts = new int[mesh_info.num_procs];
  for (int i = 0; i < q; i++)
  {
    for (int j = 0; j < q; j++)
    {
      displs[i * q + j] = i * matrix_dim + j; // Index of block to send; in units of blockdim
      counts[i * q + j] = 1;                  // We send only one block to each rank
    }
  }

  // TODO: Scatterv A and B matrices from root to ranks in the k = 0 plane

  // TODO: Send A[i, j, 0] --> A[i, j, j]. Use P2P calls.

  // TODO: Send B[i, j, 0] --> B[i, j, i]. Use P2P calls.

  // TODO: Broadcast A[i, j, j] along j axis.

  // TODO: Broadcast B[i, j, i] along i axis.

  // TODO: Multiply local A and B matrices together and place into local C.

  // TODO: Reduce results back into the k = 0 plane.

  // TODO: Gatherv results back to the root node

  // TODO: Release any resources that you may have allocated
  delete[] displs;
  delete[] counts;
  MPI_Type_free(&blk_type_resized);
  MPI_Type_free(&blocktype);
}

#endif 
