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
#include<iostream>
using namespace std;


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
  mesh_info mesh;
  mesh.blockdim=blockdim;
  int num_procs;
  int my_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  mesh.num_procs=num_procs;
  mesh.myrank=my_rank;
  int dims[3]={q,q,q};
  int periods[3]={1,1,1};
  MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &mesh.comm_3d);
  int my3drank;
  MPI_Comm_rank(mesh.comm_3d, &my3drank);
  MPI_Cart_coords(mesh.comm_3d, my3drank, 3, mesh.mycoords);
  int keep_dims_i[3] = {1, 0, 0};
  MPI_Cart_sub(mesh.comm_3d, keep_dims_i, &mesh.comm_i);
  int keep_dims_j[3] = {0, 1, 0};
  MPI_Cart_sub(mesh.comm_3d, keep_dims_j, &mesh.comm_j);
  int keep_dims_k[3] = {0, 0, 1};
  MPI_Cart_sub(mesh.comm_3d, keep_dims_k, &mesh.comm_k);
  int keep_dims_ij[3] = {1, 1, 0};
  MPI_Cart_sub(mesh.comm_3d, keep_dims_ij, &mesh.comm_ij);
  mesh.my3drank=my3drank;
  return mesh;
}





/**
 * Free all communicators associated with the mesh_info struct.
 * @param[inout] mesh_info  mesh_info struct containing communicators to free
 */
void mesh_info_free(struct mesh_info& mesh_info)
{
  MPI_Comm_free(&mesh_info.comm_3d);
  MPI_Comm_free(&mesh_info.comm_i);
  MPI_Comm_free(&mesh_info.comm_j);
  MPI_Comm_free(&mesh_info.comm_k);
  MPI_Comm_free(&mesh_info.comm_ij);
  // TODO: free all communicators allocated in initialize_topology
}




void print_mat(const float* a, int n) {
  for (int i=0; i<n; i++) {
    for (int j=0; j<n; j++) {
      cout<<a[i*n+j]<<" ";
    }
    cout<<endl;
  }
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
  int block_size=mesh_info.blockdim*mesh_info.blockdim;
  float* Aik=new float[block_size];
  float* Bkj=new float[block_size];
  const int* coords=mesh_info.mycoords;
  if (mesh_info.myrank==0) {
    print_mat(a, matrix_dim);
  }
  if (coords[2] == 0)
    {
        MPI_Scatterv(a, counts, displs, blk_type_resized, Aik, block_size, MPI_FLOAT, 0, mesh_info.comm_ij);
        MPI_Scatterv(b, counts, displs, blk_type_resized, Bkj, block_size, MPI_FLOAT, 0, mesh_info.comm_ij);
        //cout<<"Rank "<<coords[0]<<" "<<coords[1]<<" "<<coords[2]<<": ";
        print_mat(Aik, mesh_info.blockdim);
  }



  
  // TODO: Scatterv A and B matrices from root to ranks in the k = 0 plane


  

    
  // TODO: Send A[i, j, 0] --> A[i, j, j]. Use P2P calls.
  if (coords[1] != 0 && coords[2] == 0)
    {
        int recv_coords[3] = {coords[0], coords[1], coords[1]};
        int recv_rank;
        MPI_Cart_rank(mesh_info.comm_3d, recv_coords, &recv_rank);

        MPI_Send(Aik, block_size, MPI_FLOAT, recv_rank, 0, mesh_info.comm_3d);
    }
    else if (coords[1] == coords[2] && coords[2] != 0)
    {
        // Coordinates of sending process
        int send_coords[3] = {coords[0], coords[1], 0};
        int send_rank;
        MPI_Cart_rank(mesh_info.comm_3d, send_coords, &send_rank);

        MPI_Recv(Aik, block_size, MPI_FLOAT, send_rank, 0, mesh_info.comm_3d, MPI_STATUS_IGNORE);
        cout<<"Rank "<<coords[0]<<" "<<coords[1]<<" "<<coords[2]<<": ";
        print_mat(Aik, mesh_info.blockdim);
    }

  // TODO: Send B[i, j, 0] --> B[i, j, i]. Use P2P calls.
  if (coords[0] != 0 && coords[2] == 0)
    {
        // Sending from (i,j,0) to (i,j,j)
        // Coordinates of receiving process
        int recv_coords[3] = {coords[0], coords[1], coords[0]};
        int recv_rank;
        MPI_Cart_rank(mesh_info.comm_3d, recv_coords, &recv_rank);

        MPI_Send(Bkj, block_size, MPI_FLOAT, recv_rank, 0, mesh_info.comm_3d);
        
    }
    else if (coords[0] == coords[2] && coords[2] != 0)
    {
        // Coordinates of sending process
        int send_coords[3] = {coords[0], coords[1], 0};
        int send_rank;
        MPI_Cart_rank(mesh_info.comm_3d, send_coords, &send_rank);
        cout<<"Rank "<<coords[0]<<" "<<coords[1]<<" "<<coords[2]<<": ";
        print_mat(Bkj, mesh_info.blockdim);
        MPI_Recv(Bkj, block_size, MPI_FLOAT, send_rank, 0, mesh_info.comm_3d, MPI_STATUS_IGNORE);
    }

  // TODO: Broadcast A[i, j, j] along j axis.

  if (coords[1]==coords[2]) {
    MPI_Bcast(Aik, block_size, MPI_FLOAT, coords[1], mesh_info.comm_j);
  }

  if (coords[0]==coords[2]) {
    MPI_Bcast(Bkj, block_size, MPI_FLOAT, coords[0], mesh_info.comm_i);
  }

  // TODO: Broadcast B[i, j, i] along i axis.

  // TODO: Multiply local A and B matrices together and place into local C.

  float* Cijk=new float[block_size];
  omp_matmul(Aik, Bkj, Cijk, mesh_info.blockdim);

  // TODO: Reduce results back into the k = 0 plane.
  float* Cij=new float[block_size];
  MPI_Reduce(Cijk, Cij, block_size, MPI_FLOAT, MPI_SUM, 0, mesh_info.comm_k);

  // TODO: Gatherv results back to the root node

  if (coords[2] == 0) {
        MPI_Gatherv(Cij, block_size, MPI_FLOAT, c, counts, displs, blk_type_resized, 0, mesh_info.comm_ij);
  }

  // TODO: Release any resources that you may have allocated



  if (mesh_info.myrank==0) {
    cout<<"Matrix A: \n";
    print_mat(a, matrix_dim);
    cout<<"Matrix B: \n";
    print_mat(b, matrix_dim);
    cout<<"Matrix C: \n";
    print_mat(c, matrix_dim);
  }

  delete[] Aik;
  delete[] Bkj;
  delete[] Cijk;
  delete[] Cij;
  delete[] displs;
  delete[] counts;
  MPI_Type_free(&blk_type_resized);
  MPI_Type_free(&blocktype);
}

#endif 
