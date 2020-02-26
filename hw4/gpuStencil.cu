#include <math_constants.h>

#include "BC.h"
constexpr const int SIDE = 16;
constexpr const int numYPerStep = 16;
/**
 * Calculates the next finite difference step given a
 * grid point and step lengths.
 *
 * @param curr Pointer to the grid point that should be updated.
 * @param width Number of grid points in the x dimension.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 * @returns Grid value of next timestep.
 */
template<int order>
__device__
float Stencil(const float* curr, int width, float xcfl, float ycfl) {
    switch(order) {
        case 2:
            return curr[0] + xcfl * (curr[-1] + curr[1] - 2.f * curr[0]) +
                   ycfl * (curr[width] + curr[-width] - 2.f * curr[0]);

        case 4:
            return curr[0] + xcfl * (-curr[2] + 16.f * curr[1] - 30.f * curr[0]
                                     + 16.f * curr[-1] - curr[-2])
                           + ycfl * (- curr[2 * width] + 16.f * curr[width]
                                     - 30.f * curr[0] + 16.f * curr[-width]
                                     - curr[-2 * width]);

        case 8:
            return curr[0] + xcfl * (-9.f * curr[4] + 128.f * curr[3]
                                     - 1008.f * curr[2] + 8064.f * curr[1]
                                     - 14350.f * curr[0] + 8064.f * curr[-1]
                                     - 1008.f * curr[-2] + 128.f * curr[-3]
                                     - 9.f * curr[-4])
                           + ycfl * (-9.f * curr[4 * width]
                                     + 128.f * curr[3 * width]
                                     - 1008.f * curr[2 * width]
                                     + 8064.f * curr[width]
                                     - 14350.f * curr[0]
                                     + 8064.f * curr[-width]
                                     - 1008.f * curr[-2 * width]
                                     + 128.f * curr[-3 * width]
                                     - 9.f * curr[-4 * width]);

        default:
            printf("ERROR: Order %d not supported", order);
            return CUDART_NAN_F;
    }
}

/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be very simple and only use global memory
 * and 1d threads and blocks.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param nx Number of grid points in the x dimension to which the full
 *           stencil can be applied (ie the number of points that are at least
 *           order/2 grid points away from the boundary).
 * @param ny Number of grid points in the y dimension to which th full
 *           stencil can be applied.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int order>
__global__
void gpuStencilGlobal(float* next, const float* __restrict__ curr, int gx, int nx, int ny,
                float xcfl, float ycfl) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x);
    int bordersize = (gx-nx)/2;
    if( i < nx*ny ) {
        int x = (i % ny) + bordersize;
        int y = (i / ny) + bordersize;
        int pos = x + y * gx;

        next[pos]=Stencil<order>(curr+pos, gx, xcfl, ycfl);
    }
    return;
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilGlobal kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
double gpuComputationGlobal(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    float xcfl = params.xcfl();
    float ycfl = params.ycfl();

    int nx = params.nx();
    int ny = params.ny();

    int gx = params.gx();
    const int order = params.order();
    // TODO: Declare variables/Compute parameters.

    event_pair timer;
    start_timer(&timer);
    const int block_size = 512;
    int numBlocks = (nx*ny + block_size - 1) / block_size;

    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // TODO: Apply stencil.
        if (params.order()==2) {
            gpuStencilGlobal<2><<<numBlocks, block_size>>>(next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
        }
        else if (params.order()==4) {
            gpuStencilGlobal<4><<<numBlocks, block_size>>>(next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
        }
        else if (params.order()==8) {
            gpuStencilGlobal<8><<<numBlocks, block_size>>>(next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
        }
        Grid::swap(curr_grid, next_grid);
        /*
        curr_grid.fromGPU();
        if (i == 0) {
            curr_grid.saveStateToFile("Iteration_0_Global.csv");
        }
        if (i == 1000) {
            curr_grid.saveStateToFile("Iteration_1000_Global.csv");
        }
        if (i == 2000) {
            curr_grid.saveStateToFile("Iteration_2000_Global.csv");
        }
        */
    }


    check_launch("gpuStencilGlobal");
    return stop_timer(&timer);
}


/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be optimized to compute finite difference updates
 * in blocks of size (blockDim.y * numYPerStep) * blockDim.x. Each thread
 * should calculate at most numYPerStep updates. It should still only use
 * global memory.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param nx Number of grid points in the x dimension to which the full
 *           stencil can be applied (ie the number of points that are at least
 *           order/2 grid points away from the boundary).
 * @param ny Number of grid points in the y dimension to which th full
 *           stencil can be applied.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int order, int numYPerStep>
__global__
void gpuStencilBlock(float* next, const float* __restrict__ curr, int gx, int nx, int ny,
                    float xcfl, float ycfl) {
    // TODO
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int bordersize = (gx-nx)/2;
    if ( ix < nx) {
        for (int y=iy * numYPerStep; y < (iy + 1) * numYPerStep; y++) {
            if (y < ny) {
                int pos = (ix + bordersize) + (y + bordersize) * gx;
                next[pos]=Stencil<order>(curr+pos, gx, xcfl, ycfl);
            }
        }
    }

}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilBlock kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
double gpuComputationBlock(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);
    float xcfl = params.xcfl();
    float ycfl = params.ycfl();

    int nx = params.nx();
    int ny = params.ny();

    int gx = params.gx();
    // TODO: Declare variables/Compute parameters.
    int block_size_x = 256;
    int block_size_y = 4;
    int numBlocks_x = (nx + block_size_x - 1) / block_size_x;
    int numBlocks_y = (ny + numYPerStep * block_size_y - 1) / (numYPerStep * block_size_y);
    dim3 threads(block_size_x, block_size_y);
    dim3 blocks(numBlocks_x, numBlocks_y);

    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // TODO: Apply stencil.
        //gpuStencilBlock<params.order(), numYPerStep><<<blocks, threads>>>(next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
        if (params.order()==2) {
            gpuStencilBlock<2, numYPerStep><<<blocks, threads>>>(next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
        }
        else if (params.order()==4) {
            gpuStencilBlock<4, numYPerStep><<<blocks, threads>>>(next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
        }
        else if (params.order()==8) {
            gpuStencilBlock<8, numYPerStep><<<blocks, threads>>>(next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
        }
        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilBlock");
    return stop_timer(&timer);
}


/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be optimized to compute finite difference updates
 * in blocks of size side * side using shared memory.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param gy Number of grid points in the y dimension.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int side, int order>
__global__
void gpuStencilShared(float* next, const float* __restrict__ curr, int gx, int gy,
               float xcfl, float ycfl) {
    // TODO
    extern __shared__ float block[];
    int s = side;
    int bordersize = order / 2;
    int nx = gx - 2 * bordersize;
    int ny = gy - 2 * bordersize;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int dx = threadIdx.x;
    int dy = threadIdx.y;
    int size = side + 2 * bordersize;
    int pos_block = dx + bordersize + (dy + bordersize) * size;
    int pos = (ix + bordersize) + (iy + bordersize) * gx;
    if ((ix < nx) && (iy < ny)) {
        block[pos_block]=curr[pos];
        if (dx == 0) {
            for (int j=1; j<=bordersize; j++) {
                block[pos_block-j] = curr[pos-j];
            }
        }
        if ((dx == s - 1) || (ix == nx - 1)) {
            for (int j=1; j<=bordersize; j++) {
                block[pos_block+j] = curr[pos+j];
            }
        }
        if (dy == 0) {
            for (int j=1; j<=bordersize; j++) {
                block[pos_block-j*size] = curr[pos-j*gx];
            }
        }
        if ((dy == s - 1) || (iy == ny - 1)) {
            for (int j=1; j<=bordersize; j++) {
                block[pos_block+j*size] = curr[pos+j*gx];
            }
        }
    }
    __syncthreads();
    if ((ix < nx) && (iy < ny)) {
        next[pos]=Stencil<order>(block+pos_block, size, xcfl, ycfl);
    }


}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilShared kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
template<int order>
double gpuComputationShared(Grid& curr_grid, const simParams& params) {

    
    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    float xcfl = params.xcfl();
    float ycfl = params.ycfl();

    int nx = params.nx();
    int ny = params.ny();

    int gx = params.gx();
    int gy = params.gy();
    // TODO: Declare variables/Compute parameters.
    int block_size_x = SIDE;
    int block_size_y = SIDE;
    int numBlocks_x = (nx + block_size_x - 1) / block_size_x;
    int numBlocks_y = (ny + block_size_y - 1) / (block_size_y);
    dim3 threads(block_size_x, block_size_y);
    dim3 blocks(numBlocks_x, numBlocks_y);
    
    int side = (block_size_x + 2 * params.order());
    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // TODO: Apply stencil.
        if (params.order()==2) {
            gpuStencilShared<SIDE, 2><<<blocks, threads, side*side*sizeof(float)>>>(next_grid.dGrid_, curr_grid.dGrid_, gx, gy, xcfl, ycfl);
        }
        else if (params.order()==4) {
            gpuStencilShared<SIDE, 4><<<blocks, threads, side*side*sizeof(float)>>>(next_grid.dGrid_, curr_grid.dGrid_, gx, gy, xcfl, ycfl);
        }
        else if (params.order()==8) {
            gpuStencilShared<SIDE, 8><<<blocks, threads, side*side*sizeof(float)>>>(next_grid.dGrid_, curr_grid.dGrid_, gx, gy, xcfl, ycfl);
        }
        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilShared");
    return stop_timer(&timer);
}

