#ifndef _SHIFT_CUH
#define _SHIFT_CUH

/** 
 * Repeating from the tutorial, just in case you haven't looked at it.
 * "kernels" or __global__ functions are the entry points to code that executes on the GPU.
 * The keyword __global__ indicates to the compiler that this function is a GPU entry point.
 * __global__ functions must return void, and may only be called or "launched" from code that
 * executes on the CPU.
 */

typedef unsigned char uchar;
typedef unsigned int uint;

/**
 * Implements a per-element shift by loading a single byte and shifting it.
 */ 
__global__ void shift_char(const uchar *input_array, uchar *output_array,
                           uchar shift_amount, uint array_length) 
{

    const uint i = (uint)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i<array_length) {
        output_array[i]=input_array[i]+shift_amount;
    }

}

/** 
 * Here we load 4 bytes at a time instead of just 1 to improve bandwidth
 * due to a better memory access pattern.
 */
__global__ void shift_int(const uint *input_array, uint *output_array,
                          uint shift_amount, uint array_length)
{
    // TODO: fill in
    const uint i = (uint)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i<array_length) {
        output_array[i]=input_array[i]+shift_amount;
    }
}

/** 
 * Here we go even further and load 8 bytes - does it improve further?
 */
__global__ void shift_int2(const uint2 *input_array, uint2 *output_array,
                           uint shift_amount, uint array_length) 
{
    // TODO: fill in
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;
    uint n = blockDim.x * gridDim.x;
    if(i + j*n<array_length) {
        output_array[i+j*n].x=input_array[i+j*n].x;
        output_array[i+j*n].y=input_array[i+j*n].y;

        for (uint k=0; k<4; k++) {
            output_array[i+j*n].x+=(shift_amount<<k);
            output_array[i+j*n].y+=(shift_amount<<k);
        }
    }
}

// the following three kernels launch their respective kernels
// and report the time it took for the kernel to run

double doGPUShiftChar(const uchar *d_input, uchar *d_output,
                      uchar shift_amount, uint text_size, uint block_size) 
{
    // TODO: compute your grid dimensions
    int numBlocks = (text_size + block_size - 1) / block_size;

    event_pair timer;
    start_timer(&timer);
    shift_char<<<numBlocks, block_size>>>(d_input, d_output, shift_amount, text_size);
    // TODO: launch kernel

    check_launch("gpu shift cipher uint");
    return stop_timer(&timer);
}

double doGPUShiftUInt(const uchar *d_input, uchar *d_output,
                      uchar shift_amount, uint text_size, uint block_size) 
{
    // TODO: compute grid dimensions
    int numBlocks = (((text_size+4-1)/4 )+ block_size - 1) / block_size;
    // TODO: compute 4 byte shift value
    uint new_shift_amount=0;
    new_shift_amount+=(shift_amount);
    for (uint k=0; k<3; k++) {
        new_shift_amount=(new_shift_amount<<8);
        new_shift_amount+=shift_amount;
    }
    event_pair timer;
    start_timer(&timer);
    shift_int<<<numBlocks, block_size>>>((uint*)(d_input), (uint*)(d_output), new_shift_amount, (text_size+4-1)/4);
    // TODO: launch kernel
    
    check_launch("gpu shift cipher uint");
    return stop_timer(&timer);
}

double doGPUShiftUInt2(const uchar* d_input, uchar* d_output,
                       uchar shift_amount, uint text_size, uint block_size) 
{
    // TODO: compute your grid dimensions

    // TODO: compute 4 byte shift value

    event_pair timer;
    start_timer(&timer);

    // TODO: launch kernel

    check_launch("gpu shift cipher uint2");
    return stop_timer(&timer);
}


#endif
