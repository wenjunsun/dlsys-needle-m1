#include <metal_stdlib>
using namespace metal;

typedef float scalar_t;
#define MAX_VEC_SIZE 8

////////////////////////////////////////////////////////////////////////////////
// Fill operation
////////////////////////////////////////////////////////////////////////////////
kernel void fill(device scalar_t* out       [[buffer(0)]],
                 device const scalar_t* val [[buffer(1)]],
                 uint index                 [[thread_position_in_grid]])
{
    out[index] = *val;
}


////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i in strided matrix to the
// memory location in its underling compact matrix
size_t index_transform(uint   index,
                       device const int32_t* shape,
                       device const int32_t* strides,
                       device const size_t* dim,
                       device const size_t* offset)
{
    size_t idxs[MAX_VEC_SIZE];
    size_t cur_size, pre_size = 1;
    for (int i = (int)(*dim) - 1; i >= 0; i--) {
        cur_size = pre_size * shape[i]; 
        idxs[i] = index % cur_size / pre_size;
        pre_size = cur_size;
    }
    size_t comp_idx = (*offset);
    for (size_t i = 0; i < (*dim); i++) 
        comp_idx += idxs[i] * strides[i];
    return comp_idx;
}

kernel void compact(device const scalar_t* a      [[buffer(0)]],
                    device scalar_t* out          [[buffer(1)]],
                    device const int32_t* shape   [[buffer(2)]],
                    device const int32_t* strides [[buffer(3)]],
                    device const size_t* dim      [[buffer(4)]],
                    device const size_t* offset   [[buffer(5)]],
                    uint index                    [[thread_position_in_grid]])
{
    out[index] = a[index_transform(index, shape, strides, dim, offset)]; 
}

kernel void ewise_setitem(device const scalar_t* a      [[buffer(0)]],
                          device scalar_t* out          [[buffer(1)]],
                          device const int32_t* shape   [[buffer(2)]],
                          device const int32_t* strides [[buffer(3)]],
                          device const size_t* dim      [[buffer(4)]],
                          device const size_t* offset   [[buffer(5)]],
                          uint index                    [[thread_position_in_grid]])
{
    out[index_transform(index, shape, strides, dim, offset)] = a[index];
}

kernel void scalar_setitem(device scalar_t* out          [[buffer(0)]],
                           device const scalar_t* val    [[buffer(1)]],
                           device const int32_t* shape   [[buffer(2)]],
                           device const int32_t* strides [[buffer(3)]],
                           device const size_t* dim      [[buffer(4)]],
                           device const size_t* offset   [[buffer(5)]],
                           uint index                    [[thread_position_in_grid]])
{
    out[index_transform(index, shape, strides, dim, offset)] = (*val);
}


////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

kernel void ewise_add(device const scalar_t* a [[buffer(0)]],
                      device const scalar_t* b [[buffer(1)]],
                      device scalar_t* out     [[buffer(2)]],
                      uint index               [[thread_position_in_grid]])
{
    out[index] = a[index] + b[index];
}

kernel void scalar_add(device const scalar_t* a [[buffer(0)]],
                       device const scalar_t* b [[buffer(1)]],
                       device scalar_t* out     [[buffer(2)]],
                       uint index            [[thread_position_in_grid]])
{
    out[index] = a[index] + (*b);
}

kernel void ewise_mul(device const scalar_t* a [[buffer(0)]],
                      device const scalar_t* b [[buffer(1)]],
                      device scalar_t* out     [[buffer(2)]],
                      uint index            [[thread_position_in_grid]])
{
    out[index] = a[index] * b[index];
}

kernel void scalar_mul(device const scalar_t* a [[buffer(0)]],
                       device const scalar_t* b [[buffer(1)]],
                       device scalar_t* out     [[buffer(2)]],
                       uint index            [[thread_position_in_grid]])
{
    out[index] = a[index] * (*b);
}

kernel void ewise_div(device const scalar_t* a [[buffer(0)]],
                      device const scalar_t* b [[buffer(1)]],
                      device scalar_t* out     [[buffer(2)]],
                      uint index            [[thread_position_in_grid]])
{
    out[index] = a[index] / b[index];
}

kernel void scalar_div(device const scalar_t* a [[buffer(0)]],
                       device const scalar_t* b [[buffer(1)]],
                       device scalar_t* out     [[buffer(2)]],
                       uint index            [[thread_position_in_grid]])
{
    out[index] = a[index] / (*b);
}

kernel void scalar_power(device const scalar_t* a [[buffer(0)]],
                       device const scalar_t* b   [[buffer(1)]],
                       device scalar_t* out       [[buffer(2)]],
                       uint index              [[thread_position_in_grid]])
{
    out[index] = pow(a[index], (*b));
}

kernel void ewise_maximum(device const scalar_t* a [[buffer(0)]],
                          device const scalar_t* b [[buffer(1)]],
                          device scalar_t* out     [[buffer(2)]],
                          uint index            [[thread_position_in_grid]])
{
    out[index] = max(a[index], b[index]);
}

kernel void scalar_maximum(device const scalar_t* a [[buffer(0)]],
                           device const scalar_t* b [[buffer(1)]],
                           device scalar_t* out     [[buffer(2)]],
                           uint index            [[thread_position_in_grid]])
{
    out[index] = max(a[index], (*b));
}

kernel void ewise_eq(device const scalar_t* a [[buffer(0)]],
                     device const scalar_t* b [[buffer(1)]],
                     device scalar_t* out     [[buffer(2)]],
                     uint index            [[thread_position_in_grid]])
{
    out[index] = a[index] == b[index];
}

kernel void scalar_eq(device const scalar_t* a [[buffer(0)]],
                      device const scalar_t* b [[buffer(1)]],
                      device scalar_t* out     [[buffer(2)]],
                      uint index            [[thread_position_in_grid]])
{
    out[index] = a[index] == (*b);
}

kernel void ewise_ge(device const scalar_t* a  [[buffer(0)]],
                     device const scalar_t* b  [[buffer(1)]],
                     device scalar_t* out      [[buffer(2)]],
                     uint index             [[thread_position_in_grid]])
{
    out[index] = a[index] >= b[index];
}

kernel void scalar_ge(device const scalar_t* a [[buffer(0)]],
                      device const scalar_t* b [[buffer(1)]],
                      device scalar_t* out     [[buffer(2)]],
                      uint index            [[thread_position_in_grid]])
{
    out[index] = a[index] >= (*b);
}

kernel void ewise_log(device const scalar_t* a [[buffer(0)]],
                      device scalar_t* out     [[buffer(1)]],
                      uint index            [[thread_position_in_grid]])
{
    out[index] = log(a[index]);
}

kernel void ewise_exp(device const scalar_t* a [[buffer(0)]],
                      device scalar_t* out     [[buffer(1)]],
                      uint index            [[thread_position_in_grid]])
{
    out[index] = exp(a[index]);
}

kernel void ewise_tanh(device const scalar_t* a [[buffer(0)]],
                       device scalar_t* out     [[buffer(1)]],
                       uint index               [[thread_position_in_grid]])
{
    out[index] = tanh(a[index]);
}

////////////////////////////////////////////////////////////////////////////////
// Matrix mulplication
////////////////////////////////////////////////////////////////////////////////

kernel void matmul_naive(device const scalar_t* a [[buffer(0)]],
                         device const scalar_t* b [[buffer(1)]],
                         device scalar_t* out     [[buffer(2)]],
                         device const uint32_t* M [[buffer(3)]],
                         device const uint32_t* N [[buffer(4)]],
                         device const uint32_t* P [[buffer(5)]],
                         uint2 index              [[thread_position_in_grid]])
{
    int32_t j = index.x, i = index.y, m = (*M), n = (*N), p = (*P);

    // Check if the thread is in-bounds.
    if ((i < m) && (j < p)) {
        scalar_t sum = 0;
        for (int k = 0; k < n; k++) {
            // a[i][k], b[k][j]
            sum += a[n * i + k] * b[k * p + j];
        }    
        // out[i][j]
        out[i * p + j] = sum;
    }
}

kernel void matmul_block(device const scalar_t* A [[buffer(0)]],
                         device const scalar_t* B [[buffer(1)]],
                         device scalar_t* out     [[buffer(2)]],
                         device const uint32_t* M [[buffer(3)]],
                         device const uint32_t* N [[buffer(4)]],
                         device const uint32_t* P [[buffer(5)]],
                         uint2 threadgroup_pos [[ threadgroup_position_in_grid ]],
                         uint2 local_thread_idx [[ thread_position_in_threadgroup ]])
{
    // Note: be sure that this is set to the same value as "threads per group" in the calling code!
    const int BLOCK_SIZE = 8;

    const uint32_t wB = (*P);
    const uint32_t wA = (*N);
    
    // Block index
    const uint bx = threadgroup_pos.x;
    const uint by = threadgroup_pos.y;
    
    // Thread index
    const uint tx =local_thread_idx.x;
    const uint ty =local_thread_idx.y;
    
    // Index of the first sub-matrix of A processed by the block
    const uint aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    const uint aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    const uint aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    const uint bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    const uint bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    scalar_t Csub = 0;
    
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (uint a = aBegin, b = bBegin;
        a <= aEnd;
        a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        threadgroup float As[BLOCK_SIZE][BLOCK_SIZE];
        

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        threadgroup float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        threadgroup_barrier(mem_flags::mem_none);

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        threadgroup_barrier(mem_flags::mem_none);
      }

    const int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    out[c + wB * ty + tx] = Csub; 
}


////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

kernel void reduce_max(device const scalar_t* a            [[buffer(0)]],
                       device scalar_t* out                [[buffer(1)]],
                       device const size_t* reduce_size [[buffer(2)]],
                       uint index                       [[thread_position_in_grid]])
{
    size_t offset = index * (*reduce_size);
    scalar_t reduce_max = a[offset];
    for (size_t i = 1; i < (*reduce_size); i++) {
      reduce_max = max(reduce_max, a[i + offset]);
    }
    out[index] = reduce_max;
}

kernel void reduce_sum(device const scalar_t* a            [[buffer(0)]],
                       device scalar_t* out                [[buffer(1)]],
                       device const size_t* reduce_size [[buffer(2)]],
                       uint index                       [[thread_position_in_grid]])
{
    size_t offset = index * (*reduce_size);
    scalar_t reduce_sum = 0;
    for (size_t i = 0; i < (*reduce_size); i++) {
      reduce_sum += a[i + offset];
    }
    out[index] = reduce_sum;
}
