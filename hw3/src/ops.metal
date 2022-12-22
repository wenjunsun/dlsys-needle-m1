#include <metal_stdlib>
using namespace metal;

typedef float scalar_t;
#define MAX_VEC_SIZE 8

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
                      uint index            [[thread_position_in_grid]])
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
                      device scalar_t* out      [[buffer(1)]],
                      uint index             [[thread_position_in_grid]])
{
    out[index] = tanh(a[index]);
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




