#include <metal_stdlib>
using namespace metal;

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

kernel void ewise_add(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* out     [[buffer(2)]],
                      uint index            [[thread_position_in_grid]])
{
    out[index] = a[index] + b[index];
}

kernel void scalar_add(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* out     [[buffer(2)]],
                       uint index            [[thread_position_in_grid]])
{
    out[index] = a[index] + (*b);
}

kernel void ewise_mul(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* out     [[buffer(2)]],
                      uint index            [[thread_position_in_grid]])
{
    out[index] = a[index] * b[index];
}

kernel void scalar_mul(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* out     [[buffer(2)]],
                       uint index            [[thread_position_in_grid]])
{
    out[index] = a[index] * (*b);
}

kernel void ewise_div(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* out     [[buffer(2)]],
                      uint index            [[thread_position_in_grid]])
{
    out[index] = a[index] / b[index];
}

kernel void scalar_div(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* out     [[buffer(2)]],
                       uint index            [[thread_position_in_grid]])
{
    out[index] = a[index] / (*b);
}

kernel void scalar_power(device const float* a [[buffer(0)]],
                       device const float* b   [[buffer(1)]],
                       device float* out       [[buffer(2)]],
                       uint index              [[thread_position_in_grid]])
{
    out[index] = pow(a[index], (*b));
}

kernel void ewise_maximum(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* out     [[buffer(2)]],
                          uint index            [[thread_position_in_grid]])
{
    out[index] = max(a[index], b[index]);
}

kernel void scalar_maximum(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* out     [[buffer(2)]],
                           uint index            [[thread_position_in_grid]])
{
    out[index] = max(a[index], (*b));
}

kernel void ewise_eq(device const float* a [[buffer(0)]],
                     device const float* b [[buffer(1)]],
                     device float* out     [[buffer(2)]],
                     uint index            [[thread_position_in_grid]])
{
    out[index] = a[index] == b[index];
}

kernel void scalar_eq(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* out     [[buffer(2)]],
                      uint index            [[thread_position_in_grid]])
{
    out[index] = a[index] == (*b);
}

kernel void ewise_ge(device const float* a  [[buffer(0)]],
                     device const float* b  [[buffer(1)]],
                     device float* out      [[buffer(2)]],
                     uint index             [[thread_position_in_grid]])
{
    out[index] = a[index] >= b[index];
}

kernel void scalar_ge(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* out     [[buffer(2)]],
                      uint index            [[thread_position_in_grid]])
{
    out[index] = a[index] >= (*b);
}

kernel void ewise_log(device const float* a [[buffer(0)]],
                      device float* out     [[buffer(1)]],
                      uint index            [[thread_position_in_grid]])
{
    out[index] = log(a[index]);
}

kernel void ewise_exp(device const float* a [[buffer(0)]],
                      device float* out     [[buffer(1)]],
                      uint index            [[thread_position_in_grid]])
{
    out[index] = exp(a[index]);
}

kernel void ewise_tanh(device const float* a [[buffer(0)]],
                      device float* out      [[buffer(1)]],
                      uint index             [[thread_position_in_grid]])
{
    out[index] = tanh(a[index]);
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

kernel void reduce_max(device const float* a            [[buffer(0)]],
                       device float* out                [[buffer(1)]],
                       device const size_t* reduce_size [[buffer(2)]],
                       uint index                       [[thread_position_in_grid]])
{
    size_t offset = index * (*reduce_size);
    float reduce_max = a[offset];
    for (size_t i = 1; i < (*reduce_size); i++) {
      reduce_max = max(reduce_max, a[i + offset]);
    }
    out[index] = reduce_max;
}

kernel void reduce_sum(device const float* a            [[buffer(0)]],
                       device float* out                [[buffer(1)]],
                       device const size_t* reduce_size [[buffer(2)]],
                       uint index                       [[thread_position_in_grid]])
{
    size_t offset = index * (*reduce_size);
    float reduce_sum = 0;
    for (size_t i = 0; i < (*reduce_size); i++) {
      reduce_sum += a[i + offset];
    }
    out[index] = reduce_sum;
}




