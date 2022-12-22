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
                      device const float* b [[buffer(1)]],
                      device float* out     [[buffer(2)]],
                      uint index            [[thread_position_in_grid]])
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

