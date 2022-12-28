// this code is adapted from https://github.com/larsgeb/m1-gpu-cpp, credited to Lars Gebraad.

/*
CPP translation of original Objective-C MetalAdder.h. Some stuff has been moved over
here from the cpp file. Source: https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu?language=objc

Original distribution license: LICENSE-original.txt.

Abstract:
A class to manage all of the Metal objects this app creates.
*/
#pragma once

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"

#include "map"

typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

class MetalOperations
{
public:
    MTL::Device *_mDevice;

    MetalOperations(MTL::Device *device);

    // a wrapper method for using metal kernel for 1D operations.
    // buffers = the buffers we pass into the M1 GPU.
    // arrayLength = number of elements in the array we do computation on.
    // method = name of the kernel method declared in ops.metal
    void Blocking1D(std::vector<MTL::Buffer *> buffers,
                    size_t arrayLength,
                    const char *method);
    void Blocking2D(std::vector<MTL::Buffer *> buffers,
                    size_t rows,
                    size_t columns,
                    const char *method);

    // Fill operation
    void Fill(MTL::Buffer *out, scalar_t val, size_t arrayLength, const char *method);

    // Compact && Setitem operation
    void Compact(MTL::Buffer *a, MTL::Buffer *out, std::vector<int32_t> shape, std::vector<int32_t> strides,
                 size_t offset, size_t arrayLength, const char *method);
    void EwiseSetitem(MTL::Buffer *a, MTL::Buffer *out, std::vector<int32_t> shape, std::vector<int32_t> strides,
                      size_t offset, size_t arrayLength, const char *method);
    void ScalarSetitem(MTL::Buffer *out, scalar_t val, std::vector<int32_t> shape, std::vector<int32_t> strides,
                       size_t offset, size_t arrayLength, const char *method);

    // Elementwise operation
    void EwiseOp1(MTL::Buffer *a, MTL::Buffer *out, size_t arrayLength, const char *method);
    void EwiseOp2(MTL::Buffer *a, MTL::Buffer *b, MTL::Buffer *out, size_t arrayLength, const char *method);

    // Scalar operation
    void ScalarOp(MTL::Buffer *a, scalar_t b, MTL::Buffer *out, size_t arrayLength, const char *method);

    // Matrix mulplication
    void MatMul(MTL::Buffer *a, MTL::Buffer *b, MTL::Buffer *out, uint32_t M, uint32_t N, uint32_t P, const char *method);

    // Reduce operation
    void ReduceOp(MTL::Buffer *a, MTL::Buffer *out, size_t reduce_size, size_t arrayLength, const char *method);

private:
    std::map<std::string, MTL::Function *> functionMap;
    std::map<std::string, MTL::ComputePipelineState *> functionPipelineMap;

    // The command queue used to pass commands to the device.
    MTL::CommandQueue *_mCommandQueue;
};
