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

class MetalOperations
{
public:
    MTL::Device *_mDevice;

    MetalOperations(MTL::Device *device);

    // a wrapper method for using metal kernel for 1D operations.
    // buffers = the buffers we pass into the M1 GPU. Usually we have 3 buffers: x, y, and return array.
    // arrayLength = number of elements in each of the array we do computation on.
    // method = name of the kernel method declared in ops.metal
    void Blocking1D(std::vector<MTL::Buffer *> buffers,
                    size_t arrayLength,
                    const char *method);

    void addArrays(MTL::Buffer *x_array,
                   MTL::Buffer *y_array,
                   MTL::Buffer *r_array,
                   size_t arrayLength);

private:
    std::map<std::string, MTL::Function *> functionMap;
    std::map<std::string, MTL::ComputePipelineState *> functionPipelineMap;

    // The command queue used to pass commands to the device.
    MTL::CommandQueue *_mCommandQueue;
};
