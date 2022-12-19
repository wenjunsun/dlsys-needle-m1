// this code is adapted from https://github.com/larsgeb/m1-gpu-cpp, credited to Lars Gebraad.

#include "MetalOperations.hpp"
#include <iostream>
#include <list>
#include <map>
#include <vector>

MetalOperations::MetalOperations(MTL::Device *device)
{

    _mDevice = device;

    NS::Error *error = nullptr;

    // Load the shader files with a .metal file extension in the project
    auto filepath = NS::String::string("./ops.metallib", NS::ASCIIStringEncoding);
    MTL::Library *opLibrary = _mDevice->newLibrary(filepath, &error);

    if (opLibrary == nullptr)
    {
        std::cout << "Failed to find the default library. Error: "
                  << error->description()->utf8String() << std::endl;
        return;
    }

    // Get all function names
    auto fnNames = opLibrary->functionNames();

    for (size_t i = 0; i < fnNames->count(); i++)
    {

        auto name_nsstring = fnNames->object(i)->description();
        auto name_utf8 = name_nsstring->utf8String();

        // Load function into a map
        functionMap[name_utf8] =
            (opLibrary->newFunction(name_nsstring));

        // Create pipeline from function
        functionPipelineMap[name_utf8] =
            _mDevice->newComputePipelineState(functionMap[name_utf8], &error);

        if (functionPipelineMap[name_utf8] == nullptr)
        {
            std::cout << "Failed to created pipeline state object for "
                      << name_utf8 << ", error "
                      << error->description()->utf8String() << std::endl;
            return;
        }
    }

    std::cout << std::endl;

    _mCommandQueue = _mDevice->newCommandQueue();
    if (_mCommandQueue == nullptr)
    {
        std::cout << "Failed to find the command queue." << std::endl;
        return;
    }
}

void MetalOperations::Blocking1D(std::vector<MTL::Buffer *> buffers,
                                 size_t arrayLength,
                                 const char *method)
{

    MTL::CommandBuffer *commandBuffer = _mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);

    computeEncoder->setComputePipelineState(functionPipelineMap[method]);
    for (std::size_t i = 0; i < buffers.size(); ++i)
    {
        computeEncoder->setBuffer(buffers[i], 0, i);
    }

    NS::UInteger threadGroupSize =
        functionPipelineMap[method]->maxTotalThreadsPerThreadgroup();

    if (threadGroupSize > arrayLength)
        threadGroupSize = arrayLength;

    MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);
    MTL::Size gridSize = MTL::Size::Make(arrayLength, 1, 1);

    // Encode the compute command.
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
    computeEncoder->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
}

void MetalOperations::addArrays(MTL::Buffer *x_array,
                                MTL::Buffer *y_array,
                                MTL::Buffer *r_array,
                                size_t arrayLength)
{
    std::vector<MTL::Buffer *> buffers = {x_array,
                                          y_array,
                                          r_array};
    const char *method = "add_arrays";

    Blocking1D(buffers, arrayLength, method);
}