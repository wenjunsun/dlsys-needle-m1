// this code is adapted from https://github.com/larsgeb/m1-gpu-cpp, credited to Lars Gebraad.

#include "MetalOperations.hpp"
#include <iostream>
#include <list>
#include <map>
#include <vector>

#ifdef _PROJECT_ROOT_PATH
    #define CUR_ROOT_DIR _PROJECT_ROOT_PATH
#else
    #define CUR_ROOT_DIR "./"
#endif

template<class T>
MTL::Buffer* ScalarToMTLBuffer(T scalar, MTL::Device *device)
{
    MTL::Buffer *buffer = device->newBuffer(sizeof(T), MTL::ResourceStorageModeShared);
    assert(buffer != nullptr);
    auto bufferPtr = (T*)buffer->contents();
    *bufferPtr = scalar;
    return buffer;
}

template<class T>
MTL::Buffer* VecToMTLBuffer(std::vector<T> vec, MTL::Device *device)
{
    MTL::Buffer *buffer = device->newBuffer(vec.size() * sizeof(T), MTL::ResourceStorageModeShared);
    assert(buffer != nullptr);
    auto bufferPtr = (T*)buffer->contents();
    for (int i = 0; i < vec.size(); i++)
    {
        bufferPtr[i] = vec[i];
    }
    return buffer;
}

MetalOperations::MetalOperations(MTL::Device *device)
{

    _mDevice = device;

    NS::Error *error = nullptr;

    // Load the shader files with a .metal file extension in the project
    std::string s1 = CUR_ROOT_DIR;
    std::string s2 = "/build/";
    std::string s3 = "ops.metallib";
    std::string libpath = s1 + s2 + s3;
    auto filepath = NS::String::string(libpath.c_str(), NS::ASCIIStringEncoding);
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
    
    commandBuffer->release();
    computeEncoder->release();
}

void MetalOperations::Blocking2D(std::vector<MTL::Buffer *> buffers,
                                 size_t rows,
                                 size_t columns,
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

    // Both of these values must be the same!
    const int x_threads_per_group = 8;
    const int y_threads_per_group = 8;
    assert(x_threads_per_group == y_threads_per_group);
    
    // The number of thread groups (i.e., blocks) per grid.
    const int x_group_count = (columns + x_threads_per_group - 1) / x_threads_per_group;
    const int y_group_count = (rows + y_threads_per_group - 1) / y_threads_per_group;
    MTL::Size thread_group_count = MTL::Size::Make(x_group_count, y_group_count, 1);
    MTL::Size threadgroupSize = MTL::Size::Make(x_threads_per_group, y_threads_per_group, 1);
    computeEncoder->dispatchThreadgroups(thread_group_count, threadgroupSize);

    // Encode the compute command.
    computeEncoder->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    commandBuffer->release();
    computeEncoder->release();
}

void MetalOperations::Fill(MTL::Buffer *out,
                           scalar_t val,
                           size_t arrayLength,
                           const char *method)
{
    auto val_buffer = ScalarToMTLBuffer(val, _mDevice);
    std::vector<MTL::Buffer *> buffers = {out, val_buffer};
    Blocking1D(buffers, arrayLength, method);
    val_buffer->release();
}

void MetalOperations::Compact(MTL::Buffer *a,
                              MTL::Buffer *out,
                              std::vector<int32_t> shape,
                              std::vector<int32_t> strides,
                              size_t offset,
                              size_t arrayLength,
                              const char *method)
{
    auto shape_buffer = VecToMTLBuffer(shape, _mDevice);
    auto strides_buffer = VecToMTLBuffer(strides, _mDevice);
    auto dim_buffer = ScalarToMTLBuffer(shape.size(), _mDevice);
    auto offset_buffer = ScalarToMTLBuffer(offset, _mDevice);
    std::vector<MTL::Buffer *> buffers = {a, out, shape_buffer, strides_buffer, dim_buffer, offset_buffer};
    Blocking1D(buffers, arrayLength, method);
    shape_buffer->release();
    strides_buffer->release();
    dim_buffer->release();
    offset_buffer->release();
}

void MetalOperations::EwiseSetitem(MTL::Buffer *a,
                                   MTL::Buffer *out,
                                   std::vector<int32_t> shape,
                                   std::vector<int32_t> strides,
                                   size_t offset,
                                   size_t arrayLength,
                                   const char *method)
{
    auto shape_buffer = VecToMTLBuffer(shape, _mDevice);
    auto strides_buffer = VecToMTLBuffer(strides, _mDevice);
    auto dim_buffer = ScalarToMTLBuffer(shape.size(), _mDevice);
    auto offset_buffer = ScalarToMTLBuffer(offset, _mDevice);
    std::vector<MTL::Buffer *> buffers = {a, out, shape_buffer, strides_buffer, dim_buffer, offset_buffer};
    Blocking1D(buffers, arrayLength, method);
    shape_buffer->release();
    strides_buffer->release();
    dim_buffer->release();
    offset_buffer->release();
}

void MetalOperations::ScalarSetitem(MTL::Buffer *out,
                                    scalar_t val,
                                    std::vector<int32_t> shape,
                                    std::vector<int32_t> strides,
                                    size_t offset,
                                    size_t arrayLength,
                                    const char *method)
{
    auto val_buffer = ScalarToMTLBuffer(val, _mDevice);
    auto shape_buffer = VecToMTLBuffer(shape, _mDevice);
    auto strides_buffer = VecToMTLBuffer(strides, _mDevice);
    auto dim_buffer = ScalarToMTLBuffer(shape.size(), _mDevice);
    auto offset_buffer = ScalarToMTLBuffer(offset, _mDevice);
    std::vector<MTL::Buffer *> buffers = {out, val_buffer, shape_buffer, strides_buffer, dim_buffer, offset_buffer};
    Blocking1D(buffers, arrayLength, method);
    val_buffer->release();
    shape_buffer->release();
    strides_buffer->release();
    dim_buffer->release();
    offset_buffer->release();
}


void MetalOperations::EwiseOp1(MTL::Buffer *a,
                               MTL::Buffer *out,
                               size_t arrayLength,
                               const char *method)
{
    std::vector<MTL::Buffer *> buffers = {a, out};
    Blocking1D(buffers, arrayLength, method);
}

void MetalOperations::EwiseOp2(MTL::Buffer *a,
                               MTL::Buffer *b,
                               MTL::Buffer *out,
                               size_t arrayLength,
                               const char *method)
{
    std::vector<MTL::Buffer *> buffers = {a, b, out};
    Blocking1D(buffers, arrayLength, method);
}

void MetalOperations::ScalarOp(MTL::Buffer *a,
                               scalar_t b,
                               MTL::Buffer *out,
                               size_t arrayLength,
                               const char *method)
{
    auto scalar_buffer = ScalarToMTLBuffer(b, _mDevice);
    std::vector<MTL::Buffer *> buffers = {a, scalar_buffer, out};
    Blocking1D(buffers, arrayLength, method);
    scalar_buffer->release();
}


void MetalOperations::ReduceOp(MTL::Buffer *a,
                               MTL::Buffer *out,
                               size_t reduce_size,
                               size_t arrayLength,
                               const char *method)
{
    auto reduce_size_buffer = ScalarToMTLBuffer(reduce_size, _mDevice);
    std::vector<MTL::Buffer *> buffers = {a, out, reduce_size_buffer};
    Blocking1D(buffers, arrayLength, method);
    reduce_size_buffer->release();
}


void MetalOperations::MatMul(MTL::Buffer *a,
                             MTL::Buffer *b,
                             MTL::Buffer *out,
                             uint32_t M,
                             uint32_t N,
                             uint32_t P,
                             const char *method)
{
    auto M_buffer = ScalarToMTLBuffer(M, _mDevice);
    auto N_buffer = ScalarToMTLBuffer(N, _mDevice);
    auto P_buffer = ScalarToMTLBuffer(P, _mDevice);
    std::vector<MTL::Buffer *> buffers = {a, b, out, M_buffer, N_buffer, P_buffer};
    Blocking2D(buffers, M, P, method);
    M_buffer->release();
    N_buffer->release();
    P_buffer->release();
}