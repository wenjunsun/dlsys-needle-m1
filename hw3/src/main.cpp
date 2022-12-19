// this code is adapted from https://github.com/larsgeb/m1-gpu-cpp, credited to Lars Gebraad.

#include <iostream>
#include <iomanip>
#include <omp.h>
#include <assert.h>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"

#include "MetalOperations.hpp"

typedef std::chrono::microseconds time_unit;
auto unit_name = "microseconds";

const unsigned int bufferSize = 3 * sizeof(float);

int main(int argc, char *argv[])
{

    // Set up objects and buffers ------------------------------------------------------

    MTL::Device *device = MTL::CreateSystemDefaultDevice();

    std::cout << "Running on " << device->name()->utf8String() << std::endl;

    // MTL buffers to hold data.
    MTL::Buffer *a_MTL = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    MTL::Buffer *b_MTL = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    MTL::Buffer *c_MTL = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);

    // Get a C++-style reference to the buffer
    auto a_CPP = (float *)a_MTL->contents();
    auto b_CPP = (float *)b_MTL->contents();
    auto c_CPP = (float *)c_MTL->contents();

    // operations on a_CPP (CPU) affects data on a_MTL (GPU) as well, since buffer is shared.
    a_CPP[0] = 1;
    a_CPP[1] = 1;
    a_CPP[2] = 1;
    
    b_CPP[0] = 2;
    b_CPP[1] = 3;
    b_CPP[2] = 4;
    
    c_CPP[0] = 0;
    c_CPP[1] = 0;
    c_CPP[2] = 0;

    // Create GPU object
    MetalOperations *arrayOps = new MetalOperations(device);
    
    std::cout << "before add arrays (M1 GPU):" << std::endl;
    std::cout << "a_CPP:";
    for (int i = 0; i < 3; i++) {
        std::cout << a_CPP[i] << ",";
    }
    std::cout << std::endl;
    std::cout << "b_CPP:";
    for (int i = 0; i < 3; i++) {
        std::cout << b_CPP[i] << ",";
    }
    std::cout << std::endl;

    std::cout << "c_CPP:";
    for (int i = 0; i < 3; i++) {
        std::cout << c_CPP[i] << ",";
    }
    std::cout << std::endl;
    

    arrayOps->addArrays(a_MTL, b_MTL, c_MTL, 3);

    std::cout << "after add arrays (M1 GPU):" << std::endl;
    std::cout << "a_CPP:";
    for (int i = 0; i < 3; i++) {
        std::cout << a_CPP[i] << ",";
    }
    std::cout << std::endl;
    std::cout << "b_CPP:";
    for (int i = 0; i < 3; i++) {
        std::cout << b_CPP[i] << ",";
    }
    std::cout << std::endl;
    std::cout << "c_CPP:";
    for (int i = 0; i < 3; i++) {
        std::cout << c_CPP[i] << ",";
    }
    std::cout << std::endl;
}