// pybind libraries
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// metal libraries
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"

// our wrapper for writing metal code easier
#include "MetalOperations.hpp"

namespace needle {
namespace m1 {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

// M1 is not like CUDA where we need to copy memory from CPU to GPU,
// instead, M1 device has a shared buffer between CPU and GPU, and
// CPU can move data to the shared buffer, and it will be then visible
// to GPU, and vice versa.
struct M1Array {
  M1Array(const size_t size) {
    // get the enough memory for the CPU version of the buffer.
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();

    // get the M1 GPU device
    device = MTL::CreateSystemDefaultDevice();
    // create a buffer with (size * ELEM_SIZE) bytes that is shared between CPU and GPU.
    array_MTL = device->newBuffer(size * ELEM_SIZE, MTL::ResourceStorageModeManaged);
    assert(array_MTL != nullptr);
    
    // array_MTL->content() gets the C++ reference to the buffer shared with M1 GPU.
    ptr = (scalar_t*) array_MTL->contents();
    this->size = size;
  }
  ~M1Array() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
  MTL::Device *device;
  MTL::Buffer *array_MTL;
};



void Fill(M1Array* out, scalar_t val) {
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
  //TODO: implement M1 GPU version here.
}

void EwiseAdd(const M1Array& a, const M1Array& b, M1Array* out) {
  MetalOperations *arrayOps = new MetalOperations(out->device);
  arrayOps->addArrays(a.array_MTL, b.array_MTL, out->array_MTL, out->size);
}

} // namespace m1
} // namespace needle


PYBIND11_MODULE(ndarray_backend_m1, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace m1;

  m.attr("__device_name__") = "m1";
  m.attr("__tile_size__") = TILE;

  py::class_<M1Array>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &M1Array::ptr_as_int)
      .def_readonly("size", &M1Array::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const M1Array& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, M1Array* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

//   m.def("fill", Fill);
//   m.def("compact", Compact);
//   m.def("ewise_setitem", EwiseSetitem);
//   m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
//   m.def("scalar_add", ScalarAdd);

//   m.def("ewise_mul", EwiseMul);
//   m.def("scalar_mul", ScalarMul);
//   m.def("ewise_div", EwiseDiv);
//   m.def("scalar_div", ScalarDiv);
//   m.def("scalar_power", ScalarPower);

//   m.def("ewise_maximum", EwiseMaximum);
//   m.def("scalar_maximum", ScalarMaximum);
//   m.def("ewise_eq", EwiseEq);
//   m.def("scalar_eq", ScalarEq);
//   m.def("ewise_ge", EwiseGe);
//   m.def("scalar_ge", ScalarGe);

//   m.def("ewise_log", EwiseLog);
//   m.def("ewise_exp", EwiseExp);
//   m.def("ewise_tanh", EwiseTanh);

//   m.def("matmul", Matmul);
//   m.def("matmul_tiled", MatmulTiled);

//   m.def("reduce_max", ReduceMax);
//   m.def("reduce_sum", ReduceSum);
}