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

// get the M1 GPU device
MTL::Device *device = MTL::CreateSystemDefaultDevice();
// Load the metal library and initialize the MetalOperations object.  
MetalOperations *MetalOps = new MetalOperations(device);

// M1 is not like CUDA where we need to copy memory from CPU to GPU,
// instead, M1 device has a shared buffer between CPU and GPU, and
// CPU can move data to the shared buffer, and it will be then visible
// to GPU, and vice versa.
struct M1Array {
  M1Array(const size_t size) {
    // create a buffer with (size * ELEM_SIZE) bytes that is shared between CPU and GPU.
    array_MTL = device->newBuffer(size * ELEM_SIZE, MTL::ResourceStorageModeShared);
    assert(array_MTL != nullptr);
    
    // array_MTL->content() gets the C++ reference to the buffer shared with M1 GPU.
    ptr = (scalar_t*) array_MTL->contents();
    this->size = size;
  }
  ~M1Array() {array_MTL->release();}
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
  MTL::Buffer *array_MTL;
};

////////////////////////////////////////////////////////////////////////////////
// Fill operation
////////////////////////////////////////////////////////////////////////////////

void Fill(M1Array* out, scalar_t val) {
  MetalOps->Fill(out->array_MTL, val, out->size, "fill");
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem
////////////////////////////////////////////////////////////////////////////////

void Compact(const M1Array& a, M1Array* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  MetalOps->Compact(a.array_MTL, out->array_MTL, shape, strides, offset, out->size, "compact");
}

void EwiseSetitem(const M1Array& a, M1Array* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  MetalOps->EwiseSetitem(a.array_MTL, out->array_MTL, shape, strides, offset, a.size, "ewise_setitem");
}

void ScalarSetitem(size_t size, scalar_t val, M1Array* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  MetalOps->ScalarSetitem(out->array_MTL, val, shape, strides, offset, size, "scalar_setitem");
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

void EwiseAdd(const M1Array& a, const M1Array& b, M1Array* out) {
  MetalOps->EwiseOp2(a.array_MTL, b.array_MTL, out->array_MTL, out->size, "ewise_add");
}

void ScalarAdd(const M1Array& a, scalar_t b, M1Array* out) {
  MetalOps->ScalarOp(a.array_MTL, b, out->array_MTL, out->size, "scalar_add");
}

void EwiseMul(const M1Array& a, const M1Array& b, M1Array* out) {
  MetalOps->EwiseOp2(a.array_MTL, b.array_MTL, out->array_MTL, out->size, "ewise_mul");
}

void ScalarMul(const M1Array& a, scalar_t b, M1Array* out) {
  MetalOps->ScalarOp(a.array_MTL, b, out->array_MTL, out->size, "scalar_mul");
}

void EwiseDiv(const M1Array& a, const M1Array& b, M1Array* out) {
  MetalOps->EwiseOp2(a.array_MTL, b.array_MTL, out->array_MTL, out->size, "ewise_div");
}

void ScalarDiv(const M1Array& a, scalar_t b, M1Array* out) {
  MetalOps->ScalarOp(a.array_MTL, b, out->array_MTL, out->size, "scalar_div");
}

void ScalarPower(const M1Array& a, scalar_t b, M1Array* out) {
  MetalOps->ScalarOp(a.array_MTL, b, out->array_MTL, out->size, "scalar_power");
}

void EwiseMaximum(const M1Array& a, const M1Array& b, M1Array* out) {
  MetalOps->EwiseOp2(a.array_MTL, b.array_MTL, out->array_MTL, out->size, "ewise_maximum");
}

void ScalarMaximum(const M1Array& a, scalar_t b, M1Array* out) {
  MetalOps->ScalarOp(a.array_MTL, b, out->array_MTL, out->size, "scalar_maximum");
}

void EwiseEq(const M1Array& a, const M1Array& b, M1Array* out) {
  MetalOps->EwiseOp2(a.array_MTL, b.array_MTL, out->array_MTL, out->size, "ewise_eq");
}

void ScalarEq(const M1Array& a, scalar_t b, M1Array* out) {
  MetalOps->ScalarOp(a.array_MTL, b, out->array_MTL, out->size, "scalar_eq");
}

void EwiseGe(const M1Array& a, const M1Array& b, M1Array* out) {
  MetalOps->EwiseOp2(a.array_MTL, b.array_MTL, out->array_MTL, out->size, "ewise_ge");
}

void ScalarGe(const M1Array& a, scalar_t b, M1Array* out) {
  MetalOps->ScalarOp(a.array_MTL, b, out->array_MTL, out->size, "scalar_ge");
}

void EwiseLog(const M1Array& a, M1Array* out) {
  MetalOps->EwiseOp1(a.array_MTL, out->array_MTL, out->size, "ewise_log");
}

void EwiseExp(const M1Array& a, M1Array* out) {
  MetalOps->EwiseOp1(a.array_MTL, out->array_MTL, out->size, "ewise_exp");
}

void EwiseTanh(const M1Array& a, M1Array* out) {
  MetalOps->EwiseOp1(a.array_MTL, out->array_MTL, out->size, "ewise_tanh");
}

////////////////////////////////////////////////////////////////////////////////
// Matrix mulplication
////////////////////////////////////////////////////////////////////////////////
void Matmul(const M1Array& a, const M1Array& b, M1Array* out, uint32_t M, uint32_t N, uint32_t P) {
  MetalOps->MatMul(a.array_MTL, b.array_MTL, out->array_MTL, M, N, P, "matmul_naive");
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////
void ReduceMax(const M1Array& a, M1Array* out, size_t reduce_size) {
  MetalOps->ReduceOp(a.array_MTL, out->array_MTL, reduce_size, out->size, "reduce_max");
}

void ReduceSum(const M1Array& a, M1Array* out, size_t reduce_size) {
  MetalOps->ReduceOp(a.array_MTL, out->array_MTL, reduce_size, out->size, "reduce_sum");
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

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);

  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);
  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
//   m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}