cmake_minimum_required(VERSION 3.2)
project(needle C CXX)

# find correct version of Python
execute_process(COMMAND python3-config --prefix
  OUTPUT_VARIABLE Python_ROOT_DIR)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") # added
  execute_process(
    COMMAND bash -c "python3 --version"
    RESULT_VARIABLE _exit_code
    OUTPUT_VARIABLE _python3_str
    OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REPLACE " " ";" VERSION_LIST ${_python3_str})
    list(GET VERSION_LIST 0 _Python)
    list(GET VERSION_LIST 1 _python3_version)
  find_package(Python3 ${_python3_version} EXACT COMPONENTS Development Interpreter REQUIRED) # or whatever python version is installed in conda env
else()
  find_package(Python COMPONENTS Development Interpreter REQUIRED)
endif()
include_directories(${Python_INCLUDE_DIRS})

# find pybind
execute_process(COMMAND python3 -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE __pybind_path
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 PATHS ${__pybind_path})

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") # added
  if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CUDA_STANDARD 14)
  else()
    set(CMAKE_CXX_FLAGS "/std:c++11 -O2 ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CUDA_STANDARD 14)
  endif()
else()
  if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CUDA_STANDARD 14)
  else()
    set(CMAKE_CXX_FLAGS "/std:c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CUDA_STANDARD 14)
  endif()
endif()

include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})
###################
### CPU BACKEND ###
###################
add_library(ndarray_backend_cpu MODULE ndarray_backend/ndarray_backend_cpu.cc)
target_link_libraries(ndarray_backend_cpu PUBLIC ${LINKER_LIBS})
pybind11_extension(ndarray_backend_cpu)
pybind11_strip(ndarray_backend_cpu)


# directly output to ffi folder
set_target_properties(ndarray_backend_cpu
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/needle/backend_ndarray
  CXX_VISIBILITY_PRESET "hidden"
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set_property(TARGET ndarray_backend_cpu PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()


###################
### M1 BACKEND ###
###################
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

  set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  add_definitions(-D_PROJECT_ROOT_PATH=\"${ROOT_DIR}\")
  find_library(METAL Metal)
  find_library(FOUNDATION Foundation)
  find_library(METALKIT MetalKit)
  add_library(ndarray_backend_m1 MODULE ndarray_backend/ndarray_backend_m1.cpp ndarray_backend/MetalOperations.cpp ndarray_backend/MetalOperations.hpp)
  target_include_directories(ndarray_backend_m1 PRIVATE ./metal-cpp)
  target_link_libraries(ndarray_backend_m1 PUBLIC ${LINKER_LIBS} ${METAL} ${FOUNDATION} ${METALKIT})
  target_compile_options(ndarray_backend_m1 PRIVATE -std=c++17 -stdlib=libc++ -O2 -fno-objc-arc)
  pybind11_extension(ndarray_backend_m1)
  pybind11_strip(ndarray_backend_m1)


  # directly output to ffi folder
  set_target_properties(ndarray_backend_m1
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/needle/backend_ndarray
    CXX_VISIBILITY_PRESET "hidden"
  )

  set_property(TARGET ndarray_backend_m1 PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()


####################
### CUDA BACKEND ###
####################
find_package(CUDA)
if(CUDA_FOUND)
  message(STATUS "Found cuda, building cuda backend")

  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY})

  # invoke nvidia smi to detect if we really have a GPU
  execute_process(COMMAND "nvidia-smi" ERROR_QUIET  RESULT_VARIABLE NV_RET)
  if(NV_RET EQUAL "0")
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
  else()
    # set to 3.7 the flag of K80
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.7)
  endif()

  # set arch flags properly
  CUDA_ADD_LIBRARY(ndarray_backend_cuda MODULE ndarray_backend/ndarray_backend_cuda.cu OPTIONS ${ARCH_FLAGS})

  target_link_libraries(ndarray_backend_cuda ${LINKER_LIBS})
  pybind11_extension(ndarray_backend_cuda)
  pybind11_strip(ndarray_backend_cuda)

  # directly output to ffi folder
  set_target_properties(ndarray_backend_cuda
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/needle/backend_ndarray
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
)

endif()
