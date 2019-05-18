cmake_minimum_required(VERSION 3.14)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.1/bin/nvcc CACHE PATH "" FORCE)

include(CheckLanguage)
check_language(CUDA)
#
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

project(CudaLightKernels CUDA)

include_directories(.)
include_directories(../CudaLightKernels)
include_directories(../cub-1.8.0)

add_library(
    CudaLightKernels

    Types.h
    Common.cuh
    Flags.cuh

    BufferInitializer.cu
    BufferInitializer.cuh

    CuBlasWrappers.cu
    CuBlasWrappers.cuh

    CubWrappers.cu
    CubWrappers.cuh

    CuSparseWrappers.cu
    CuSparseWrappers.cuh

    DeviceManager.cu
    DeviceManager.cuh

    ForgeHelpers.cu
    ForgeHelpers.cuh

    MemoryManager.cu
    MemoryManager.cuh
)
target_link_libraries(CudaLightKernels -lcudart -lcublas -lcusolver -lcusparse)

# computational capability
target_compile_options(CudaLightKernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -gencode arch=compute_61,code=sm_61>)

# force 64 bits
target_compile_options(CudaLightKernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -m64>)

# no aliasing
target_compile_options(CudaLightKernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -restrict>)

# host warnings
target_compile_options(CudaLightKernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --compiler-options -Wall>)