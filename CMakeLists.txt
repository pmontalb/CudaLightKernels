include_guard()

cmake_minimum_required(VERSION 3.14)
project(CudaLightKernels)

include(${CMAKE_SOURCE_DIR}/cmake/All.cmake)
set(LANGUAGES_USE_CUDA ON CACHE BOOL "" FORCE)

create_cuda_library(
    NAME
        CudaLightKernels
    SOURCES
        BufferInitializer.cu CuBlasWrappers.cu CubWrappers.cu
        CuSparseWrappers.cu DeviceManager.cu ForgeHelpers.cu MemoryManager.cu
    PUBLIC_INCLUDE_DIRECTORIES
        .
)
