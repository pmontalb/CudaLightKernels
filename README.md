# CudaLightKernels

This API is a collection of common CUDA kernels calls that I use in my programs. This is by no means a fully-fledged library. It simplifies the code and it reduces the overhead in calling native CUDA code.

This library should be changed not too often, so that once the binary is compiled, it can be paired with a manager for being used in different programming languages. 

## Types
- <i>MemorySpace</i>: defines whether a buffer will be allocated: host or device
- <i>MathDomain</i>: defines the buffer's type: integer, single or double precision
- <i>MatrixOperation</i>: defines the matrix operation type: transpose or no-op

## Buffer classes
- <i>MemoryBuffer</i>: a buffer is identified by its memory space, math domain and size. The combination of this 3 gives you a pointer that points to the allocated memory

- <i>MemoryTile</i>: helper class that represents two dimensional buffers, and introduces the concept of rows and columns

- <i>MemoryCube</i>: helper class that represents three dimensional buffers

## Sparse buffer classes
- <i>SparseMemoryBuffer</i>: helper class that represents sparse buffers, for being used by cuSparse

- <i>SparseMemoryTile</i>: helper class that represents Compressed Sparse Row (CSR) matrices

## Kernels structure
Since there's no name mangling, I decided to use the convention of having every function starting with a leading underscore. This way you can have a helper manager that defines the same function with no underscores that just wraps the kernel call and checks the return value.

I provided basic functionality for allocating/deallocating buffers, and calling cuBlas/cuSparse kernels
