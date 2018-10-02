#include "BufferInitializer.cuh"
#include "DeviceManager.cuh"
#include <stdio.h>

EXTERN_C
{
	EXPORT int _Initialize(MemoryBuffer buf, const double value)
	{
		switch (buf.mathDomain)
		{
		case MathDomain::Float:
			CUDA_CALL_SINGLE(__Initialize__<float>, (float*)buf.pointer, buf.size, (float)value);
			break;
		case MathDomain::Double:
			CUDA_CALL_DOUBLE(__Initialize__<double>, (double*)buf.pointer, buf.size, value);
			break;
		case MathDomain::Int:
			CUDA_CALL_SINGLE(__Initialize__<int>, (int*)buf.pointer, buf.size, (int)value);
			break;
		default:
			return CudaKernelException::_NotImplementedException;
		}

		return cudaGetLastError();
	}
	EXPORT int _InitializeRaw(const ptr_t pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const double value)
	{
		MemoryBuffer buf(pointer, size, memorySpace, mathDomain);
		return _Initialize(buf, value);
    }

	EXPORT int _LinSpace(MemoryBuffer buf, const double x0, const double x1)
	{
		const double dx = (x1 - x0) / (buf.size - 1);

		switch (buf.mathDomain)
		{
		case MathDomain::Float:
			CUDA_CALL_SINGLE(__LinSpace__<float>, (float*)buf.pointer, buf.size, (float)x0, (float)dx);
			break;
		case MathDomain::Double:
			CUDA_CALL_DOUBLE(__LinSpace__<double>, (double*)buf.pointer, buf.size, (double)x0, (double)dx);
			break;
		default:
			return CudaKernelException::_NotImplementedException;
		}

		return cudaGetLastError();
	}
	EXPORT int _LinSpaceRaw(const ptr_t pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const double x0, const double x1)
	{
		MemoryBuffer buf(pointer, size, memorySpace, mathDomain);
		return _LinSpace(buf, x0, x1);
	}

	EXPORT int _RandUniform(MemoryBuffer buf, const unsigned seed)
	{
		dim3 block, grid;
		const unsigned halfSz = (buf.size + 1) >> 1;
		detail::GetBestDimension(block, grid, N_BLOCKS_SINGLE, halfSz);

		curandState *states = 0;
		int err = cudaMalloc((void **)&states, grid.x * block.x * sizeof(curandState));
		if (err)
			return err;
		CUDA_CALL_XY(__SetupCuRand__, grid, block, states, halfSz, seed);

		switch (buf.mathDomain)
		{
		case MathDomain::Float:
			CUDA_CALL_XYZ(__RandUniform__<float>, grid, block, block.x * sizeof(unsigned int), (float*)buf.pointer, states, halfSz, buf.size);
			break;
		case MathDomain::Double:
			CUDA_CALL_XYZ(__RandUniform__<double>, grid, block, block.x * sizeof(unsigned int), (double*)buf.pointer, states, halfSz, buf.size);
			break;
		case MathDomain::Int:
			CUDA_CALL_XYZ(__RandUniform__<int>, grid, block, block.x * sizeof(unsigned int), (int*)buf.pointer, states, halfSz, buf.size);
			break;
		default:
			return CudaKernelException::_NotImplementedException;
		}

		cudaFree(states);

		return cudaGetLastError();
	}
	EXPORT int _RandUniformRaw(const ptr_t pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const unsigned seed)
	{
		MemoryBuffer buf(pointer, size, memorySpace, mathDomain);
		return _RandUniform(buf,seed);
	}

	EXPORT int _RandNormal(MemoryBuffer buf, const unsigned seed)
	{
		dim3 block, grid;
		const unsigned halfSz = buf.size >> 1;
		detail::GetBestDimension(block, grid, N_BLOCKS_SINGLE, halfSz);

		curandState *states = 0;
		int err = cudaMalloc((void **)&states, grid.x * block.x * sizeof(curandState));
		if (err)
			return err;
		CUDA_CALL_XY(__SetupCuRand__, grid, block, states, halfSz, seed);

		switch (buf.mathDomain)
		{
		case MathDomain::Float:
			CUDA_CALL_XYZ(__RandNormal__<float>, grid, block, block.x * sizeof(unsigned int), (float*)buf.pointer, states, halfSz, buf.size);
			break;
		case MathDomain::Double:
			CUDA_CALL_XYZ(__RandNormal__<double>, grid, block, block.x * sizeof(unsigned int), (double*)buf.pointer, states, halfSz, buf.size);
			break;
		case MathDomain::Int:
			CUDA_CALL_XYZ(__RandNormal__<int>, grid, block, block.x * sizeof(unsigned int), (int*)buf.pointer, states, halfSz, buf.size);
			break;
		default:
			return CudaKernelException::_NotImplementedException;
		}

		cudaFree(states);

		return cudaGetLastError();
	}
	EXPORT int _RandNormalRaw(const ptr_t pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const unsigned seed)
	{
		MemoryBuffer buf(pointer, size, memorySpace, mathDomain);
		return _RandNormal(buf, seed);
	}

	EXPORT int _Eye(MemoryTile buf)
	{
		dim3 blockDim(16, 16);
		dim3 gridDim((buf.nRows + 15) / 16, (buf.nRows + 15) / 16);

		switch (buf.mathDomain)
		{
		case MathDomain::Float:
			CUDA_CALL_XY(__Eye__<float>, gridDim, blockDim, (float*)buf.pointer, buf.nRows);
			break;
		case MathDomain::Double:
			CUDA_CALL_XY(__Eye__<double>, gridDim, blockDim, (double*)buf.pointer, buf.nRows);
			break;
		default:
			return CudaKernelException::_NotImplementedException;
		}
			
		return cudaGetLastError();
	}
	EXPORT int _EyeRaw(const ptr_t pointer, const unsigned nRows, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryTile tile(pointer, nRows, nRows, memorySpace, mathDomain);
		return _Eye(tile);
	}

	EXPORT int _OnesUpperTriangular(MemoryTile buf)
	{
		dim3 blockDim(16, 16);
		dim3 gridDim((buf.nRows + 15) / 16, (buf.nRows + 15) / 16);

		switch (buf.mathDomain)
		{
		case MathDomain::Float:
			CUDA_CALL_XY(__OnesUpperTriangular__<float>, gridDim, blockDim, (float*)buf.pointer, buf.nRows);
			break;
		case MathDomain::Double:
			CUDA_CALL_XY(__OnesUpperTriangular__<double>, gridDim, blockDim, (double*)buf.pointer, buf.nRows);
			break;
		default:
			return -1;
		}

		return cudaGetLastError();
	}
	EXPORT int _OnesUpperTriangularRaw(const ptr_t pointer, const unsigned nRows, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryTile tile(pointer, nRows, nRows, memorySpace, mathDomain);
		return _OnesUpperTriangular(tile);
	}
}

template <typename T>
GLOBAL void __Initialize__(T* RESTRICT ptr, const ptr_t sz, const T value)
{
	CUDA_FUNCTION_PROLOGUE
	CUDA_FOR_LOOP_PROLOGUE

		ptr[i] = value;

	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __LinSpace__(T* RESTRICT ptr, const ptr_t sz, const T x0, const T dx)
{
	CUDA_FUNCTION_PROLOGUE
	CUDA_FOR_LOOP_PROLOGUE

		ptr[i] = x0 + i * dx;

	CUDA_FOR_LOOP_EPILOGUE
}

GLOBAL void __SetupCuRand__(CURAND_STATE_PTR states, const ptr_t sz, const unsigned seed)
{
	// Determine thread ID
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Initialise the RNG
	curand_init(seed, tid, 0, &states[tid]);
}

template <typename T>
GLOBAL void __RandUniform__(T* RESTRICT ptr, CURAND_STATE_PTR states, const unsigned sz, const unsigned fullSz)
{
	CUDA_FUNCTION_PROLOGUE

	curandState localState = states[tid];

	CUDA_FOR_LOOP_PROLOGUE

		ptr[2 * i] = static_cast<T>(curand_uniform(&localState));
	    if (2 * i + 1 < fullSz)
			ptr[2 * i + 1] = static_cast<T>(1.0) - ptr[2 * i];

	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __RandNormal__(T* RESTRICT ptr, CURAND_STATE_PTR states, const unsigned sz, const unsigned fullSz)
{
	CUDA_FUNCTION_PROLOGUE

	curandState localState = states[tid];

	CUDA_FOR_LOOP_PROLOGUE

		ptr[2 * i] = static_cast<T>(curand_normal(&localState));
	    if (2 * i + 1 < fullSz)
		    ptr[2 * i + 1] = -ptr[2 * i];

	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __Eye__(T* RESTRICT A, const size_t sz)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (j < sz && i < sz)
	{
		if (i == j)
			A[i + sz * j] = static_cast<T>(1.0);
		else
			A[i + sz * j] = static_cast<T>(0.0);
	}
}

template <typename T>
GLOBAL void __OnesUpperTriangular__(T* RESTRICT A, size_t sz)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (j < sz && i < sz)
	{
		if (i <= j)
			A[i + sz * j] = static_cast<T>(1.0);
		else
			A[i + sz * j] = static_cast<T>(0.0);
	}
}