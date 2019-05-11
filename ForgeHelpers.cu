#include "ForgeHelpers.cuh"

EXTERN_C
{
	EXPORT int _MakePair(MemoryBuffer z, const MemoryBuffer x, const MemoryBuffer y)
	{
		assert(z.mathDomain == MathDomain::Float);
		switch (z.mathDomain)
		{
			case MathDomain::Float:
			{
				CUDA_CALL_SINGLE(__MakePair__, (float*)z.pointer, (float*)x.pointer, (float*)y.pointer, x.size);
				break;
			}
			case MathDomain::Double:
			{
				CUDA_CALL_DOUBLE(__MakePair__, (float*)z.pointer, (double*)x.pointer, (double*)y.pointer, x.size);
				break;
			}
			case MathDomain::Int:
			{
				CUDA_CALL_SINGLE(__MakePair__, (float*)z.pointer, (int*)x.pointer, (int*)y.pointer, x.size);
				break;
			}
			default:
				return CudaKernelException::_NotImplementedException;
		}

		return cudaGetLastError();
	}

	EXPORT int _MakeTriple(MemoryBuffer v, const MemoryBuffer x, const MemoryBuffer y, const MemoryBuffer z)
	{
		assert(v.mathDomain == MathDomain::Float);
		assert(x.size * y.size == z.size);
		switch (z.mathDomain)
		{
			case MathDomain::Float:
			{
				CUDA_CALL_SINGLE(__MakeTriple__, (float*)v.pointer, (float*)x.pointer, (float*)y.pointer, (float*)z.pointer, x.size, y.size);
				break;
			}
			case MathDomain::Double:
			{
				CUDA_CALL_DOUBLE(__MakeTriple__, (float*)v.pointer, (double*)x.pointer, (double*)y.pointer, (double*)z.pointer, x.size, y.size);
				break;
			}
			case MathDomain::Int:
			{
				CUDA_CALL_SINGLE(__MakeTriple__, (float*)v.pointer, (int*)x.pointer, (int*)y.pointer, (int*)z.pointer, x.size, y.size);
				break;
			}
			default:
				return CudaKernelException::_NotImplementedException;
		}

		return cudaGetLastError();
	}
}

template <typename T>
GLOBAL void __MakePair__(float* RESTRICT z, const T* RESTRICT x, const T* RESTRICT y, const size_t sz)
{
	CUDA_FUNCTION_PROLOGUE

	CUDA_FOR_LOOP_PROLOGUE

		const size_t idx = i << 1;
		z[idx    ] = static_cast<float>(x[i]);
		z[idx + 1] = static_cast<float>(y[i]);

	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __MakeTriple__(float* RESTRICT v, const T* RESTRICT x, const T* RESTRICT y, const T* RESTRICT z, const size_t nRows, const size_t nCols)
{
	const int tidX = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int stepX = gridDim.x * blockDim.x;
	const int tidY = blockDim.y * blockIdx.y + threadIdx.y;
	const unsigned int stepY = gridDim.y * blockDim.y;

	for (size_t i = tidX; i < nRows; i += stepX)
	{
		for (size_t j = tidY; j < nCols; j += stepY)
		{
			const size_t offset = j + i * nCols;
			v[3 * offset    ] = x[i];
			v[3 * offset + 1] = y[j];
			v[3 * offset + 2] = z[i + j * nRows];
		}
	}
}