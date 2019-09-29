#include "CubWrappers.cuh"
#include "MemoryManager.cuh"

struct Abs
{
	template <typename T>
	HOST DEVICE __forceinline__ T operator()(const T &a) const
	{
		return a > 0 ? a : -a;
	}
};

struct AbsMin
{
	template <typename T>
	HOST DEVICE __forceinline__ T operator()(const T &a, const T &b) const
	{
		return CUB_MIN(Abs()(a), Abs()(b));
	}
};

struct AbsMax
{
	template <typename T>
	HOST DEVICE __forceinline__ T operator()(const T &a, const T &b) const
	{
		return CUB_MAX(Abs()(a), Abs()(b));
	}
};

EXTERN_C
{
	EXPORT int _Sum(double& sum, const MemoryBuffer& v)
	{
		MemoryBuffer output(0, 1, v.memorySpace, v.mathDomain);
		// Determine temporary device storage requirements
		_Alloc(output);

		MemoryBuffer cacheBuffer;
		_DetermineSumCache(cacheBuffer, v, output);
		
		auto ret = _SumWithProvidedCache(sum, v, cacheBuffer, output);
		_Free(cacheBuffer);
		_Free(output);

		return cudaGetLastError();
	}

	EXPORT int _DetermineSumCache(MemoryBuffer& cacheBuffer, const MemoryBuffer& v, const MemoryBuffer& oneElementCache)
	{
		void* cache = nullptr;
		size_t cacheSize = 0;
		
		switch (v.mathDomain)
		{
			case MathDomain::Float:
			{
				cub::DeviceReduce::Sum(cache, cacheSize, (float*)v.pointer, (float*)oneElementCache.pointer, v.size);
				
				// Allocate temporary storage
				cudaMalloc(&cache, cacheSize);
				break;
			}
			case MathDomain::Double:
			{
				cub::DeviceReduce::Sum(cache, cacheSize, (double*)v.pointer, (double*)oneElementCache.pointer, v.size);
				
				// Allocate temporary storage
				cudaMalloc(&cache, cacheSize);
				break;
			}
			case MathDomain::Int:
			{
				cub::DeviceReduce::Sum(cache, cacheSize, (int*)v.pointer, (int*)oneElementCache.pointer, v.size);
				
				// Allocate temporary storage
				cudaMalloc(&cache, cacheSize);
				break;
			}
			default:
				return CudaKernelException::_NotImplementedException;
		}
		
		cacheBuffer.pointer = reinterpret_cast<ptr_t>(cache);
		cacheBuffer.size = static_cast<unsigned>(cacheSize);
		cacheBuffer.memorySpace = MemorySpace::Device;
		cacheBuffer.mathDomain = v.mathDomain;
		
		return cudaGetLastError();
	}

	EXPORT int _SumWithProvidedCache(double& sum, const MemoryBuffer& v, MemoryBuffer& cache, MemoryBuffer& outputCache)
	{
		switch (v.mathDomain)
		{
			case MathDomain::Float:
			{
				// Run sum-reduction
				size_t totalSize = cache.size;
				cub::DeviceReduce::Sum((void*)cache.pointer, totalSize, (float*)v.pointer, (float*)outputCache.pointer, v.size);
				
				float _sum;
				cudaMemcpy(&_sum, (float*)outputCache.pointer, sizeof(float), cudaMemcpyDeviceToHost);
				sum = _sum;
				
				break;
			}
			case MathDomain::Double:
			{
				// Run sum-reduction
				size_t totalSize = cache.size;
				cub::DeviceReduce::Sum((void*)cache.pointer, totalSize, (double*)v.pointer, (double*)outputCache.pointer, v.size);
				
				cudaMemcpy(&sum, (double*)outputCache.pointer, sizeof(double), cudaMemcpyDeviceToHost);
				break;
			}
			case MathDomain::Int:
			{
				// Run sum-reduction
				size_t totalSize = cache.size;
				cub::DeviceReduce::Sum((void*)cache.pointer, totalSize, (int*)v.pointer, (int*)outputCache.pointer, v.size);
				
				int _sum;
				cudaMemcpy(&_sum, (int*)outputCache.pointer, sizeof(int), cudaMemcpyDeviceToHost);
				sum = _sum;
				
				break;
			}
			default:
				return CudaKernelException::_NotImplementedException;
		}
		
		return cudaGetLastError();
	}

	EXPORT int _Min(double& min, const MemoryBuffer& v)
	{
		void* cache = nullptr;

		MemoryBuffer output(0, 1, v.memorySpace, v.mathDomain);
		// Determine temporary device storage requirements
		_Alloc(output);

		size_t temp_storage_bytes = 0;

		switch (v.mathDomain)
		{
			case MathDomain::Float:
			{
				cub::DeviceReduce::Min(cache, temp_storage_bytes, (float*)v.pointer, (float*)output.pointer, v.size);

				// Allocate temporary storage
				cudaMalloc(&cache, temp_storage_bytes);

				// Run sum-reduction
				cub::DeviceReduce::Min(cache, temp_storage_bytes, (float*)v.pointer, (float*)output.pointer, v.size);

				float _min;
				cudaMemcpy(&_min, (float*)output.pointer, sizeof(float), cudaMemcpyDeviceToHost);
				min = _min;

				break;
			}
			case MathDomain::Double:
			{
				cub::DeviceReduce::Min(cache, temp_storage_bytes, (double*)v.pointer, (double*)output.pointer, v.size);

				// Allocate temporary storage
				cudaMalloc(&cache, temp_storage_bytes);

				// Run sum-reduction
				cub::DeviceReduce::Min(cache, temp_storage_bytes, (double*)v.pointer, (double*)output.pointer, v.size);

				cudaMemcpy(&min, (double*)output.pointer, sizeof(double), cudaMemcpyDeviceToHost);
				break;
			}
			case MathDomain::Int:
			{
				cub::DeviceReduce::Min(cache, temp_storage_bytes, (int*)v.pointer, (int*)output.pointer, v.size);

				// Allocate temporary storage
				cudaMalloc(&cache, temp_storage_bytes);

				// Run sum-reduction
				cub::DeviceReduce::Min(cache, temp_storage_bytes, (int*)v.pointer, (int*)output.pointer, v.size);

				int _min;
				cudaMemcpy(&_min, (int*)output.pointer, sizeof(int), cudaMemcpyDeviceToHost);
				min = _min;

				break;
			}
			default:
				return CudaKernelException::_NotImplementedException;
		}

		cudaFree(cache);
		_Free(output);

		return cudaGetLastError();
	}

	EXPORT int _Max(double& max, const MemoryBuffer& v)
	{
		void* cache = nullptr;

		MemoryBuffer output(0, 1, v.memorySpace, v.mathDomain);
		// Determine temporary device storage requirements
		_Alloc(output);

		size_t temp_storage_bytes = 0;

		switch (v.mathDomain)
		{
			case MathDomain::Float:
			{
				cub::DeviceReduce::Max(cache, temp_storage_bytes, (float*)v.pointer, (float*)output.pointer, v.size);

				// Allocate temporary storage
				cudaMalloc(&cache, temp_storage_bytes);

				// Run sum-reduction
				cub::DeviceReduce::Max(cache, temp_storage_bytes, (float*)v.pointer, (float*)output.pointer, v.size);

				float _max;
				cudaMemcpy(&_max, (float*)output.pointer, sizeof(float), cudaMemcpyDeviceToHost);
				max = _max;

				break;
			}
			case MathDomain::Double:
			{
				cub::DeviceReduce::Max(cache, temp_storage_bytes, (double*)v.pointer, (double*)output.pointer, v.size);

				// Allocate temporary storage
				cudaMalloc(&cache, temp_storage_bytes);

				// Run sum-reduction
				cub::DeviceReduce::Min(cache, temp_storage_bytes, (double*)v.pointer, (double*)output.pointer, v.size);

				cudaMemcpy(&max, (double*)output.pointer, sizeof(double), cudaMemcpyDeviceToHost);
				break;
			}
			case MathDomain::Int:
			{
				cub::DeviceReduce::Max(cache, temp_storage_bytes, (int*)v.pointer, (int*)output.pointer, v.size);

				// Allocate temporary storage
				cudaMalloc(&cache, temp_storage_bytes);

				// Run sum-reduction
				cub::DeviceReduce::Max(cache, temp_storage_bytes, (int*)v.pointer, (int*)output.pointer, v.size);

				int _max;
				cudaMemcpy(&_max, (int*)output.pointer, sizeof(int), cudaMemcpyDeviceToHost);
				max = _max;

				break;
			}
			default:
				return CudaKernelException::_NotImplementedException;
		}

		cudaFree(cache);
		_Free(output);

		return cudaGetLastError();
	}

	EXPORT int _AbsMin(double& min, const MemoryBuffer& v)
	{
		void* cache = nullptr;

		MemoryBuffer output(0, 1, v.memorySpace, v.mathDomain);
		// Determine temporary device storage requirements
		_Alloc(output);

		size_t temp_storage_bytes = 0;

		switch (v.mathDomain)
		{
			case MathDomain::Float:
			{
				cub::DeviceReduce::Reduce(cache, temp_storage_bytes, (float*)v.pointer, (float*)output.pointer, v.size, ::AbsMin(), 1e9);

				// Allocate temporary storage
				cudaMalloc(&cache, temp_storage_bytes);

				// Run sum-reduction
				cub::DeviceReduce::Reduce(cache, temp_storage_bytes, (float*)v.pointer, (float*)output.pointer, v.size, ::AbsMin(), 1e9);

				float _min;
				cudaMemcpy(&_min, (float*)output.pointer, sizeof(float), cudaMemcpyDeviceToHost);
				min = _min;

				break;
			}
			case MathDomain::Double:
			{
				cub::DeviceReduce::Reduce(cache, temp_storage_bytes, (double*)v.pointer, (double*)output.pointer, v.size, ::AbsMin(), 1e9);

				// Allocate temporary storage
				cudaMalloc(&cache, temp_storage_bytes);

				// Run sum-reduction
				cub::DeviceReduce::Reduce(cache, temp_storage_bytes, (double*)v.pointer, (double*)output.pointer, v.size, ::AbsMin(), 1e9);

				cudaMemcpy(&min, (double*)output.pointer, sizeof(double), cudaMemcpyDeviceToHost);
				break;
			}
			case MathDomain::Int:
			{
				cub::DeviceReduce::Reduce(cache, temp_storage_bytes, (int*)v.pointer, (int*)output.pointer, v.size, ::AbsMin(), 1e9);

				// Allocate temporary storage
				cudaMalloc(&cache, temp_storage_bytes);

				// Run sum-reduction
				cub::DeviceReduce::Reduce(cache, temp_storage_bytes, (int*)v.pointer, (int*)output.pointer, v.size, ::AbsMin(), 1e9);

				int _min;
				cudaMemcpy(&_min, (int*)output.pointer, sizeof(int), cudaMemcpyDeviceToHost);
				min = _min;

				break;
			}
			default:
				return CudaKernelException::_NotImplementedException;
		}

		cudaFree(cache);
		_Free(output);

		return cudaGetLastError();
	}

	EXPORT int _AbsMax(double& max, const MemoryBuffer& v)
	{
		void* cache = nullptr;

		MemoryBuffer output(0, 1, v.memorySpace, v.mathDomain);
		// Determine temporary device storage requirements
		_Alloc(output);

		size_t temp_storage_bytes = 0;

		switch (v.mathDomain)
		{
			case MathDomain::Float:
			{
				cub::DeviceReduce::Reduce(cache, temp_storage_bytes, (float*)v.pointer, (float*)output.pointer, v.size, ::AbsMax(), 0);

				// Allocate temporary storage
				cudaMalloc(&cache, temp_storage_bytes);

				// Run sum-reduction
				cub::DeviceReduce::Reduce(cache, temp_storage_bytes, (float*)v.pointer, (float*)output.pointer, v.size, ::AbsMax(), 0);

				float _max;
				cudaMemcpy(&_max, (float*)output.pointer, sizeof(float), cudaMemcpyDeviceToHost);
				max = _max;

				break;
			}
			case MathDomain::Double:
			{
				cub::DeviceReduce::Reduce(cache, temp_storage_bytes, (double*)v.pointer, (double*)output.pointer, v.size, ::AbsMax(), 0);

				// Allocate temporary storage
				cudaMalloc(&cache, temp_storage_bytes);

				// Run sum-reduction
				cub::DeviceReduce::Reduce(cache, temp_storage_bytes, (double*)v.pointer, (double*)output.pointer, v.size, ::AbsMax(), 0);

				cudaMemcpy(&max, (double*)output.pointer, sizeof(double), cudaMemcpyDeviceToHost);
				break;
			}
			case MathDomain::Int:
			{
				cub::DeviceReduce::Reduce(cache, temp_storage_bytes, (int*)v.pointer, (int*)output.pointer, v.size, ::AbsMax(), 0);

				// Allocate temporary storage
				cudaMalloc(&cache, temp_storage_bytes);

				// Run sum-reduction
				cub::DeviceReduce::Reduce(cache, temp_storage_bytes, (int*)v.pointer, (int*)output.pointer, v.size, ::AbsMax(), 0);

				int _max;
				cudaMemcpy(&_max, (int*)output.pointer, sizeof(int), cudaMemcpyDeviceToHost);
				max = _max;

				break;
			}
			default:
				return CudaKernelException::_NotImplementedException;
		}

		cudaFree(cache);
		_Free(output);

		return cudaGetLastError();
	}
}