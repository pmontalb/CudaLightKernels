#include "CubWrappers.cuh"
#include "MemoryManager.cuh"

EXTERN_C
{
	EXPORT int _Sum(double& sum, const MemoryBuffer v)
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
				cub::DeviceReduce::Sum(cache, temp_storage_bytes, (float*)v.pointer, (float*)output.pointer, v.size);

				// Allocate temporary storage
				cudaMalloc(&cache, temp_storage_bytes);

				// Run sum-reduction
				cub::DeviceReduce::Sum(cache, temp_storage_bytes, (float*)v.pointer, (float*)output.pointer, v.size);

				float _sum;
				cudaMemcpy(&_sum, (float*)output.pointer, sizeof(float), cudaMemcpyDeviceToHost);
				sum = _sum;

				break;
			}
			case MathDomain::Double:
			{
				cub::DeviceReduce::Sum(cache, temp_storage_bytes, (double*)v.pointer, (double*)output.pointer, v.size);

				// Allocate temporary storage
				cudaMalloc(&cache, temp_storage_bytes);

				// Run sum-reduction
				cub::DeviceReduce::Sum(cache, temp_storage_bytes, (double*)v.pointer, (double*)output.pointer, v.size);

				cudaMemcpy(&sum, (double*)output.pointer, sizeof(double), cudaMemcpyDeviceToHost);
				break;
			}
			case MathDomain::Int:
			{
				cub::DeviceReduce::Sum(cache, temp_storage_bytes, (int*)v.pointer, (int*)output.pointer, v.size);

				// Allocate temporary storage
				cudaMalloc(&cache, temp_storage_bytes);

				// Run sum-reduction
				cub::DeviceReduce::Sum(cache, temp_storage_bytes, (int*)v.pointer, (int*)output.pointer, v.size);

				int _sum;
				cudaMemcpy(&_sum, (int*)output.pointer, sizeof(int), cudaMemcpyDeviceToHost);
				sum = _sum;

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