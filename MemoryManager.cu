#include "MemoryManager.cuh"
#include <stdio.h>

EXTERN_C
{
	namespace clk
	{
		EXPORT int _HostToHostCopy(MemoryBuffer dest, const MemoryBuffer source)
		{
			return cudaMemcpy((void *)dest.pointer, (void *)source.pointer, dest.TotalSize(), cudaMemcpyHostToHost);
		}

		EXPORT int _HostToDeviceCopy(MemoryBuffer dest, const MemoryBuffer source)
		{
			return cudaMemcpy((void *)dest.pointer, (void *)source.pointer, dest.TotalSize(), cudaMemcpyHostToDevice);
		}

		EXPORT int _DeviceToHostCopy(MemoryBuffer dest, const MemoryBuffer source)
		{
			return cudaMemcpy((void *)dest.pointer, (void *)source.pointer, dest.TotalSize(), cudaMemcpyDeviceToHost);
		}

		EXPORT int _DeviceToDeviceCopy(MemoryBuffer dest, const MemoryBuffer source)
		{
			return cudaMemcpy((void *)dest.pointer, (void *)source.pointer, dest.TotalSize(), cudaMemcpyDeviceToDevice);
		}

		EXPORT int _AutoCopy(MemoryBuffer dest, const MemoryBuffer source)
		{
			return cudaMemcpy((void *)dest.pointer, (void *)source.pointer, dest.TotalSize(), cudaMemcpyDefault);
		}

		EXPORT int _Alloc(MemoryBuffer& buf)
		{
			return cudaMalloc((void **)&buf.pointer, buf.TotalSize());
		}

		EXPORT int _AllocHost(MemoryBuffer& buf)
		{
			return cudaMallocHost((void **)&buf.pointer, buf.TotalSize());
		}

		EXPORT int _Free(const MemoryBuffer buf)
		{
			cudaThreadSynchronize();
			return cudaFree((void *)buf.pointer);
		}

		EXPORT int _FreeHost(const MemoryBuffer buf)
		{
			return cudaFreeHost((void *)buf.pointer);
		}
	}
}