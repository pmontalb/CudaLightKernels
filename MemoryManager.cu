#include "MemoryManager.cuh"
#include <stdio.h>

EXTERN_C
{
	EXPORT int _HostToHostCopy(MemoryBuffer& dest, const MemoryBuffer& source)
	{
		return cudaMemcpy((void *)dest.pointer, (void *)source.pointer, dest.TotalSize(), cudaMemcpyHostToHost);
	}
	EXPORT int _HostToHostCopyRaw(const ptr_t destPointer, const ptr_t sourcePointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer dest(destPointer, size, memorySpace, mathDomain);
		MemoryBuffer source(sourcePointer, size, memorySpace, mathDomain);
		return _HostToHostCopy(dest, source);
    }

	EXPORT int _HostToDeviceCopy(MemoryBuffer& dest, const MemoryBuffer& source)
	{
		return cudaMemcpy((void *)dest.pointer, (void *)source.pointer, dest.TotalSize(), cudaMemcpyHostToDevice);
	}
	EXPORT int _HostToDeviceCopyRaw(const ptr_t destPointer, const ptr_t sourcePointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer dest(destPointer, size, memorySpace, mathDomain);
		MemoryBuffer source(sourcePointer, size, memorySpace, mathDomain);
		return _HostToDeviceCopy(dest, source);
	}

	EXPORT int _DeviceToHostCopy(MemoryBuffer& dest, const MemoryBuffer& source)
	{
		return cudaMemcpy((void *)dest.pointer, (void *)source.pointer, dest.TotalSize(), cudaMemcpyDeviceToHost);
	}
	EXPORT int _DeviceToHostCopyRaw(const ptr_t destPointer, const ptr_t sourcePointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer dest(destPointer, size, memorySpace, mathDomain);
		MemoryBuffer source(sourcePointer, size, memorySpace, mathDomain);
		return _DeviceToHostCopy(dest, source);
	}

	EXPORT int _DeviceToDeviceCopy(MemoryBuffer& dest, const MemoryBuffer& source)
	{
		return cudaMemcpy((void *)dest.pointer, (void *)source.pointer, dest.TotalSize(), cudaMemcpyDeviceToDevice);
	}
	EXPORT int _DeviceToDeviceCopyRaw(const ptr_t destPointer, const ptr_t sourcePointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer dest(destPointer, size, memorySpace, mathDomain);
		MemoryBuffer source(sourcePointer, size, memorySpace, mathDomain);
		return _DeviceToDeviceCopy(dest, source);
	}

	EXPORT int _AutoCopy(MemoryBuffer& dest, const MemoryBuffer& source)
	{
		return cudaMemcpy((void *)dest.pointer, (void *)source.pointer, dest.TotalSize(), cudaMemcpyDefault);
	}
	EXPORT int _AutoCopyRaw(const ptr_t destPointer, const ptr_t sourcePointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer dest(destPointer, size, memorySpace, mathDomain);
		MemoryBuffer source(sourcePointer, size, memorySpace, mathDomain);
		return _AutoCopy(dest, source);
	}

	EXPORT int _Alloc(MemoryBuffer& buf)
	{
		return cudaMalloc((void **)&buf.pointer, buf.TotalSize());
	}
	EXPORT int _AllocRaw(ptr_t& pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer tmp(0, size, memorySpace, mathDomain);
		int ret = _Alloc(tmp);

		pointer = tmp.pointer;

		return ret;
    }

	EXPORT int _AllocHost(MemoryBuffer& buf)
	{
		return cudaMallocHost((void **)&buf.pointer, buf.TotalSize());
	}
	EXPORT int _AllocHostRaw(ptr_t& pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer tmp(0, size, memorySpace, mathDomain);
		int ret = _AllocHost(tmp);

		pointer = tmp.pointer;

		return ret;
	}

	EXPORT int _Free(const MemoryBuffer& buf)
	{
		cudaDeviceSynchronize();
		return cudaFree((void *)buf.pointer);
	}
	EXPORT int _FreeRaw(const ptr_t pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer buf(pointer, size, memorySpace, mathDomain);
		return _Free(buf);
	}

	EXPORT int _FreeHost(const MemoryBuffer& buf)
	{
		return cudaFreeHost((void *)buf.pointer);
	}
	EXPORT int _FreeHostRaw(const ptr_t pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer buf(pointer, size, memorySpace, mathDomain);
		return _FreeHost(buf);
	}
}