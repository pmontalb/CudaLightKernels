#pragma once

#include "Common.cuh"
#include "Flags.cuh"

#include "Types.h"

EXTERN_C
{
	EXPORT int _HostToHostCopy(MemoryBuffer dest, const MemoryBuffer source);
    EXPORT int _HostToHostCopyRaw(const ptr_t destPointer, const ptr_t sourcePointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain);

	EXPORT int _HostToDeviceCopy(MemoryBuffer dest, const MemoryBuffer source);
	EXPORT int _HostToDeviceCopyRaw(const ptr_t destPointer, const ptr_t sourcePointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain);

	EXPORT int _DeviceToHostCopy(MemoryBuffer dest, const MemoryBuffer source);
	EXPORT int _DeviceToHostCopyRaw(const ptr_t destPointer, const ptr_t sourcePointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain);

	EXPORT int _DeviceToDeviceCopy(MemoryBuffer dest, const MemoryBuffer source);
	EXPORT int _DeviceToDeviceCopyRaw(const ptr_t destPointer, const ptr_t sourcePointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain);

	EXPORT int _AutoCopy(MemoryBuffer dest, const MemoryBuffer source);
	EXPORT int _AutoCopyRaw(const ptr_t destPointer, const ptr_t sourcePointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain);

	EXPORT int _Alloc(MemoryBuffer& ptr);
	EXPORT int _AllocRaw(ptr_t& pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain);

	EXPORT int _AllocHost(MemoryBuffer& ptr);
	EXPORT int _AllocHostRaw(ptr_t& pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain);

	EXPORT int _Free(const MemoryBuffer ptr);
	EXPORT int _FreeRaw(const ptr_t pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain);

	EXPORT int _FreeHost(const MemoryBuffer ptr);
	EXPORT int _FreeHostRaw(const ptr_t pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain);
}
