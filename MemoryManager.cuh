#pragma once

#include "Common.cuh"
#include "Flags.cuh"

#include "Types.h"

EXTERN_C
{
	namespace clk
	{
		EXPORT int _HostToHostCopy(MemoryBuffer dest, const MemoryBuffer source);

		EXPORT int _HostToDeviceCopy(MemoryBuffer dest, const MemoryBuffer source);

		EXPORT int _DeviceToHostCopy(MemoryBuffer dest, const MemoryBuffer source);

		EXPORT int _DeviceToDeviceCopy(MemoryBuffer dest, const MemoryBuffer source);

		EXPORT int _AutoCopy(MemoryBuffer dest, const MemoryBuffer source);

		EXPORT int _Alloc(MemoryBuffer& ptr);

		EXPORT int _AllocHost(MemoryBuffer& ptr);

		EXPORT int _Free(const MemoryBuffer ptr);

		EXPORT int _FreeHost(const MemoryBuffer ptr);
	}
}