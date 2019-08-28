#pragma once

#include "Common.cuh"
#include "Flags.cuh"
#include "Types.h"

#include <cub/cub.cuh>

EXTERN_C
{
	EXPORT int _Sum(double& sum, const MemoryBuffer v);
	inline EXPORT int _SumRaw(double& sum, const ptr_t v, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _Sum(sum, MemoryBuffer(v, size, memorySpace, mathDomain));
	}

	EXPORT int _Min(double& min, const MemoryBuffer x);
	inline EXPORT int _MinRaw(double& min, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _Min(min, MemoryBuffer(x, size, memorySpace, mathDomain));
	}

	EXPORT int _Max(double& max, const MemoryBuffer x);
	inline EXPORT int _MaxRaw(double& max, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _Max(max, MemoryBuffer(x, size, memorySpace, mathDomain));
	}

	EXPORT int _AbsMin(double& min, const MemoryBuffer x);
	inline EXPORT int _AbsMinRaw(double& min, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _AbsMin(min, MemoryBuffer(x, size, memorySpace, mathDomain));
	}

	EXPORT int _AbsMax(double& max, const MemoryBuffer x);
	inline EXPORT int _AbsMaxRaw(double& max, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _AbsMax(max, MemoryBuffer(x, size, memorySpace, mathDomain));
	}
}