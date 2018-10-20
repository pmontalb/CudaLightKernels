#pragma once

#include "Common.cuh"
#include "Flags.cuh"
#include "Types.h"

#include <cub/cub.cuh>

EXTERN_C
{
	EXPORT int _Sum(double& sum, const MemoryBuffer v);
	EXPORT int _SumRaw(double& sum, const ptr_t v, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _Sum(sum, MemoryBuffer(v, size, memorySpace, mathDomain));
	}
}