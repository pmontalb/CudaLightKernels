#pragma once

//#include "Common.cuh"
#include "Flags.cuh"
#include "Types.h"

EXTERN_C
{
	// z[2 * i] = x[i]; z[2 * i + 1] = y[i]
	EXPORT int _MakePair(MemoryBuffer z, const MemoryBuffer x, const MemoryBuffer y);
	EXPORT int _MakePairRaw(ptr_t z, const ptr_t x, const ptr_t y, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _MakePair(MemoryBuffer(z, size, memorySpace, mathDomain), MemoryBuffer(x, size, memorySpace, mathDomain), MemoryBuffer(y, size, memorySpace, mathDomain));
	}

	// v[3 * i] = x[i]; v[3 * i + 1] = y[i]; v[3 * i + 2] = z[i]
	EXPORT int _MakeTriple(MemoryBuffer v, const MemoryBuffer x, const MemoryBuffer y, const MemoryBuffer z);
	EXPORT int _MakeTripleRaw(ptr_t v, const ptr_t x, const ptr_t y, const ptr_t z, const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _MakeTriple(MemoryBuffer(z, 3 * nRows * nCols, memorySpace, mathDomain),
						 MemoryBuffer(x, nRows, memorySpace, mathDomain), 
						 MemoryBuffer(y, nCols, memorySpace, mathDomain),
						   MemoryBuffer(z, nRows * nCols, memorySpace, mathDomain));
	}

	EXPORT int _MakeRgbaJetColorMap(MemoryBuffer out, const MemoryBuffer in);
	EXPORT int _MakeRgbaJetColorMapRaw(ptr_t out, const ptr_t in, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _MakeRgbaJetColorMap(MemoryBuffer(out, 4 * size, memorySpace, mathDomain), MemoryBuffer(in, size, memorySpace, mathDomain));
	}
}

template <typename T>
GLOBAL void __MakePair__(float* RESTRICT z, const T* RESTRICT x, const T* RESTRICT y, const size_t sz);

template <typename T>
GLOBAL void __MakeTriple__(float* RESTRICT v, const T* RESTRICT x, const T* RESTRICT y, const T* RESTRICT z, const size_t nRows, const size_t nCols);