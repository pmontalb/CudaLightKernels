#include "Common.cuh"
#include "Flags.cuh"
#include "Types.h"
#include <curand.h>
#include <curand_kernel.h>

EXTERN_C
{
	EXPORT int _Zero(MemoryBuffer buf);
	EXPORT int _ZeroRaw(const ptr_t pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain);
	
	EXPORT int _Initialize(MemoryBuffer buf, const double value);
    EXPORT int _InitializeRaw(const ptr_t pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const double value);

	EXPORT int _LinSpace(MemoryBuffer buf, const double x0, const double x1);
	EXPORT int _LinSpaceRaw(const ptr_t pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const double x0, const double x1);

	EXPORT int _RandUniform(MemoryBuffer buf, const unsigned seed);
	EXPORT int _RandUniformRaw(const ptr_t pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const unsigned seed);

	EXPORT int _RandNormal(MemoryBuffer buf, const unsigned seed);
	EXPORT int _RandNormalRaw(const ptr_t pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const unsigned seed);

	EXPORT int _Eye(MemoryTile buf);
	EXPORT int _EyeRaw(const ptr_t pointer, const unsigned nRows, const MemorySpace memorySpace, const MathDomain mathDomain);

	EXPORT int _OnesUpperTriangular(MemoryTile buf);
	EXPORT int _OnesUpperTriangularRaw(const ptr_t pointer, const unsigned nRows, const MemorySpace memorySpace, const MathDomain mathDomain);

	EXPORT int _RandShuffle(MemoryBuffer buf, const unsigned seed);
	EXPORT int _RandShuffleRaw(const ptr_t pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const unsigned seed);

	EXPORT int _RandShufflePair(MemoryBuffer buf1, MemoryBuffer bu2, const unsigned seed);
	EXPORT int _RandShufflePairRaw(const ptr_t p1, const ptr_t p2, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const unsigned seed);
	
	EXPORT int _RandShuffleColumns(MemoryTile buf, const unsigned seed);
	EXPORT int _RandShuffleColumnsRaw(const ptr_t pointer, const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain, const unsigned seed);
	
	EXPORT int _RandShuffleColumnsPair(MemoryTile buf1, MemoryTile bu2, const unsigned seed);
	EXPORT int _RandShuffleColumnsPairRaw(const ptr_t p1, const ptr_t p2, const unsigned nRows1, const unsigned nRows2, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain, const unsigned seed);
}


#define CURAND_STATE_PTR curandState* const RESTRICT

template <typename T>
GLOBAL void __Initialize__(T* RESTRICT ptr, const ptr_t sz, const T value);

template <typename T>
GLOBAL void __LinSpace__(T* RESTRICT ptr, const ptr_t sz, const T x0, const T dx);

GLOBAL void __SetupCuRand__(CURAND_STATE_PTR state, const ptr_t sz, const unsigned seed);

template <typename T>
GLOBAL void __RandUniform__(T* RESTRICT ptr, CURAND_STATE_PTR state, const unsigned sz, const unsigned fullSz);

template <typename T>
GLOBAL void __RandNormal__(T* RESTRICT ptr, CURAND_STATE_PTR state, const unsigned sz, const unsigned fullSz);

template <typename T>
GLOBAL void __Eye__(T* RESTRICT ptr, const ptr_t sz);

template <typename T>
GLOBAL void __OnesUpperTriangular__(T* RESTRICT ptr, const ptr_t sz);

template <typename T>
GLOBAL void __RandShuffle__(T* RESTRICT ptr, CURAND_STATE_PTR state, const unsigned sz);

template <typename T>
GLOBAL void __RandShufflePair__(T* RESTRICT p1, T* RESTRICT p2, CURAND_STATE_PTR state, const unsigned sz);

template <typename T>
GLOBAL void __RandShuffleColumns__(T* RESTRICT ptr, CURAND_STATE_PTR state, const unsigned nRows, const unsigned nCols);

template <typename T>
GLOBAL void __RandShuffleColumnsPair__(T* RESTRICT ptr1, T* RESTRICT ptr2, CURAND_STATE_PTR state, const unsigned nRows1, const unsigned nRows2, const unsigned nCols);