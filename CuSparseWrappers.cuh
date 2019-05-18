#pragma once

#include "Common.cuh"
#include "Flags.cuh"
#include <cusparse_v2.h>
#include "Types.h"

EXTERN_C
{
	/**
	* zDense = alpha * xSparse + yDense
	*/
	EXPORT int _SparseAdd(MemoryBuffer z, const SparseMemoryBuffer x, const MemoryBuffer y, const double alpha = 1.0);
    EXPORT int _SparseAddRaw(const ptr_t z, const ptr_t x, const ptr_t y, const unsigned nNonZeros, const ptr_t nonZeroIndices, const MemorySpace memorySpace, const MathDomain mathDomain, const unsigned size, const double alpha = 1.0);

	/**
	* zDense = yDense - xSparse
	*/
	EXPORT int _SparseSubtract(MemoryBuffer z, const SparseMemoryBuffer x, const MemoryBuffer y) { return _SparseAdd(z, x, y, -1.0); }
	inline EXPORT int _SparseSubtractRaw(const ptr_t z, const ptr_t x, const ptr_t y, const unsigned nNonZeros, const ptr_t nonZeroIndices, const MemorySpace memorySpace, const MathDomain mathDomain, const unsigned size)
	{
	    return _SparseAddRaw(z, x, y, nNonZeros, nonZeroIndices, memorySpace, mathDomain, size, -1.0);
	}

	/**
	*	yDense = ASparse * xDense
	*/
	EXPORT int _SparseDot(MemoryBuffer y, const SparseMemoryTile A, const MemoryBuffer x, const MatrixOperation aOperation = MatrixOperation::None, const double alpha = 1.0);
	EXPORT int _SparseDotRaw(const ptr_t y, const ptr_t A, const ptr_t x, 
						  const unsigned nNonZeros, const ptr_t nonZeroColumnIndices, const ptr_t nNonZeroRows,
						  const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain, 
						  const MatrixOperation aOperation = MatrixOperation::None, const double alpha = 1.0);


	/**
	*	ADense = BSparse * CDense
	*/
	EXPORT int _SparseMultiply(MemoryTile A, const SparseMemoryTile B, const MemoryTile C, const unsigned leadingDimensionB, const unsigned leadingDimensionC, const MatrixOperation bOperation = MatrixOperation::None, const double alpha = 1.0);
	EXPORT int _SparseMultiplyRaw(const ptr_t A, const ptr_t B, const ptr_t C, 
								  const unsigned nNonZeros, const ptr_t nonZeroColumnIndices, const ptr_t nNonZeroRows,
								  const unsigned nRowsB, const unsigned nRowsC, const unsigned nColsC, const MemorySpace memorySpace, const MathDomain mathDomain,
								  const unsigned leadingDimensionB, const unsigned leadingDimensionC, const MatrixOperation bOperation = MatrixOperation::None, const double alpha = 1.0);
}
