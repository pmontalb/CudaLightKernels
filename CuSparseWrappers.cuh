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

	/**
	* zDense = yDense - xSparse
	*/
	inline EXPORT int _SparseSubtract(MemoryBuffer z, const SparseMemoryBuffer x, const MemoryBuffer y) { return _SparseAdd(z, x, y, -1.0); }

	/**
	*	yDense = ASparse * xDense
	*/
	EXPORT int _SparseDot(MemoryBuffer y, const SparseMemoryTile A, const MemoryBuffer x, const MatrixOperation aOperation = MatrixOperation::None, const double alpha = 1.0);


	/**
	*	ADense = BSparse * CDense
	*/
	EXPORT int _SparseMultiply(MemoryTile A, const SparseMemoryTile B, const MemoryTile C, const unsigned leadingDimensionB, const unsigned leadingDimensionC, const MatrixOperation bOperation = MatrixOperation::None, const double alpha = 1.0);
}
