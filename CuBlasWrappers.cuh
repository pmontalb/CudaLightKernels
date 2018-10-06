#pragma once

#include "Common.cuh"
#include "Flags.cuh"
#include "Types.h"

#define USE_NAIVE_ELEMENTWISE_PRODUCT

EXTERN_C
{
	/**
	* z = alpha * x + y
	*/
	EXPORT int _Add(MemoryBuffer z, const MemoryBuffer x, const MemoryBuffer y, const double alpha = 1.0);
    EXPORT int _AddRaw(const ptr_t z, const ptr_t x, const ptr_t y, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const double alpha = 1.0);

	/**
	* z = x - y
	*/
	inline EXPORT int _Subtract(MemoryBuffer z, const MemoryBuffer x, const MemoryBuffer y) { return _Add(z, y, x, -1.0); }
	inline EXPORT int _SubtractRaw(const ptr_t z, const ptr_t x, const ptr_t y, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain) { return _AddRaw(z, y, x, size, memorySpace, mathDomain, -1.0); }

	/**
	* z += alpha * x
	*/
	EXPORT int _AddEqual(MemoryBuffer z, const MemoryBuffer x, const double alpha = 1.0);
	EXPORT int _AddEqualRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const double alpha = 1.0);

	/**
	* A += alpha * B (NB: it uses <t>geam, maybe more efficient than <t>axpy?)
	*/
	EXPORT int _AddEqualMatrix(MemoryTile A, const MemoryTile B, const MatrixOperation aOperation = MatrixOperation::None, const MatrixOperation bOperation = MatrixOperation::None, const double alpha = 1.0);
	EXPORT int _AddEqualMatrixRaw(const ptr_t A, const ptr_t B, const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain, const MatrixOperation aOperation = MatrixOperation::None, const MatrixOperation bOperation = MatrixOperation::None, const double alpha = 1.0);

	/**
	* z -= x
	*/
	inline EXPORT int _SubtractEqual(MemoryBuffer z, const MemoryBuffer x) { return _AddEqual(z, x, -1.0); };
	inline EXPORT int _SubtractEqualRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain) { return _AddEqualRaw(z, x, size, memorySpace, mathDomain, -1.0); };

	/**
	* z *= alpha
	*/
	EXPORT int _Scale(MemoryBuffer z, const double alpha);
	EXPORT int _ScaleRaw(const ptr_t z, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const double alpha);

	/**
	* z = alpha * x * y: NB: there's no such a function in cuBLAS -> I use SBMV with a diagonal matrix == vector
	*/
	EXPORT int _ElementwiseProduct(MemoryBuffer z, const MemoryBuffer x, const MemoryBuffer y, const double alpha = 1.0);
	EXPORT int _ElementwiseProductRaw(const ptr_t z, const ptr_t x, const ptr_t y, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const double alpha = 1.0);

	/*
	*	A = alpha * B * C
	*/
	EXPORT int _Multiply(MemoryTile A, const MemoryTile B, const MemoryTile C, const unsigned leadingDimensionB, const unsigned leadingDimensionC, const MatrixOperation bOperation = MatrixOperation::None, const MatrixOperation cOperation = MatrixOperation::None, const double alpha = 1.0);
	EXPORT int _MultiplyRaw(const ptr_t A, const ptr_t B, const ptr_t C, const unsigned nRowsB, const unsigned nRowsC, const unsigned nColsC, const MemorySpace memorySpace, const MathDomain mathDomain, const unsigned leadingDimensionB, const unsigned leadingDimensionC, const MatrixOperation bOperation = MatrixOperation::None, const MatrixOperation cOperation = MatrixOperation::None, const double alpha = 1.0);

	/**
	*	y = alpha * A * x
	*/
	EXPORT int _Dot(MemoryBuffer y, const MemoryTile A, const MemoryBuffer x, const MatrixOperation aOperation = MatrixOperation::None, const double alpha = 1.0);
	EXPORT int _DotRaw(const ptr_t y, const ptr_t A, const ptr_t x, const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain, const MatrixOperation aOperation = MatrixOperation::None, const double alpha = 1.0);

	/**
	*	A += alpha * x * y^T
	*/
	EXPORT int _KroneckerProduct(MemoryTile A, const MemoryBuffer x, const MemoryBuffer y, const double alpha = 1.0);
	EXPORT int _KroneckerProductRaw(const ptr_t A, const ptr_t x, const ptr_t y, const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain, const double alpha = 1.0);


	/**
	* A = cumsum(A)
	*/
	EXPORT int _CumulativeRowSum(MemoryTile A);
	EXPORT int _CumulativeRowSumRaw(const ptr_t A, const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain);

	/**
	* X such that A * X = B by means of LU factorization
	*/
	EXPORT int _Solve(const MemoryTile A, MemoryTile B, const MatrixOperation aOperation = MatrixOperation::None);
	EXPORT int _SolveRaw(const ptr_t A, const ptr_t B, const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain, const MatrixOperation aOperation = MatrixOperation::None);

	/**
	* A = A^(-1) by means of LU factorization
	*/
	EXPORT int _Invert(MemoryTile A, const MatrixOperation aOperation = MatrixOperation::None);
	EXPORT int _InvertRaw(const ptr_t A, const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain, const MatrixOperation aOperation = MatrixOperation::None);
}

template <typename T>
GLOBAL void __ElementwiseProductNaive__(T* RESTRICT z, const T* RESTRICT x, const T* RESTRICT y, const size_t sz, const T alpha = static_cast<T>(1.0));