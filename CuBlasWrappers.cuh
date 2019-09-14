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
	EXPORT int _Subtract(MemoryBuffer z, const MemoryBuffer x, const MemoryBuffer y);
	inline EXPORT int _SubtractRaw(const ptr_t z, const ptr_t x, const ptr_t y, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain) { return _AddRaw(z, y, x, size, memorySpace, mathDomain, -1.0); }

	/**
	* z += alpha * x
	*/
	EXPORT int _AddEqual(MemoryBuffer z, const MemoryBuffer x, const double alpha = 1.0);
	EXPORT int _AddEqualRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const double alpha = 1.0);

	/**
	* A = alpha * B + beta * A (NB: it uses <t>geam, maybe more efficient than <t>axpy?)
	*/
	EXPORT int _AddEqualMatrix(MemoryTile A, const MemoryTile B, const MatrixOperation aOperation = MatrixOperation::None, const MatrixOperation bOperation = MatrixOperation::None, const double alpha = 1.0, const double beta = 1.0);
	EXPORT int _AddEqualMatrixRaw(const ptr_t A, const ptr_t B, const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain, const MatrixOperation aOperation = MatrixOperation::None, const MatrixOperation bOperation = MatrixOperation::None, const double alpha = 1.0, const double beta = 1.0);

	/**
	* z -= x
	*/
	EXPORT int _SubtractEqual(MemoryBuffer z, const MemoryBuffer x);
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
	*	A = alpha * B * C + beta * A
	*/
	EXPORT int _Multiply(MemoryTile A, const MemoryTile B, const MemoryTile C, const unsigned leadingDimensionB, const unsigned leadingDimensionC, const MatrixOperation bOperation = MatrixOperation::None, const MatrixOperation cOperation = MatrixOperation::None, const double alpha = 1.0, const double beta = 0.0);
	EXPORT int _MultiplyRaw(const ptr_t A, const ptr_t B, const ptr_t C, const unsigned nRowsB, const unsigned nRowsC, const unsigned nColsC, const MemorySpace memorySpace, const MathDomain mathDomain, const unsigned leadingDimensionB, const unsigned leadingDimensionC, const MatrixOperation bOperation = MatrixOperation::None, const MatrixOperation cOperation = MatrixOperation::None, const double alpha = 1.0, const double beta = 0.0);

	/**
	*	y = alpha * A * x + beta * y
	*/
	EXPORT int _Dot(MemoryBuffer y, const MemoryTile A, const MemoryBuffer x, const MatrixOperation aOperation = MatrixOperation::None, const double alpha = 1.0, const double beta = 0.0);
	EXPORT int _DotRaw(const ptr_t y, const ptr_t A, const ptr_t x, const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain, const MatrixOperation aOperation = MatrixOperation::None, const double alpha = 1.0, const double beta = 0.0);

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
	* x = sum(A[:, ])
	*/
	EXPORT int _RowWiseSum(MemoryBuffer x, const MemoryTile A, MemoryBuffer cache = MemoryBuffer());
	EXPORT int _RowWiseSumRaw(const ptr_t x, const ptr_t A, const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain, const ptr_t cache = 0);

	/**
* x = sum(A[:, ])
*/
	EXPORT int _CubeWiseSum(MemoryTile A, const MemoryCube T, MemoryCube cacheReshape = MemoryCube(), MemoryBuffer cacheOnes = MemoryBuffer());
	EXPORT int _CubeWiseSumRaw(const ptr_t A, const ptr_t T, const unsigned nRows, const unsigned nCols, const unsigned nCubes, const MemorySpace memorySpace, const MathDomain mathDomain, const ptr_t cacheReshape = 0, const ptr_t cacheOnes = 0);

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

	EXPORT int _ArgAbsMin(int& argMin, const MemoryBuffer x);
	inline EXPORT int _ArgAbsMinRaw(int& argMin, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _ArgAbsMin(argMin, MemoryBuffer(x, size, memorySpace, mathDomain));
	}

	// NB: it returns 1-based indices
	EXPORT int _ColumnWiseArgAbsMin(MemoryBuffer argMin, const MemoryTile A);
	inline EXPORT int _ColumnWiseArgAbsMinRaw(const ptr_t argMin, const ptr_t A, const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _ColumnWiseArgAbsMin(MemoryBuffer(argMin, nCols, memorySpace, mathDomain), MemoryTile(A, nRows, nCols, memorySpace, mathDomain));
	}

	EXPORT int _ArgAbsMax(int& argMax, const MemoryBuffer x);
	inline EXPORT int _ArgAbsMaxRaw(int& argMax, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _ArgAbsMax(argMax, MemoryBuffer(x, size, memorySpace, mathDomain));
	}

	// NB: it returns 1-based indices
	EXPORT int _ColumnWiseArgAbsMax(MemoryBuffer argMax, const MemoryTile A);
	inline EXPORT int _ColumnWiseArgAbsMaxRaw(const ptr_t argMax, const ptr_t A, const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _ColumnWiseArgAbsMax(MemoryBuffer(argMax, nCols, memorySpace, mathDomain), MemoryTile(A, nRows, nCols, memorySpace, mathDomain));
	}

	// z = { 1 if x == 0; 0 otherwise }
	EXPORT int _IsNonZero(MemoryBuffer z, const MemoryBuffer x);
	inline EXPORT int _IsNonZeroRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _IsNonZero(MemoryBuffer(z, size, memorySpace, mathDomain), MemoryBuffer(x, size, memorySpace, mathDomain));
	}

	// norm = ||x||_2
	EXPORT int _EuclideanNorm(double& norm, const MemoryBuffer x);
	inline EXPORT int _EuclideanNormRaw(double& norm, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _EuclideanNorm(norm, MemoryBuffer(x, size, memorySpace, mathDomain));
	}
}

// z = alpha * x + beta * y + gamma * z
GLOBAL void __IntAffineOperationNaive__(int* RESTRICT z, const int* RESTRICT x, const int* RESTRICT y, const size_t sz, const int alpha, const int beta, const int gamma);

template <typename T>
GLOBAL void __ElementwiseProductNaive__(T* RESTRICT z, const T* RESTRICT x, const T* RESTRICT y, const size_t sz, const T alpha = static_cast<T>(1.0));

template <typename T>
GLOBAL void __IsNonZero__(T* RESTRICT z, const T* RESTRICT x, const size_t sz);

template <typename T>
GLOBAL void __Reshape__(T* RESTRICT out, const T* RESTRICT in, const size_t nRows, const size_t nCols, const size_t nCubes);