#pragma once

#include "Common.cuh"
#include "Flags.cuh"
#include "Types.h"


EXTERN_C
{
	namespace clk
	{
		/**
		* z = alpha * x + y
		*/
		EXPORT int _Add(MemoryBuffer z, const MemoryBuffer x, const MemoryBuffer y, const double alpha = 1.0);

		/**
		* z = x - y
		*/
		inline EXPORT int _Subtract(MemoryBuffer z, const MemoryBuffer x, const MemoryBuffer y) { return _Add(z, y, x, -1.0); }

		/**
		* z += alpha * x
		*/
		EXPORT int _AddEqual(MemoryBuffer z, const MemoryBuffer x, const double alpha = 1.0);

		/**
		* A += alpha * B (NB: it uses <t>geam, maybe more efficient than <t>axpy?)
		*/
		EXPORT int _AddEqualMatrix(MemoryTile A, const MemoryTile B, const MatrixOperation aOperation = MatrixOperation::None, const MatrixOperation bOperation = MatrixOperation::None, const double alpha = 1.0);

		/**
		* z -= x
		*/
		inline EXPORT int _SubtractEqual(MemoryBuffer z, const MemoryBuffer x) { return _AddEqual(z, x, -1.0); };

		/**
		* z *= alpha
		*/
		EXPORT int _Scale(MemoryBuffer z, const double alpha);

		/**
		* z = alpha * x * y: NB: there's no such a function in cuBLAS -> I use SBMV with a diagonal matrix == vector
		*/
		EXPORT int _ElementwiseProduct(MemoryBuffer z, const MemoryBuffer x, const MemoryBuffer y, const double alpha = 1.0);

		/*
		*	A = alpha * B * C
		*/
		EXPORT int _Multiply(MemoryTile A, const MemoryTile B, const MemoryTile C, const unsigned leadingDimensionB, const unsigned leadingDimensionC, const MatrixOperation bOperation = MatrixOperation::None, const MatrixOperation cOperation = MatrixOperation::None, const double alpha = 1.0);

		/**
		*	y = alpha * A * x
		*/
		EXPORT int _Dot(MemoryBuffer y, const MemoryTile A, const MemoryBuffer x, const MatrixOperation aOperation = MatrixOperation::None, const double alpha = 1.0);

		/**
		* A = cumsum(A)
		*/
		EXPORT int _CumulativeRowSum(MemoryTile A);

		/**
		* X such that A * X = B by means of LU factorization
		*/
		EXPORT int _Solve(const MemoryTile A, MemoryTile B, const MatrixOperation aOperation = MatrixOperation::None);

		/**
		* A = A^(-1) by means of LU factorization
		*/
		EXPORT int _Invert(MemoryTile A, const MatrixOperation aOperation = MatrixOperation::None);
	}
}