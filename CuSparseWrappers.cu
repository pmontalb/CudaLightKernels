#include "CuSparseWrappers.cuh"
#include "DeviceManager.cuh"

EXTERN_C
{
	/**
	* zDense = alpha * xSparse + yDense
	*/
	EXPORT int _SparseAdd(MemoryBuffer& z, const SparseMemoryBuffer& x, const MemoryBuffer& y, const double alpha)
	{
		const cusparseHandle_t& cuSparseHandle = detail::CuSparseHandle();
		const cublasHandle_t& cuBlasHandle = detail::CublasHandle();

		int err = -1;
		switch (z.mathDomain)
		{
		case MathDomain::Float:
		{
			if (cublasScopy(cuBlasHandle, z.size, (float*)y.pointer, 1, (float*)z.pointer, 1))
				return CudaKernelException::_InternalException;

			const float _alpha = (float)alpha;
			err = cusparseSaxpyi(cuSparseHandle, x.size, &_alpha, (float*)x.pointer, (int*)x.indices, (float*)z.pointer, CUSPARSE_INDEX_BASE_ZERO);
			break;
		};
		case MathDomain::Double:
		{
			if (cublasDcopy(cuBlasHandle, z.size, (double*)y.pointer, 1, (double*)z.pointer, 1))
				return CudaKernelException::_InternalException;

			err = cusparseDaxpyi(cuSparseHandle, x.size, &alpha, (double*)x.pointer, (int*)x.indices, (double*)z.pointer, CUSPARSE_INDEX_BASE_ZERO);
			break;
		};;
		default: 
			return CudaKernelException::_NotImplementedException;
		}
		
		cudaDeviceSynchronize(); // axpy is asynch!

		if (err)
			return err;
		return cudaGetLastError();
	}
    EXPORT int _SparseAddRaw(const ptr_t z, const ptr_t x, const ptr_t y, const unsigned nNonZeros, const ptr_t nonZeroIndices, const MemorySpace memorySpace, const MathDomain mathDomain, const unsigned size, const double alpha)
    {
		MemoryBuffer _z(z, size, memorySpace, mathDomain);
		SparseMemoryBuffer _x(x, nNonZeros, nonZeroIndices, memorySpace, mathDomain);
		MemoryBuffer _y(y, size, memorySpace, mathDomain);

		return _SparseAdd(_z, _x, _y, alpha);
    }

    EXPORT int _SparseSubtract(MemoryBuffer& z, const SparseMemoryBuffer& x, const MemoryBuffer& y)
    {
	    return _SparseAdd(z, x, y, -1.0);
    }

	/**
	*	yDense = ASparse * xDense
	*/
	EXPORT int _SparseDot(MemoryBuffer& y, const SparseMemoryTile& A, const MemoryBuffer& x, const MatrixOperation aOperation, const double alpha)
	{
		const cusparseHandle_t& handle = detail::CuSparseHandle();
		const cusparseMatDescr_t& descr = detail::CsrMatrixDescription();

		int err = -1;

		switch (y.mathDomain)
		{
		case MathDomain::Float:
		{
			const float beta = 0.0f;
			const float _alpha = (float)alpha;

			err = cusparseScsrmv(handle, cusparseOperation[static_cast<int>(aOperation)],
				A.nRows, A.nCols, A.size,
				&_alpha, descr,
				(float*)A.pointer, (int*)A.nNonZeroRows, (int*)A.nonZeroColumnIndices,
				(float*)x.pointer,
				&beta,
				(float*)y.pointer);
			break;
		};
		case MathDomain::Double:
		{
			const double beta = 0.0;

			err = cusparseDcsrmv(handle, cusparseOperation[static_cast<int>(aOperation)],
				A.nRows, A.nCols, A.size,
				&alpha, descr,
				(double*)A.pointer, (int*)A.nNonZeroRows, (int*)A.nonZeroColumnIndices,
				(double*)x.pointer,
				&beta,
				(double*)y.pointer);
			break;
		};;
		default:
			return CudaKernelException::_NotImplementedException;
		}

		if (err)
			return err;
		return cudaGetLastError();
	}
	EXPORT int _SparseDotRaw(const ptr_t y, const ptr_t A, const ptr_t x,
						  const unsigned nNonZeros, const ptr_t nonZeroColumnIndices, const ptr_t nNonZeroRows,
						  const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain,
						  const MatrixOperation aOperation, const double alpha)
	{
		MemoryBuffer _y(y, nCols, memorySpace, mathDomain);
		SparseMemoryTile _A(A, nNonZeros, nonZeroColumnIndices, nNonZeroRows, nRows, nCols, memorySpace, mathDomain);
		MemoryBuffer _x(x, nCols, memorySpace, mathDomain);

		return _SparseDot(_y, _A, _x, aOperation, alpha);
    }

	/**
	*	ADense = BSparse * CDense
	*/
	EXPORT int _SparseMultiply(MemoryTile& A, const SparseMemoryTile& B, const MemoryTile& C, const MatrixOperation bOperation, const double alpha)
	{
		const cusparseHandle_t& handle = detail::CuSparseHandle();
		const cusparseMatDescr_t& descr = detail::CsrMatrixDescription();

		int err = -1;

		switch (A.mathDomain)
		{
		case MathDomain::Float:
		{
			const float beta = 0.0f;
			const float _alpha = (float)alpha;

			err = cusparseScsrmm(handle, cusparseOperation[static_cast<int>(bOperation)],
				B.leadingDimension, C.nCols, C.leadingDimension, B.nNonZeroRows,
				&_alpha,
				descr, (float*)B.pointer, (int*)B.nNonZeroRows, (int*)B.nonZeroColumnIndices,
				(float*)C.pointer, C.leadingDimension,
				&beta,
				(float*)A.pointer, A.leadingDimension);
			break;
		}
		case MathDomain::Double:
		{
			const double beta = 0.0;

			err = cusparseDcsrmm(handle, cusparseOperation[static_cast<int>(bOperation)],
				B.leadingDimension, C.nCols, C.leadingDimension, B.nNonZeroRows,
				&alpha,
				descr, (double*)B.pointer, (int*)B.nNonZeroRows, (int*)B.nonZeroColumnIndices,
				(double*)C.pointer, C.leadingDimension,
				&beta,
				(double*)A.pointer, B.leadingDimension);
			break;
		}
		default:
			return CudaKernelException::_NotImplementedException;
		}

		if (err)
			return err;
		return cudaGetLastError();
	}

	EXPORT int _SparseMultiplyRaw(const ptr_t A, const ptr_t B, const ptr_t C,
								  const unsigned nNonZeros, const ptr_t nonZeroColumnIndices, const ptr_t nNonZeroRows,
								  const unsigned nRowsB, const unsigned nRowsC, const unsigned nColsC, const MemorySpace memorySpace, const MathDomain mathDomain,
								  const unsigned leadingDimensionB, const unsigned leadingDimensionC, const MatrixOperation bOperation, const double alpha)
	{
		MemoryTile _A(A, nRowsB, nColsC, leadingDimensionB, memorySpace, mathDomain);
		SparseMemoryTile _B(B, nNonZeros, nonZeroColumnIndices, nNonZeroRows, nRowsB, nRowsC, leadingDimensionB, memorySpace, mathDomain);
		MemoryTile _C(C, nRowsC, nColsC, leadingDimensionC, memorySpace, mathDomain);

		return _SparseMultiply(_A, _B, _C, bOperation, alpha);
	}
}