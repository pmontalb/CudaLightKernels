#include "CuSparseWrappers.cuh"
#include "DeviceManager.cuh"

EXTERN_C
{
	/**
	* zDense = alpha * xSparse + yDense
	*/
	EXPORT int _SparseAdd(MemoryBuffer z, const SparseMemoryBuffer x, const MemoryBuffer y, const double alpha)
	{
		const cusparseHandle_t& cuSparseHandle = detail::CuSparseHandle();
		const cublasHandle_t& cuBlasHandle = detail::CublasHandle();

		switch (z.mathDomain)
		{
		case MathDomain::Float:
		{
			const int err = cublasScopy(cuBlasHandle, z.size, (float*)y.pointer, 1, (float*)z.pointer, 1);
			if (err)
				return err;

			const float _alpha = (float)alpha;
			cusparseSaxpyi(cuSparseHandle, x.nNonZeros, &_alpha, (float*)x.pointer, (int*)x.indices, (float*)z.pointer, CUSPARSE_INDEX_BASE_ZERO);
			break;
		};
		case MathDomain::Double:
		{
			const int err = cublasDcopy(cuBlasHandle, z.size, (double*)y.pointer, 1, (double*)z.pointer, 1);
			if (err)
				return err;

			cusparseDaxpyi(cuSparseHandle, x.nNonZeros, &alpha, (double*)x.pointer, (int*)x.indices, (double*)z.pointer, CUSPARSE_INDEX_BASE_ZERO);
			break;
		};;
		default: 
			return -1;
		}
		
		cudaDeviceSynchronize(); // axpy is asynch!
		return cudaGetLastError();
	}

	/**
	*	yDense = ASparse * xDense
	*/
	EXPORT int _SparseDot(MemoryBuffer y, const SparseMemoryTile A, const MemoryBuffer x, const double alpha)
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

			err = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				A.nRows, A.nCols, A.nNonZeros,
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

			err = cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				A.nRows, A.nCols, A.nNonZeros,
				&alpha, descr,
				(double*)A.pointer, (int*)A.nNonZeroRows, (int*)A.nonZeroColumnIndices,
				(double*)x.pointer,
				&beta,
				(double*)y.pointer);
			break;
		};;
		default:
			return -1;
		}

		if (err)
			return err;
		return cudaGetLastError();
	}

	/**
	*	ADense = BSparse * CDense
	*/
	EXPORT int _SparseMultiply(MemoryTile A, const SparseMemoryTile B, const MemoryTile C, const unsigned leadingDimensionB, const unsigned leadingDimensionC, const double alpha)
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

			err = cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				leadingDimensionB, C.nCols, leadingDimensionC, B.nNonZeroRows,
				&_alpha,
				descr, (float*)B.pointer, (int*)B.nNonZeroRows, (int*)B.nonZeroColumnIndices,
				(float*)C.pointer, leadingDimensionC,
				&beta,
				(float*)A.pointer, leadingDimensionB);
			break;
		}
		case MathDomain::Double:
		{
			const double beta = 0.0;

			err = cusparseDcsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				leadingDimensionB, C.nCols, leadingDimensionC, B.nNonZeroRows,
				&alpha,
				descr, (double*)B.pointer, (int*)B.nNonZeroRows, (int*)B.nonZeroColumnIndices,
				(double*)C.pointer, leadingDimensionC,
				&beta,
				(double*)A.pointer, leadingDimensionB);
			break;
		}
		default:
			return -1;
		}

		if (err)
			return err;
		return cudaGetLastError();
	}

}