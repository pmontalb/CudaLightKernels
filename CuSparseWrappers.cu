#include "CuSparseWrappers.cuh"
#include "DeviceManager.cuh"

EXTERN_C
{
	EXPORT int _AllocateCsrHandle(SparseMemoryTile&)
	{
		// TODO: use cuSparse generic API
		return 0;
	}
	EXPORT int _DestroyCsrHandle(SparseMemoryTile&)
	{
		// TODO: use cuSparse generic API
		return 0;
	}

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
	EXPORT int _SparseDot(MemoryBuffer& y, SparseMemoryTile& A, const MemoryBuffer& x, const MatrixOperation aOperation, const double alpha, const double beta)
	{
		const cusparseHandle_t& handle = detail::CuSparseHandle();
		const cusparseMatDescr_t& descr = detail::CsrMatrixDescription();

		int err;

		switch (y.mathDomain)
		{
		case MathDomain::Float:
		{
			const float _beta = (float)beta;
			const float _alpha = (float)alpha;

			err = cusparseScsrmv(handle, cusparseOperation[static_cast<int>(aOperation)],
				A.nRows, A.nCols, A.size,
				&_alpha, descr,
				(float*)A.pointer, (int*)A.nNonZeroRows, (int*)A.nonZeroColumnIndices,
				(float*)x.pointer,
				&_beta,
				(float*)y.pointer);
			break;
		}
		case MathDomain::Double:
		{
			err = cusparseDcsrmv(handle, cusparseOperation[static_cast<int>(aOperation)],
				A.nRows, A.nCols, A.size,
				&alpha, descr,
				(double*)A.pointer, (int*)A.nNonZeroRows, (int*)A.nonZeroColumnIndices,
				(double*)x.pointer,
				&beta,
				(double*)y.pointer);
			break;
		}
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
	EXPORT int _SparseMultiply(MemoryTile& A,  SparseMemoryTile& B, const MemoryTile& C, const MatrixOperation bOperation, const double alpha)
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
			    B.nRows, C.nCols, B.nCols, B.size,
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
				B.nRows, C.nCols, B.nCols, B.size,
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

	EXPORT int _SparseSolve(const SparseMemoryTile& A, MemoryTile& B, const LinearSystemSolverType solver)
	{
		const auto& handle = detail::CuSolverSparseHandle();
		const auto& descr = detail::CsrMatrixDescription();

		int err = -1;

		switch (A.mathDomain)
		{
			case MathDomain::Float:
			{
				for (size_t j = 0; j < B.nCols; ++j)
				{
					int singular = 0;

					switch (solver)
					{
						case LinearSystemSolverType::Qr:
							err = cusolverSpScsrlsvqr(handle,
													  A.nRows,
													  A.size,
													  descr,
													  (float *) A.pointer,
													  (int *) A.nNonZeroRows,
													  (int *) A.nonZeroColumnIndices,
													  ((float *) B.pointer) + j * B.nRows,
													  1e-7,
													  0,
													  (float *) B.pointer,
													  &singular);
							if (err)
								return err;

							break;
						case LinearSystemSolverType::Lu:
							// only host at the moment!
//							err = cusolverSpScsrlsvlu(handle,
//													  A.nRows,
//													  A.size,
//													  descr,
//													  (float *) A.pointer,
//													  (int *) A.nNonZeroRows,
//													  (int *) A.nonZeroColumnIndices,
//													  ((float *) B.pointer) + j * B.nRows,
//													  1e-7,
//													  0,
//													  (float *) B.pointer,
//													  &singular);
//							if (err)
//								return err;
//							break;
						default:
							return CudaKernelException::_NotImplementedException;
					}
				}
				break;
			}
			case MathDomain::Double:
			{
				for (size_t j = 0; j < B.nCols; ++j)
				{
					int singular = 0;
					switch (solver)
					{
						case LinearSystemSolverType::Qr:
							err = cusolverSpDcsrlsvqr(handle,
													  A.nRows,
													  A.size,
													  descr,
													  (double *) A.pointer,
													  (int *) A.nNonZeroRows,
													  (int *) A.nonZeroColumnIndices,
													  ((double *) B.pointer) + j * B.nRows,
													  1e-7,
													  0,
													  (double *) B.pointer,
													  &singular);
							if (err)
								return err;
							break;
						case LinearSystemSolverType::Lu:
							// only host at the moment
//							err = cusolverSpDcsrlsvlu(handle,
//													  A.nRows,
//													  A.size,
//													  descr,
//													  (double *) A.pointer,
//													  (int *) A.nNonZeroRows,
//													  (int *) A.nonZeroColumnIndices,
//													  ((double *) B.pointer) + j * B.nRows,
//													  1e-7,
//													  0,
//													  (float *) B.pointer,
//													  &singular);
//							if (err)
//								return err;
//							break;
						default:
							return CudaKernelException::_NotImplementedException;
					}
				}
				break;
			}
			default:
				return CudaKernelException::_NotImplementedException;
		}

		cudaDeviceSynchronize();

		return cudaGetLastError();
	}
}