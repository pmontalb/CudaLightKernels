#include "CuSparseWrappers.cuh"
#include "DeviceManager.cuh"
#include "MemoryManager.cuh"

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

                        const float beta = 1.0f;
			const float _alpha = (float)alpha;
                        cusparseSpVecDescr_t x_descr;
                        cusparseCreateSpVec(&x_descr, x.size, x.size, (int*)x.indices, (float*)x.pointer, cusparseIndexType_t ::CUSPARSE_INDEX_32I, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO, cudaDataType_t::CUDA_R_32F);

                        cusparseDnVecDescr_t z_descr;
                        cusparseCreateDnVec(&z_descr, x.size, (float*)z.pointer, cudaDataType_t::CUDA_R_32F);

                        err = cusparseAxpby(cuSparseHandle, &_alpha, x_descr, &beta, z_descr);
			break;
		};
		case MathDomain::Double:
		{
			if (cublasDcopy(cuBlasHandle, z.size, (double*)y.pointer, 1, (double*)z.pointer, 1))
				return CudaKernelException::_InternalException;

                        const double beta = 1.0;

                        cusparseSpVecDescr_t x_descr;
                        cusparseCreateSpVec(&x_descr, x.size, x.size, (int*)x.indices, (float*)x.pointer, cusparseIndexType_t ::CUSPARSE_INDEX_32I, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO, cudaDataType_t::CUDA_R_64F);

                        cusparseDnVecDescr_t z_descr;
                        cusparseCreateDnVec(&z_descr, x.size, (float*)z.pointer, cudaDataType_t::CUDA_R_64F);

                        err = cusparseAxpby(cuSparseHandle, &alpha, x_descr, &beta, z_descr);

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

		int err;

		switch (y.mathDomain)
		{
		case MathDomain::Float:
		{
			const auto _beta = (float)beta;
			const auto _alpha = (float)alpha;

                        cusparseSpMatDescr_t A_descr;
                        cusparseCreateCsr(&A_descr, A.nRows, A.nCols, A.size, (int*)A.nNonZeroRows, (int*)A.nonZeroColumnIndices, (float*)A.pointer,
                                          cusparseIndexType_t::CUSPARSE_INDEX_32I, cusparseIndexType_t::CUSPARSE_INDEX_32I, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO, cudaDataType_t::CUDA_R_32F);

                        cusparseDnVecDescr_t x_descr;
                        cusparseCreateDnVec(&x_descr, x.size, (float*)x.pointer, cudaDataType_t::CUDA_R_32F);

                        cusparseDnVecDescr_t y_descr;
                        cusparseCreateDnVec(&y_descr, y.size, (float*)y.pointer, cudaDataType_t::CUDA_R_32F);

                        size_t bufferSize;
                        MemoryBuffer buf(0, 0, MemorySpace::Device, y.mathDomain);
                        err = cusparseSpMV_bufferSize(handle, cusparseOperation[static_cast<int>(aOperation)], &_alpha,
                                           A_descr, x_descr, &_beta, y_descr, cudaDataType_t::CUDA_R_32F, cusparseSpMVAlg_t::CUSPARSE_SPMV_CSR_ALG1, &bufferSize);
                        if (bufferSize > 0) {
                          buf.size = static_cast<unsigned>(bufferSize);
                          _Alloc(buf);
                        }
                        err = cusparseSpMV(handle, cusparseOperation[static_cast<int>(aOperation)], &_alpha,
                                     A_descr, x_descr, &_beta, y_descr, cudaDataType_t::CUDA_R_32F, cusparseSpMVAlg_t::CUSPARSE_SPMV_CSR_ALG1, (float*)buf.pointer);
                        if (bufferSize > 0)
                          _Free(buf);

			break;
		}
		case MathDomain::Double:
		{
                  cusparseSpMatDescr_t A_descr;
                  cusparseCreateCsr(&A_descr, A.nRows, A.nCols, A.size, (int*)A.nNonZeroRows, (int*)A.nonZeroColumnIndices, (double*)A.pointer,
                                    cusparseIndexType_t::CUSPARSE_INDEX_32I, cusparseIndexType_t::CUSPARSE_INDEX_32I, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO, cudaDataType_t::CUDA_R_64F);

                  cusparseDnVecDescr_t x_descr;
                  cusparseCreateDnVec(&x_descr, x.size, (double*)x.pointer, cudaDataType_t::CUDA_R_64F);

                  cusparseDnVecDescr_t y_descr;
                  cusparseCreateDnVec(&y_descr, y.size, (double*)y.pointer, cudaDataType_t::CUDA_R_64F);

                  size_t bufferSize;
                  MemoryBuffer buf(0, 0, MemorySpace::Device, y.mathDomain);
                  err = cusparseSpMV_bufferSize(handle, cusparseOperation[static_cast<int>(aOperation)], &alpha,
                                                A_descr, x_descr, &beta, y_descr, cudaDataType_t::CUDA_R_64F, cusparseSpMVAlg_t::CUSPARSE_SPMV_CSR_ALG1, &bufferSize);
                  if (bufferSize > 0) {
                    buf.size = static_cast<unsigned>(bufferSize);
                    _Alloc(buf);
                  }
                  err = cusparseSpMV(handle, cusparseOperation[static_cast<int>(aOperation)], &alpha,
                                     A_descr, x_descr, &beta, y_descr, cudaDataType_t::CUDA_R_64F, cusparseSpMVAlg_t::CUSPARSE_SPMV_CSR_ALG1, (double*)buf.pointer);
                  if (bufferSize > 0)
                    _Free(buf);
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

                        cusparseDnMatDescr_t A_descr;
                        cusparseCreateDnMat(&A_descr, A.nRows, A.nCols, A.leadingDimension, (float*)A.pointer, cudaDataType_t::CUDA_R_32F, cusparseOrder_t ::CUSPARSE_ORDER_COL);

                        cusparseSpMatDescr_t B_descr;
                        cusparseCreateCsr(&B_descr, B.nRows, B.nCols, B.size, (int*)B.nNonZeroRows, (int*)B.nonZeroColumnIndices, (float*)B.pointer,
                                          cusparseIndexType_t::CUSPARSE_INDEX_64I, cusparseIndexType_t::CUSPARSE_INDEX_64I, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO, cudaDataType_t::CUDA_R_32F);

                        cusparseDnMatDescr_t C_descr;
                        cusparseCreateDnMat(&C_descr, C.nRows, C.nCols, C.leadingDimension, (float*)C.pointer, cudaDataType_t::CUDA_R_32F, cusparseOrder_t ::CUSPARSE_ORDER_COL);

                        size_t bufferSize;
                        MemoryBuffer buf(0, 0, MemorySpace::Device, A.mathDomain);
                        err = cusparseSpMM_bufferSize(handle, cusparseOperation[static_cast<int>(bOperation)],
                                           cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &_alpha, B_descr, C_descr, &beta, A_descr, cudaDataType_t::CUDA_R_32F, cusparseSpMMAlg_t::CUSPARSE_SPMM_CSR_ALG1, &bufferSize);
                        if (bufferSize > 0) {
                          buf.size = static_cast<unsigned>(bufferSize);
                          _Alloc(buf);
                        }
                        err = cusparseSpMM(handle, cusparseOperation[static_cast<int>(bOperation)],
                                     cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &_alpha, B_descr, C_descr, &beta, A_descr, cudaDataType_t::CUDA_R_32F, cusparseSpMMAlg_t::CUSPARSE_SPMM_CSR_ALG1, (float*)(buf.pointer));
                        if (bufferSize > 0)
                          _Free(buf);
			break;
		}
		case MathDomain::Double:
		{
			const double beta = 0.0;

                        cusparseDnMatDescr_t A_descr;
                        cusparseCreateDnMat(&A_descr, A.nRows, A.nCols, A.leadingDimension, (double*)A.pointer, cudaDataType_t::CUDA_R_64F, cusparseOrder_t ::CUSPARSE_ORDER_COL);

                        cusparseSpMatDescr_t B_descr;
                        cusparseCreateCsr(&B_descr, B.nRows, B.nCols, B.size, (int*)B.nNonZeroRows, (int*)B.nonZeroColumnIndices, (double*)B.pointer,
                                          cusparseIndexType_t::CUSPARSE_INDEX_32I, cusparseIndexType_t::CUSPARSE_INDEX_32I, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO, cudaDataType_t::CUDA_R_64F);

                        cusparseDnMatDescr_t C_descr;
                        cusparseCreateDnMat(&C_descr, C.nRows, C.nCols, C.leadingDimension, (double*)C.pointer, cudaDataType_t::CUDA_R_64F, cusparseOrder_t ::CUSPARSE_ORDER_COL);

                        size_t bufferSize;
                        MemoryBuffer buf(0, 0, MemorySpace::Device, A.mathDomain);
                        err = cusparseSpMM_bufferSize(handle, cusparseOperation[static_cast<int>(bOperation)],
                                           cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, B_descr, C_descr, &beta, A_descr, cudaDataType_t::CUDA_R_64F, cusparseSpMMAlg_t::CUSPARSE_SPMM_CSR_ALG1, &bufferSize);
                        if (bufferSize > 0) {
                          buf.size = static_cast<unsigned>(bufferSize);
                          _Alloc(buf);
                        }

                        err = cusparseSpMM(handle, cusparseOperation[static_cast<int>(bOperation)],
                                           cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, B_descr, C_descr, &beta, A_descr, cudaDataType_t::CUDA_R_64F, cusparseSpMMAlg_t::CUSPARSE_SPMM_CSR_ALG1, (double*)(buf.pointer));
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
