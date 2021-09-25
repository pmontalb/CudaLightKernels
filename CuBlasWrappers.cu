#include "CuBlasWrappers.cuh"
#include "DeviceManager.cuh"
#include "BufferInitializer.cuh"
#include "MemoryManager.cuh"
#include <cublas.h>

EXTERN_C
{
	/**
	* z = x + y
	*/
	EXPORT int _Add(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha)
	{
		const cublasHandle_t& handle = detail::CublasHandle();

		switch (z.mathDomain)
		{
		case MathDomain::Float:
			{
				int err = cublasScopy(handle, z.size, (float*)y.pointer, 1, (float*)z.pointer, 1);
				if (err)
					return err;

				const float _alpha = (float)alpha;
				return cublasSaxpy(handle, z.size, &_alpha, (float*)x.pointer, 1, (float*)z.pointer, 1);
			}
		case MathDomain::Double:
			{
				int err = cublasDcopy(handle, z.size, (double*)y.pointer, 1, (double*)z.pointer, 1);
				if (err)
					return err;

				return cublasDaxpy(handle, z.size, &alpha, (double*)x.pointer, 1, (double*)z.pointer, 1);
			}
		case MathDomain::Int:
			{
			    CUDA_CALL_SINGLE(__IntAffineOperationNaive__, (int*)z.pointer, (int*)x.pointer, (int*)y.pointer, z.size, (int)alpha, 1, 0);
				return cudaGetLastError();
			}
		default:
			return CudaKernelException::_NotImplementedException;
		}
	}
	EXPORT int _AddRaw(const ptr_t z, const ptr_t x, const ptr_t y, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const double alpha)
	{
		MemoryBuffer _z(z, size, memorySpace, mathDomain);
		MemoryBuffer _x(x, size, memorySpace, mathDomain);
		MemoryBuffer _y(y, size, memorySpace, mathDomain);
		return _Add(_z, _x, _y, alpha);
	 }

	/**
	* z += x
	*/
	EXPORT int _AddEqual(MemoryBuffer& z, const MemoryBuffer& x, const double alpha)
	{
		const cublasHandle_t& handle = detail::CublasHandle();

		switch (z.mathDomain)
		{
		case MathDomain::Float:
		{
			const float _alpha = (float)alpha;
			return cublasSaxpy(handle, z.size, &_alpha, (float*)x.pointer, 1, (float*)z.pointer, 1);
		}
		case MathDomain::Double:
			return cublasDaxpy(handle, z.size, &alpha, (double*)x.pointer, 1, (double*)z.pointer, 1);
		case MathDomain::Int:
			CUDA_CALL_SINGLE(__IntAffineOperationNaive__, (int*)z.pointer, (int*)x.pointer, (int*)x.pointer, z.size, 0, (int)alpha, 1);
			return cudaGetLastError();
		default:
			return CudaKernelException::_NotImplementedException;
		}
	}
	EXPORT int _AddEqualRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const double alpha)
	{
		MemoryBuffer _z(z, size, memorySpace, mathDomain);
		MemoryBuffer _x(x, size, memorySpace, mathDomain);
		return _AddEqual(_z, _x, alpha);
    }

    EXPORT int _Subtract(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y)
    {
        return _Add(z, y, x, -1.0);
    }

    EXPORT int _SubtractEqual(MemoryBuffer& z, const MemoryBuffer& x)
    {
        return _AddEqual(z, x, -1.0);
    }

	/**
	* A += alpha * B
	*/
	EXPORT int _AddEqualMatrix(MemoryTile& A, const MemoryTile& B, const MatrixOperation aOperation, const MatrixOperation bOperation, const double alpha, const double beta)
	{
		const cublasHandle_t& handle = detail::CublasHandle();

		switch (A.mathDomain)
		{
		case MathDomain::Float:
		{
			const float _alpha = (float)alpha;
			const float _beta = (float)beta;
			return cublasSgeam(handle, cublasOperation[static_cast<unsigned>(aOperation)], cublasOperation[static_cast<unsigned>(bOperation)],
				A.nRows, A.nCols,
				&_alpha,
				(float*)A.pointer, A.leadingDimension,
				&_beta,
				(float*)B.pointer, B.leadingDimension,
				(float*)A.pointer, A.leadingDimension);
		}
		case MathDomain::Double:
		{
			return cublasDgeam(handle, cublasOperation[static_cast<unsigned>(aOperation)], cublasOperation[static_cast<unsigned>(bOperation)],
				A.nRows, A.nCols,
				&alpha,
				(double*)A.pointer, A.leadingDimension,
				&beta,
				(double*)B.pointer, B.leadingDimension,
				(double*)A.pointer, A.leadingDimension);
		}
		default:
			return CudaKernelException::_NotImplementedException;
		}
	}
	EXPORT int _AddEqualMatrixRaw(const ptr_t A, const ptr_t B, const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain, const MatrixOperation aOperation, const MatrixOperation bOperation, const double alpha, const double beta)
	{
		MemoryTile _A(A, nRows, nCols, memorySpace, mathDomain);
		MemoryTile _B(B, nRows, nCols, memorySpace, mathDomain);
		return _AddEqualMatrix(_A, _B, aOperation, bOperation, alpha, beta);
	}

	/**
	* z *= alpha
	*/
	EXPORT int _Scale(MemoryBuffer& z, const double alpha)
	{
		const cublasHandle_t& handle = detail::CublasHandle();
		switch (z.mathDomain)
		{
		case MathDomain::Float:
		{
			const float _alpha = (float)alpha;
			return cublasSscal(handle, z.size, &_alpha, (float*)z.pointer, 1);
		}
		case MathDomain::Double:
			return cublasDscal(handle, z.size, &alpha, (double*)z.pointer, 1);
		case MathDomain::Int:
			CUDA_CALL_SINGLE(__IntAffineOperationNaive__, (int*)z.pointer, (int*)z.pointer, (int*)z.pointer, z.size, 0, 0, (int)alpha);
			return cudaGetLastError();
		default:
			return CudaKernelException::_NotImplementedException;
		}
	}
	EXPORT int _ScaleRaw(const ptr_t z, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const double alpha)
	{
		MemoryBuffer _z(z, size, memorySpace, mathDomain);
		return _Scale(_z, alpha);
	}

	EXPORT int _ScaleColumns(MemoryTile& z, const MemoryBuffer& alpha)
	{
		const cublasHandle_t& handle = detail::CublasHandle();

		cublasPointerMode_t originalPointerMode;
		cublasGetPointerMode(handle, &originalPointerMode);
		cublasSetPointerMode(handle, cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE);
		switch (z.mathDomain)
		{
			case MathDomain::Float:
				for (size_t i = 0; i < z.nCols; ++i)
					cublasSscal(handle, z.nRows, (float*)(alpha.pointer + i * alpha.ElementarySize()), (float*)(z.pointer + i * z.nRows * z.ElementarySize()), 1);
				break;
			case MathDomain::Double:
				for (size_t i = 0; i < z.nCols; ++i)
					cublasDscal(handle, z.nRows, (double*)(alpha.pointer + i * alpha.ElementarySize()), (double*)(z.pointer + i * z.nRows * z.ElementarySize()), 1);
				break;
			default:
				cublasSetPointerMode(handle, originalPointerMode);
				return CudaKernelException::_NotImplementedException;
		}
		cublasSetPointerMode(handle, originalPointerMode);

		return cudaGetLastError();
	}
	EXPORT int _ScaleColumnsRaw(const ptr_t z, const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain, const ptr_t alpha)
	{
		MemoryTile _z(z, nRows, nCols, memorySpace, mathDomain);
		MemoryBuffer _alpha(alpha, nCols, memorySpace, mathDomain);
		return _ScaleColumns(_z, _alpha);
	}

	EXPORT int _ElementwiseProduct(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha)
	{
		//#define USE_NAIVE_ELEMENTWISE_PRODUCT
		#ifndef USE_NAIVE_ELEMENTWISE_PRODUCT
			const cublasHandle_t& handle = detail::CublasHandle();

			switch (z.mathDomain)
			{
			case MathDomain::Float:
			{
				const float _alpha = (float)alpha;
				const float beta = 0.0f;

				return cublasSsbmv(handle, CUBLAS_FILL_MODE_UPPER,
					z.size, 0,  // Just the diagonal; 0 super-diagonal bands
					&_alpha,
					(float*)x.pointer, 1,
					(float*)y.pointer, 1,
					&beta,
					(float*)z.pointer, 1);
			}
			case MathDomain::Double:
			{
				const double beta = 0.0;
				return cublasDsbmv(handle, CUBLAS_FILL_MODE_UPPER,
					z.size, 0,  // Just the diagonal; 0 super-diagonal bands
					&alpha,
					(double*)x.pointer, 1,
					(double*)y.pointer, 1,
					&beta,
					(double*)z.pointer, 1);
			}
			case MathDomain::Int:
				CUDA_CALL_SINGLE(__ElementwiseProductNaive__<int COMMA false>, (int*)z.pointer, (int*)x.pointer, (int*)y.pointer, z.size, (int)alpha);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
			}
		#else

			switch (z.mathDomain)
			{
				case MathDomain::Float:
					CUDA_CALL_SINGLE(__ElementwiseProductNaive__<float COMMA false>, (float*)z.pointer, (float*)x.pointer, (float*)y.pointer, z.size, (float)alpha);
					break;
				case MathDomain::Double:
					CUDA_CALL_DOUBLE(__ElementwiseProductNaive__<double COMMA false>, (double*)z.pointer, (double*)x.pointer, (double*)y.pointer, z.size, alpha);
					break;
				case MathDomain::Int:
					CUDA_CALL_SINGLE(__ElementwiseProductNaive__<int COMMA false>, (int*)z.pointer, (int*)x.pointer, (int*)y.pointer, z.size, (int)alpha);
					break;
				default:
					return CudaKernelException::_NotImplementedException;
			}

		#endif // USE_NAIVE_ELEMENTWISE_PRODUCT

		return cudaGetLastError();
	}
	EXPORT int _ElementwiseProductRaw(const ptr_t z, const ptr_t x, const ptr_t y, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const double alpha)
	{
		MemoryBuffer _z(z, size, memorySpace, mathDomain);
		MemoryBuffer _x(x, size, memorySpace, mathDomain);
		MemoryBuffer _y(y, size, memorySpace, mathDomain);
		return _ElementwiseProduct(_z, _x, _y, alpha);
	}

	EXPORT int _ElementwiseDivision(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha)
	{
		#ifndef USE_NAIVE_ELEMENTWISE_PRODUCT
			return CudaKernelException::_NotImplementedException;
		#else

		switch (z.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_SINGLE(__ElementwiseProductNaive__<float COMMA true>, (float*)z.pointer, (float*)x.pointer, (float*)y.pointer, z.size, (float)alpha);
				break;
			case MathDomain::Double:
				CUDA_CALL_DOUBLE(__ElementwiseProductNaive__<double COMMA true>, (double*)z.pointer, (double*)x.pointer, (double*)y.pointer, z.size, alpha);
				break;
			case MathDomain::Int:
				CUDA_CALL_SINGLE(__ElementwiseProductNaive__<int COMMA true>, (int*)z.pointer, (int*)x.pointer, (int*)y.pointer, z.size, (int)alpha);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}

		#endif // USE_NAIVE_ELEMENTWISE_PRODUCT

		return cudaGetLastError();
	}
	EXPORT int _ElementwiseDivisionRaw(const ptr_t z, const ptr_t x, const ptr_t y, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const double alpha)
	{
		MemoryBuffer _z(z, size, memorySpace, mathDomain);
		MemoryBuffer _x(x, size, memorySpace, mathDomain);
		MemoryBuffer _y(y, size, memorySpace, mathDomain);
		return _ElementwiseDivision(_z, _x, _y, alpha);
	}

	EXPORT int _SubMultiply(MemoryTile& A, const MemoryTile& B, const MemoryTile& C, const unsigned nRowsB, const unsigned nColsB, const unsigned nColsC, const MatrixOperation bOperation, const MatrixOperation cOperation, const double alpha, const double beta)
	{
		const cublasHandle_t& handle = detail::CublasHandle();
		switch (A.mathDomain)
		{
			case MathDomain::Float:
			{
				const float _alpha = (float)alpha;
				const float _beta = (float)beta;
				return cublasSgemm(handle, cublasOperation[static_cast<unsigned>(bOperation)], cublasOperation[static_cast<unsigned>(cOperation)],
				                   nRowsB, nColsC, nColsB,
				                   &_alpha,
				                   (float*)B.pointer, B.leadingDimension,
				                   (float*)C.pointer, C.leadingDimension,
				                   &_beta,
				                   (float*)A.pointer, A.leadingDimension);
			}
			case MathDomain::Double:
			{
				return cublasDgemm(handle, cublasOperation[static_cast<unsigned>(bOperation)], cublasOperation[static_cast<unsigned>(cOperation)],
				                   nRowsB, nColsC, nColsB,
				                   &alpha,
				                   (double*)B.pointer, B.leadingDimension,
				                   (double*)C.pointer, C.leadingDimension,
				                   &beta,
				                   (double*)A.pointer, A.leadingDimension);
			}
			default:
				return CudaKernelException::_NotImplementedException;
		}
	}
	EXPORT int _SubMultiplyRaw(const ptr_t A, const ptr_t B, const ptr_t C, const unsigned nRowsB, const unsigned nRowsC, const unsigned nColsC, const MemorySpace memorySpace, const MathDomain mathDomain, const unsigned leadingDimensionA, const unsigned leadingDimensionB, const unsigned leadingDimensionC, const unsigned nColsB, const MatrixOperation bOperation, const MatrixOperation cOperation, const double alpha, const double beta)
	{
		MemoryTile _A(A, leadingDimensionA, nColsC, leadingDimensionA, memorySpace, mathDomain);
		MemoryTile _B(B, leadingDimensionB, nRowsC, leadingDimensionB, memorySpace, mathDomain);
		MemoryTile _C(C, leadingDimensionC, nColsC, leadingDimensionC, memorySpace, mathDomain);
		return _SubMultiply(_A, _B, _C, nRowsB, nColsB, nColsC, bOperation, cOperation, alpha, beta);
	}

	EXPORT int _Multiply(MemoryTile& A, const MemoryTile& B, const MemoryTile& C, const MatrixOperation bOperation, const MatrixOperation cOperation, const double alpha, const double beta)
	{
		return _SubMultiply(A, B, C, B.nRows, B.nCols, C.nCols, bOperation, cOperation, alpha, beta);
	}
	EXPORT int _MultiplyRaw(const ptr_t A, const ptr_t B, const ptr_t C, const unsigned nRowsB, const unsigned nRowsC, const unsigned nColsC, const MemorySpace memorySpace, const MathDomain mathDomain, const unsigned leadingDimensionB, const unsigned leadingDimensionC, const MatrixOperation bOperation, const MatrixOperation cOperation, const double alpha, const double beta)
{
	MemoryTile _A(A, nRowsB, nColsC, leadingDimensionB, memorySpace, mathDomain);
	MemoryTile _B(B, nRowsB, nRowsC, leadingDimensionB, memorySpace, mathDomain);
	MemoryTile _C(C, nRowsC, nColsC, leadingDimensionC, memorySpace, mathDomain);
	return _Multiply(_A, _B, _C, bOperation, cOperation, alpha, beta);
}

	EXPORT int _BatchedMultiply(MemoryCube& A, const MemoryCube& B, const MemoryCube& C, const unsigned strideB, const unsigned strideC, const MatrixOperation bOperation, const MatrixOperation cOperation, const double alpha, const double beta)
	{
		const cublasHandle_t& handle = detail::CublasHandle();
		switch (A.mathDomain)
		{
			case MathDomain::Float:
			{
				float _alpha = (float)alpha;
				float _beta = (float)beta;

				return cublasSgemmStridedBatched(handle, cublasOperation[static_cast<unsigned>(bOperation)], cublasOperation[static_cast<unsigned>(cOperation)],
				                                 A.nRows, A.nCols, B.nCols,
				                                 &_alpha,
				                                 (float*)B.pointer, B.leadingDimension, strideB,
				                                 (float*)C.pointer, C.leadingDimension, strideC,
				                                 &_beta,
				                                 (float*)A.pointer, A.leadingDimension, A.nRows * A.nCols,
				                                 A.nCubes);
			}
			case MathDomain::Double:
			{
				return cublasDgemmStridedBatched(handle, cublasOperation[static_cast<unsigned>(bOperation)], cublasOperation[static_cast<unsigned>(cOperation)],
				                                 A.nRows, A.nCols, B.nCols,
				                                 &alpha,
				                                 (double*)B.pointer, B.leadingDimension, strideB,
				                                 (double*)C.pointer, C.leadingDimension, strideC,
				                                 &beta,
				                                 (double*)A.pointer, A.leadingDimension, A.nRows * A.nCols,
				                                 A.nCubes);
			}
			default:
				return CudaKernelException::_NotImplementedException;
		}
	}
	EXPORT int _BatchedMultiplyRaw(const ptr_t A, const ptr_t B, const ptr_t C, const unsigned nRowsB, const unsigned nRowsC, const unsigned nColsC, const unsigned nCubes, const MemorySpace memorySpace, const MathDomain mathDomain, const unsigned leadingDimensionB, const unsigned leadingDimensionC, const MatrixOperation bOperation, const MatrixOperation cOperation, const double alpha, const double beta)
	{
		MemoryCube _A(A, nRowsB, nColsC, nCubes, memorySpace, mathDomain);
		MemoryCube _B(B, nRowsB, nRowsC, nCubes, memorySpace, mathDomain);
		MemoryCube _C(C, nRowsC, nColsC, nCubes, memorySpace, mathDomain);
		return _BatchedMultiply(_A, _B, _C, _B.nRows * _B.nCols, _C.nRows * _C.nCols, bOperation, cOperation, alpha, beta);
	}

	EXPORT int _Dot(MemoryBuffer& y, const MemoryTile& A, const MemoryBuffer& x, const MatrixOperation aOperation, const double alpha, const double beta)
	{
		const cublasHandle_t& handle = detail::CublasHandle();
		switch (A.mathDomain)
		{
		case MathDomain::Float:
		{
			const float _alpha = (float)alpha;
			const float _beta = (float)beta;
			return cublasSgemv(handle, cublasOperation[static_cast<unsigned>(aOperation)],
				A.nRows, A.nCols,
				&_alpha,
				(float*)A.pointer, A.leadingDimension,
				(float*)x.pointer, 1,
				&_beta,
				(float*)y.pointer, 1);
		}
		case MathDomain::Double:
		{
			return cublasDgemv(handle, cublasOperation[static_cast<unsigned>(aOperation)],
				A.nRows, A.nCols,
				&alpha,
				(double*)A.pointer, A.leadingDimension,
				(double*)x.pointer, 1,
				&beta,
				(double*)y.pointer, 1);
		}
		default:
			return CudaKernelException::_NotImplementedException;
		}
	}
	EXPORT int _DotRaw(const ptr_t y, const ptr_t A, const ptr_t x, const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain, const MatrixOperation aOperation, const double alpha, const double beta)
	{
		MemoryBuffer _x(x, nCols, memorySpace, mathDomain);
		MemoryBuffer _y(y, nCols, memorySpace, mathDomain);
		MemoryTile _A(A, nRows, nCols, memorySpace, mathDomain);
		return _Dot(_y, _A, _x, aOperation, alpha, beta);
	}

	EXPORT int _KroneckerProduct(MemoryTile& A, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha)
	{
		const cublasHandle_t& handle = detail::CublasHandle();
		switch (A.mathDomain)
		{
			case MathDomain::Float:
			{
				const float _alpha = (float)alpha;
				return cublasSger(handle, x.size, y.size, &_alpha, (float*)x.pointer, 1, (float*)y.pointer, 1, (float*)A.pointer, A.nRows);
			}
			case MathDomain::Double:
			{
				return cublasDger(handle, x.size, y.size, &alpha, (double*)x.pointer, 1, (double*)y.pointer, 1, (double*)A.pointer, A.nRows);
			}
			default:
				return CudaKernelException::_NotImplementedException;
		}
	}
	EXPORT int _KroneckerProductRaw(const ptr_t A, const ptr_t x, const ptr_t y, const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain, const double alpha)
	{
		MemoryBuffer _x(x, nRows, memorySpace, mathDomain);
		MemoryBuffer _y(y, nCols, memorySpace, mathDomain);
		MemoryTile _A(A, nRows, nCols, memorySpace, mathDomain);
		return _KroneckerProduct(_A, _x, _y, alpha);
	}

	EXPORT int _BatchedTransposedKroneckerProduct(MemoryCube& T, const MemoryTile& x, const MemoryTile& y, const double alpha)
	{
		static constexpr size_t nStreams = { 32 };
		cudaStream_t streams[nStreams];
		int err = 0;
		for (size_t i = 0; i < nStreams; i++)
		{
			err = cudaStreamCreate(&streams[i]);
			if (err)
				return err;
		}

		const cublasHandle_t& handle = detail::CublasHandle();
		const size_t nCubesPerStream = (T.nCubes + nStreams) / nStreams;

		size_t cubeStart = 0;
		size_t cubeEnd = nCubesPerStream;

		MemoryTile cache1(T.pointer, T.nRows, T.nCols, T.memorySpace, T.mathDomain);
		MemoryBuffer cache2(x.pointer, x.nRows, x.memorySpace, x.mathDomain);
		MemoryBuffer cache3(y.pointer, y.nRows, y.memorySpace, y.mathDomain);

		const size_t tOffset = T.nRows * T.nCols * T.ElementarySize();
		const size_t xOffset = x.nRows * x.ElementarySize();
		const size_t yOffset = y.nRows * y.ElementarySize();
		for (size_t i = 0; i < nStreams; i++)
		{
			cublasSetStream(handle, streams[i]);
			for (size_t j = cubeStart; j < cubeEnd; ++j)
			{
				err = _KroneckerProduct(cache1, cache2, cache3, alpha);
				if (err)
					return err;

				cache1.pointer += tOffset;
				cache2.pointer += xOffset;
				cache3.pointer += yOffset;
			}

			cubeStart = cubeEnd;
			cubeEnd = min(cubeEnd + nCubesPerStream, static_cast<size_t>(T.nCubes));

			if (cubeStart == T.nCubes)
				break;
		}

		for (size_t i = 0; i < nStreams; i++)
		{
			err = cudaStreamDestroy(streams[i]);
			if (err)
				return err;
		}

		// reset stream
		err = cublasSetStream(handle, nullptr);
		if (err)
			return err;

		return cudaGetLastError();
	}
	EXPORT int _BatchedTransposedKroneckerProductRaw(const ptr_t A, const ptr_t x, const ptr_t y, const unsigned nRows, const unsigned nCols, const unsigned nCubes, const MemorySpace memorySpace, const MathDomain mathDomain, const double alpha)
	{
		MemoryTile _x(x, nRows, nCubes, memorySpace, mathDomain);
		MemoryTile _y(y, nCols, nCubes, memorySpace, mathDomain);
		MemoryCube _A(A, nRows, nCols, nCubes, memorySpace, mathDomain);
		return _BatchedTransposedKroneckerProduct(_A, _x, _y, alpha);
	}

	EXPORT int _CumulativeRowSum(MemoryTile& A)
	{
		const cublasHandle_t& handle = detail::CublasHandle();
		int err = -1;

		switch(A.mathDomain)
		{
		case MathDomain::Float:
		{
			float *onesPtr = nullptr;
			err = cudaMalloc((void **)&onesPtr, A.nRows * A.nCols * sizeof(float));
			if (err)
				return err;
			MemoryTile ones((ptr_t)onesPtr, A.nRows, A.nCols, A.memorySpace, A.mathDomain);
			_OnesUpperTriangular(ones);

			float *buffer = nullptr;
			err = cudaMalloc((void **)&buffer, A.nRows * A.nCols * sizeof(float));
			if (err)
				return err;

			err = cudaMemcpy(buffer, (void*)A.pointer, A.nRows * A.nCols * sizeof(float), cudaMemcpyDeviceToDevice);
			if (err)
				return err;

			float alpha = 1.0f, beta = 0.0f;
			err = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
				A.nRows, A.nCols, A.nRows,
				&alpha,
				buffer, A.leadingDimension,
				(float*)ones.pointer, A.leadingDimension,
				&beta,
				(float*)A.pointer, A.leadingDimension);

			cudaFree((void*)ones.pointer);
			cudaFree(buffer);
			break;
		}
		case MathDomain::Double:
		{
			double *onesPtr = nullptr;
			err = cudaMalloc((void **)&onesPtr, A.nRows * A.nCols * sizeof(double));
			if (err)
				return err;

			MemoryTile ones((ptr_t)onesPtr, A.nRows, A.nCols, A.memorySpace, A.mathDomain);
			_OnesUpperTriangular(ones);

			double *buffer = nullptr;
			err = cudaMalloc((void **)&buffer, A.nRows * A.nCols * sizeof(double));
			if (err)
				return err;

			cudaMemcpy(buffer, (void*)A.pointer, A.nRows * A.nCols * sizeof(double), cudaMemcpyDeviceToDevice);

			double alpha = 1.0, beta = 0.0;
			err = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
				A.nRows, A.nCols, A.nRows,
				&alpha,
				buffer, A.leadingDimension,
				(double*)ones.pointer, A.leadingDimension,
				&beta,
				(double*)A.pointer, A.leadingDimension);

			cudaFree((void*)ones.pointer);
			cudaFree(buffer);
			break;
		}
		default:
			return CudaKernelException::_NotImplementedException;
		}

		return err;
	}
	EXPORT int _CumulativeRowSumRaw(const ptr_t A, const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryTile _A(A, nRows, nCols, memorySpace, mathDomain);
		return _CumulativeRowSum(_A);
	}

	/**
	* x = sum(A[:, ])
	*/
	EXPORT int _RowWiseSum(MemoryBuffer& x, const MemoryTile& A, MemoryBuffer& cache, const MatrixOperation aOperation)
	{
		if (cache.size != A.nCols)
		{
			if (cache.pointer != 0)
				_Free(cache);
			cache.pointer = 0;
		}

		if (cache.pointer == 0)
		{
			cache = MemoryBuffer(cache.pointer, A.nCols, A.memorySpace, A.mathDomain);
			_Alloc(cache);
			_Initialize(cache, 1.0);
		}

		return _Dot(x, A, cache, aOperation);
	}
	EXPORT int _RowWiseSumRaw(const ptr_t x, const ptr_t A, const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain, const ptr_t cache, const MatrixOperation aOperation)
	{
		MemoryBuffer _x(x, nRows, memorySpace, mathDomain);
		MemoryTile _A(A, nRows, nCols, memorySpace, mathDomain);
		MemoryBuffer _cache(cache, nCols, memorySpace, mathDomain);
		return _RowWiseSum(_x, _A, _cache, aOperation);
	}

	/**
	* x = sum(A[:, :, ])
	*/
	EXPORT int _CubeWiseSum(MemoryTile& A, const MemoryCube& T, MemoryCube& cacheReshape, MemoryBuffer& cacheOnes)
	{
		if (cacheOnes.size != T.nCubes)
		{
			if (cacheOnes.pointer != 0)
				_Free(cacheOnes);
			cacheOnes.pointer = 0;
		}

		if (cacheOnes.pointer == 0)
		{
			cacheOnes = MemoryBuffer(0, T.nCubes, T.memorySpace, T.mathDomain);
			_Alloc(cacheOnes);
			_Initialize(cacheOnes, 1.0);
		}

		// reshape T into nCols blocks of [nRows * nCubes]
		if (cacheReshape.nRows != T.nRows || cacheReshape.nCols != T.nCubes || cacheReshape.nCubes != T.nCols)
		{
			if (cacheReshape.pointer != 0)
				_Free(cacheReshape);
			cacheReshape.pointer = 0;
		}

		if (cacheReshape.pointer == 0)
		{
			cacheReshape = MemoryCube(0, T.nRows, T.nCubes, T.nCols, T.memorySpace, T.mathDomain);
			_Alloc(cacheReshape);
		}

		const cublasHandle_t& handle = detail::CublasHandle();
		switch (A.mathDomain)
		{
			case MathDomain::Float:
			{
				dim3 blockDim(32, 32);
				dim3 gridDim((cacheReshape.nRows + 32 - 1) / 32, (cacheReshape.nCols + 32 - 1) / 32);
				CUDA_CALL_XY(__Reshape__<float>, gridDim, blockDim, (float*)cacheReshape.pointer, (float*)T.pointer, T.nRows, T.nCols, T.nCubes);
				break;
			}
			case MathDomain::Double:
				CUDA_CALL_DOUBLE(__Reshape__<double>, (double*)cacheReshape.pointer, (double*)T.pointer, T.nRows, T.nCols, T.nCubes);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}

		int err = cudaGetLastError();
		if (err)
			return err;

		MemoryCube tmp1(A.pointer, A.nRows, 1, T.nCubes, A.memorySpace, A.mathDomain);
		MemoryCube tmp2(cacheReshape.pointer, cacheReshape.nRows, cacheReshape.nCols, 0, A.memorySpace, A.mathDomain);
		MemoryCube tmp3(cacheOnes.pointer, cacheOnes.size, 0, 0, A.memorySpace, A.mathDomain);
		return _BatchedMultiply(tmp1, tmp2, tmp3,cacheReshape.nRows * cacheReshape.nCols, 0,MatrixOperation::None, MatrixOperation::None, 1.0, 0.0);
	}
	EXPORT int _CubeWiseSumRaw(const ptr_t A, const ptr_t T, const unsigned nRows, const unsigned nCols, const unsigned nCubes, const MemorySpace memorySpace, const MathDomain mathDomain, const ptr_t cacheReshape, const ptr_t cacheOnes)
	{
		MemoryTile _A(A, nRows, nCols, memorySpace, mathDomain);
		MemoryCube _T(T, nRows, nCols, nCubes, memorySpace, mathDomain);
		MemoryCube _cacheReshape(cacheReshape, nRows, nCubes, nCols, memorySpace, mathDomain);
		MemoryBuffer _cacheOnes(cacheOnes, nCubes, memorySpace, mathDomain);
		return _CubeWiseSum(_A, _T, _cacheReshape, _cacheOnes);
	}

	/**
	* X such that A * X = b by means of LU factorization
	*/
	EXPORT int _Solve(const MemoryTile& A, MemoryTile& B, const MatrixOperation aOperation, const LinearSystemSolverType solver)
	{
		const auto& handle = detail::CuSolverHandle();
		const auto& cublasHandle = detail::CublasHandle();

		int err;

		int *info = nullptr;

		switch (A.mathDomain)
		{
			case MathDomain::Float:
			{
				float *buffer = nullptr;

				// Need to copy A, as it will be overwritten by its factorization
				float *aPtr = nullptr;
				err = cudaMalloc(&aPtr, A.nRows * A.nRows * sizeof(float));
				if (err)
					return err;
				cudaMemcpy(aPtr, (float *) A.pointer, A.nRows * A.nRows * sizeof(float), cudaMemcpyDeviceToDevice);

				// calculate buffer size required by the solver
				int bufferSize = 0;
				switch (solver)
				{
					case LinearSystemSolverType::Lu:
						if (cusolverDnSgetrf_bufferSize(handle, A.nRows, A.nRows, aPtr, A.leadingDimension, &bufferSize))
							return CudaKernelException::_InternalException;
						break;
					case LinearSystemSolverType::Qr:
						if (cusolverDnSgeqrf_bufferSize(handle, A.nRows, A.nRows, aPtr, A.leadingDimension, &bufferSize))
							return CudaKernelException::_InternalException;
						break;
					default:
						return CudaKernelException::_NotImplementedException;
				}
				err = cudaMalloc(&buffer, bufferSize * sizeof(float));
				if (err)
					return err;

				// Initializes auxliary value for solver
				err = cudaMalloc(&info, sizeof(int));
				if (err)
					return err;
				err = cudaMemset(info, 0, sizeof(int));
				if (err)
					return err;

				// allocate memory for pivoting
				switch (solver)
				{
					case LinearSystemSolverType::Lu:
					{
						int *ipiv = nullptr;

						err = cudaMalloc(&ipiv, A.nRows * sizeof(int));
						if (err)
							return err;

						// Factorize A (and overwrite it with L)
						if (cusolverDnSgetrf(handle, A.nRows, A.nRows, aPtr, A.leadingDimension, buffer, ipiv, info))
							return CudaKernelException::_InternalException;

						// Solve
						err = cusolverDnSgetrs(handle, cublasOperation[static_cast<unsigned>(aOperation)], A.nRows, B.nCols, aPtr, A.leadingDimension, ipiv, (float *) B.pointer, B.leadingDimension, info);

						cudaFree(ipiv);

						break;
					}
					case LinearSystemSolverType::Qr:
					{
						float *tau = nullptr;
						err = cudaMalloc((void **) &tau, sizeof(float) * A.nRows);
						if (err)
							return err;

						// compute QR factorization
						if (cusolverDnSgeqrf(handle, A.nRows, A.nRows, aPtr, A.leadingDimension, tau, buffer, bufferSize, info))
							return CudaKernelException::_InternalException;

						// Q^T * B
						err = cusolverDnSormqr(
								handle,
								CUBLAS_SIDE_LEFT,
								CUBLAS_OP_T,
								A.nRows,
								B.nCols,
								A.nRows,
								aPtr,
								A.leadingDimension,
								tau,
								(float *) B.pointer,
								B.leadingDimension,
								buffer,
								bufferSize,
								info);
						if (err)
							return err;

						// Solve (x = R \ (Q^T * B))
						static constexpr float one = 1.0f;
						err = cublasStrsm(
								cublasHandle,
								CUBLAS_SIDE_LEFT,
								CUBLAS_FILL_MODE_UPPER,
								cublasOperation[static_cast<unsigned>(aOperation)],
								CUBLAS_DIAG_NON_UNIT,
								A.nRows,
								B.nCols,
								&one,
								aPtr,
								A.leadingDimension,
								(float *)B.pointer,
								B.leadingDimension);
						break;
					}
					default:
						return CudaKernelException::_NotImplementedException;
				}

				cudaFree(buffer);
				cudaFree(aPtr);
				break;
			}
			case MathDomain::Double:
			{
				double *buffer = nullptr;

				// Need to copy A, as it will be overwritten by its factorization
				double *aPtr = nullptr;
				err = cudaMalloc(&aPtr, A.nRows * A.nRows * sizeof(double));
				if (err)
					return err;
				cudaMemcpy(aPtr, (double *) A.pointer, A.nRows * A.nRows * sizeof(double), cudaMemcpyDeviceToDevice);

				// calculate buffer size required by the solver
				int bufferSize = 0;
				switch (solver)
				{
					case LinearSystemSolverType::Lu:
						if (cusolverDnDgetrf_bufferSize(handle, A.nRows, A.nRows, aPtr, A.leadingDimension, &bufferSize))
							return CudaKernelException::_InternalException;
						break;
					case LinearSystemSolverType::Qr:
						if (cusolverDnDgeqrf_bufferSize(handle, A.nRows, A.nRows, aPtr, A.leadingDimension, &bufferSize))
							return CudaKernelException::_InternalException;
						break;
					default:
						return CudaKernelException::_NotImplementedException;
				}

				err = cudaMalloc(&buffer, bufferSize * sizeof(double));
				if (err)
					return err;

				// Initializes auxliary value for solver
				err = cudaMalloc(&info, sizeof(int));
				if (err)
					return err;
				err = cudaMemset(info, 0, sizeof(int));
				if (err)
					return err;

				// allocate memory for pivoting
				switch (solver)
				{
					case LinearSystemSolverType::Lu:
					{
						int *ipiv = nullptr;

						err = cudaMalloc(&ipiv, A.nRows * sizeof(int));
						if (err)
							return err;

						// Factorize A (and overwrite it with L)
						if (cusolverDnDgetrf(handle, A.nRows, A.nRows, aPtr, A.leadingDimension, buffer, ipiv, info))
							return CudaKernelException::_InternalException;

						// Solve
						err = cusolverDnDgetrs(handle, cublasOperation[static_cast<unsigned>(aOperation)], A.nRows, B.nCols, aPtr, A.leadingDimension, ipiv, (double *) B.pointer, B.leadingDimension, info);
						break;
					}
					case LinearSystemSolverType::Qr:
					{
						double *tau = nullptr;
						err = cudaMalloc((void **) &tau, sizeof(double) * A.nRows);
						if (err)
							return err;

						// compute QR factorization
						if (cusolverDnDgeqrf(handle, A.nRows, A.nRows, aPtr, A.leadingDimension, tau, buffer, bufferSize, info))
							return CudaKernelException::_InternalException;

						// B = Q^T * B
						err = cusolverDnDormqr(
								handle,
								CUBLAS_SIDE_LEFT,
								CUBLAS_OP_T,
								A.nRows,
								B.nCols,
								A.nRows,
								aPtr,
								A.leadingDimension,
								tau,
								(double *) B.pointer,
								B.leadingDimension,
								buffer,
								bufferSize,
								info);
						if (err)
							return err;

						// Solve (x = R \ (Q^T * B))
						static constexpr double one = 1.0;
						err = cublasDtrsm(
								cublasHandle,
								CUBLAS_SIDE_LEFT,
								CUBLAS_FILL_MODE_UPPER,
								cublasOperation[static_cast<unsigned>(aOperation)],
								CUBLAS_DIAG_NON_UNIT,
								A.nRows,
								B.nCols,
								&one,
								aPtr,
								A.leadingDimension,
								(double *) B.pointer,
								B.leadingDimension);
						break;
					}
					default:
						return CudaKernelException::_NotImplementedException;
				}

				cudaFree(buffer);
				cudaFree(aPtr);
				break;
			}
			default:
				return CudaKernelException::_NotImplementedException;
		}

		cudaDeviceSynchronize();

		// free memory
		cudaFree(info);

		return err;
	}
	EXPORT int _SolveRaw(const ptr_t A, const ptr_t B, const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain, const MatrixOperation aOperation)
	{
		MemoryTile _A(A, nRows, nCols, memorySpace, mathDomain);
		MemoryTile _B(B, nRows, nCols, memorySpace, mathDomain);
		return _Solve(_A, _B, aOperation);
	}

	/**
	* A = A^(-1) by means of LU factorization
	*/
	EXPORT int _Invert(MemoryTile& A, const MatrixOperation aOperation)
	{
		float* eyePtr = nullptr;
		int err = cudaMalloc(&eyePtr, A.TotalSize());
		if (err)
			return err;
		MemoryTile eye((ptr_t)eyePtr, A.nRows, A.nRows, A.memorySpace, A.mathDomain);
		err = _Eye(eye);
		if (err)
			return err;
		err = _Solve(A, eye, aOperation);
		if (err)
			return err;

		// This might not be the fastest implementation, but it's general enough
		switch (A.mathDomain)
		{
		case MathDomain::Float:
		{
			err = cudaMemcpy((float*)A.pointer, (float*)eye.pointer, A.TotalSize(), cudaMemcpyDefault);
			if (err)
				return err;
			break;
		}
		case MathDomain::Double:
		{
			err = cudaMemcpy((double*)A.pointer, (double*)eye.pointer, A.TotalSize(), cudaMemcpyDefault);
			if (err)
				return err;
			break;
		}
		default:
			return CudaKernelException::_NotImplementedException;
		}

		cudaFree((void*)eye.pointer);
		return err;
	}
	EXPORT int _InvertRaw(const ptr_t A, const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain, const MatrixOperation aOperation)
	{
		MemoryTile _A(A, nRows, nCols, memorySpace, mathDomain);
		return _Invert(_A, aOperation);
	}

	EXPORT int _ArgAbsMin(int& argMin, const MemoryBuffer& x)
	{
		const cublasHandle_t& handle = detail::CublasHandle();
		int err = 0;
		switch (x.mathDomain)
		{
			case MathDomain::Float:
				err = cublasIsamin(handle, x.size, (float*)x.pointer, 1, &argMin);
				break;
			case MathDomain::Double:
				err = cublasIdamin(handle, x.size, (double*)x.pointer, 1, &argMin);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}

		if (err)
			return err;

		// cublasI<t>amin uses 1-indexed array
		--argMin;
		return cudaGetLastError();
	}

	EXPORT int _ColumnWiseArgAbsMin(MemoryBuffer& argMin, const MemoryTile& A)
	{
		const cublasHandle_t& handle = detail::CublasHandle();
		cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

		int err = 0;
		switch (A.mathDomain)
		{
			case MathDomain::Float:
				for (size_t j = 0; j < A.nCols; ++j)
					err = cublasIsamin(handle, A.nRows, (float*)(A.pointer + j * A.nRows * A.ElementarySize()), 1, (int*)(argMin.pointer + j * argMin.ElementarySize()));
				break;
			case MathDomain::Double:
				for (size_t j = 0; j < A.nCols; ++j)
					err = cublasIdamin(handle, A.nRows, (double*)(A.pointer + j * A.nRows * A.ElementarySize()), 1, (int*)(argMin.pointer + j * argMin.ElementarySize()));
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}

		cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

		if (err)
			return err;

		return cudaGetLastError();
	}

	EXPORT int _ArgAbsMax(int& argMax, const MemoryBuffer& x)
	{
		const cublasHandle_t& handle = detail::CublasHandle();
		int err = 0;
		switch (x.mathDomain)
		{
			case MathDomain::Float:
				err = cublasIsamax(handle, x.size, (float*)x.pointer, 1, &argMax);
				break;
			case MathDomain::Double:
				err = cublasIdamax(handle, x.size, (double*)x.pointer, 1, &argMax);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}

		if (err)
			return err;

		// cublasI<t>amax uses 1-indexed array
		--argMax;
		return cudaGetLastError();
	}

	EXPORT int _ColumnWiseArgAbsMax(MemoryBuffer& argMax, const MemoryTile& A)
	{
		const cublasHandle_t& handle = detail::CublasHandle();
		cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

		int err = 0;
		switch (A.mathDomain)
		{
			case MathDomain::Float:
				for (size_t j = 0; j < A.nCols; ++j)
					err = cublasIsamax(handle, A.nRows, (float*)(A.pointer + j * A.nRows * A.ElementarySize()), 1, (int*)(argMax.pointer + j * argMax.ElementarySize()));
				break;
			case MathDomain::Double:
				for (size_t j = 0; j < A.nCols; ++j)
					err = cublasIdamax(handle, A.nRows, (double*)(A.pointer + j * A.nRows * A.ElementarySize()), 1, (int*)(argMax.pointer + j * argMax.ElementarySize()));
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}

		cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

		if (err)
			return err;

		return cudaGetLastError();
	}

	EXPORT int _IsNonZero(MemoryBuffer& z, const MemoryBuffer& x)
	{
		switch (z.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_SINGLE(__IsNonZero__<float>, (float*)z.pointer, (float*)x.pointer, z.size);
				break;
			case MathDomain::Double:
				CUDA_CALL_DOUBLE(__IsNonZero__<double>, (double*)z.pointer, (double*)x.pointer, z.size);
				break;
			case MathDomain::Int:
				CUDA_CALL_SINGLE(__IsNonZero__<int>, (int*)z.pointer, (int*)x.pointer, z.size);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}

		return cudaGetLastError();
	}

	EXPORT int _EuclideanNorm(double& norm, const MemoryBuffer& x)
	{
		const cublasHandle_t& handle = detail::CublasHandle();

		switch (x.mathDomain)
		{
			case MathDomain::Float:
			{
				auto _norm = (float)norm;
				int err = cublasSnrm2(handle, x.size, (float*)x.pointer, 1, &_norm);

				norm = _norm;
				return err;
			}
			case MathDomain::Double:
				return cublasDnrm2(handle, x.size, (double*)x.pointer, 1, &norm);
			case MathDomain::Int:
			default:
				return CudaKernelException::_NotImplementedException;
		}
	}
}

GLOBAL void __IntAffineOperationNaive__(int* z, const int* x, const int* y, const size_t sz, const int alpha, const int beta, const int gamma)
{
	CUDA_FUNCTION_PROLOGUE
	CUDA_FOR_LOOP_PROLOGUE

		z[i] = alpha * x[i] + beta * y[i] + gamma * z[i];

	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T, bool inverse>
GLOBAL void __ElementwiseProductNaive__(T* RESTRICT z, const T* RESTRICT x, const T* RESTRICT y, const size_t sz, const T alpha)
{
	CUDA_FUNCTION_PROLOGUE
	CUDA_FOR_LOOP_PROLOGUE

		z[i] = x[i] * (!inverse ? y[i] : (static_cast<T>(1.0) / y[i])) * alpha;

	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __IsNonZero__(T* RESTRICT z, const T* RESTRICT x, const size_t sz)
{
	CUDA_FUNCTION_PROLOGUE
	CUDA_FOR_LOOP_PROLOGUE

		z[i] = (x[i] > static_cast<T>(1e-12) || x[i] < static_cast<T>(-1e-12)) ? 1 : 0;

	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __Reshape__(T* RESTRICT out, const T* RESTRICT in, const size_t nRows, const size_t nCols, const size_t nCubes)
{
	const int tidX = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int stepX = gridDim.x * blockDim.x;

	const int tidY = blockDim.y * blockIdx.y + threadIdx.y;
	const unsigned int stepY = gridDim.y * blockDim.y;

	const size_t inMatrixSize = nRows * nCols;
	const size_t outMatrixSize = nRows * nCubes;
	for (size_t i = tidX; i < nRows; i += stepX)
	{
		for (size_t j = tidY; j < nCols; j += stepY)
		{
			const size_t inStride = i + j * nRows;
			const size_t outOffset = i + j * outMatrixSize;
			for (size_t k = 0; k < nCubes; ++k)
				out[outOffset + k * nRows] = in[inStride + k * inMatrixSize];
		}
	}
}
