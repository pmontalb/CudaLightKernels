#include "CuBlasWrappers.cuh"
#include "DeviceManager.cuh"
#include "BufferInitializer.cuh"
#include <cublas.h>

EXTERN_C
{
	namespace clk
	{
		/**
		* z = x + y
		*/
		EXPORT int _Add(MemoryBuffer z, const MemoryBuffer x, const MemoryBuffer y, const double alpha)
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
			default:
				return -1;
			}
		}

		/**
		* z += x
		*/
		EXPORT int _AddEqual(MemoryBuffer z, const MemoryBuffer x, const double alpha)
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
			default:
				return -1;
			}
		}

		/**
		* A += alpha * B
		*/
		EXPORT int _AddEqualMatrix(MemoryTile A, const MemoryTile B, const MatrixOperation aOperation, const MatrixOperation bOperation,const double alpha)
		{
			const cublasHandle_t& handle = detail::CublasHandle();

			switch (A.mathDomain)
			{
			case MathDomain::Float:
			{
				const float _alpha = 1.0f;
				const float beta = (float)alpha;
				return cublasSgeam(handle, cublasOperation[aOperation], cublasOperation[bOperation],
					A.nRows, A.nCols,
					&_alpha,
					(float*)A.pointer, A.nRows,
					&beta,
					(float*)B.pointer, A.nRows,
					(float*)A.pointer, A.nRows);
			}
			case MathDomain::Double:
			{
				const double _alpha = 1.0;
				return cublasDgeam(handle, cublasOperation[aOperation], cublasOperation[bOperation],
					A.nRows, A.nCols,
					&_alpha,
					(double*)A.pointer, A.nRows,
					&alpha,
					(double*)B.pointer, A.nRows,
					(double*)A.pointer, A.nRows);
			}
			default:
				return -1;
			}
		}

		/**
		* z *= alpha
		*/
		EXPORT int _Scale(MemoryBuffer z, const double alpha)
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
			default:
				return -1;
			}
		}

		EXPORT int _ElementwiseProduct(MemoryBuffer z, const MemoryBuffer x, const MemoryBuffer y, const double alpha)
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
				const float beta = 0.0f;

				return cublasSsbmv(handle, CUBLAS_FILL_MODE_UPPER,
					z.size,
					0,  // Just the diagonal; 0 super-diagonal bands
					&_alpha,
					(float*)z.pointer, 1,
					(float*)x.pointer, 1,
					&beta,
					(float*)z.pointer, 1);
			}
			case MathDomain::Double:
			{
				int err = cublasDcopy(handle, z.size, (double*)y.pointer, 1, (double*)z.pointer, 1);
				if (err)
					return err;

				const double beta = 0.0;
				return cublasDsbmv(handle, CUBLAS_FILL_MODE_UPPER,
					z.size,
					0,  // Just the diagonal; 0 super-diagonal bands
					&alpha,
					(double*)z.pointer, 1,
					(double*)x.pointer, 1,
					&beta,
					(double*)z.pointer, 1);
			}
			default:
				return -1;
			}



		}

		EXPORT int _Multiply(MemoryTile A, const MemoryTile B, const MemoryTile C, const unsigned leadingDimensionB, const unsigned leadingDimensionC, const MatrixOperation bOperation, const MatrixOperation cOperation, const double alpha)
		{
			const cublasHandle_t& handle = detail::CublasHandle();
			switch (A.mathDomain)
			{
			case MathDomain::Float:
			{
				const float _alpha = (float)alpha;
				const float beta = 0.0f;
				return cublasSgemm(handle, cublasOperation[bOperation], cublasOperation[cOperation],
					leadingDimensionB, C.nCols, leadingDimensionC,
					&_alpha,
					(float*)B.pointer, leadingDimensionB,
					(float*)C.pointer, leadingDimensionC,
					&beta,
					(float*)A.pointer, leadingDimensionB);
			}
			case MathDomain::Double:
			{
				const double beta = 0.0;
				return cublasDgemm(handle, cublasOperation[bOperation], cublasOperation[cOperation],
					leadingDimensionB, C.nCols, leadingDimensionC,
						&alpha,
						(double*)B.pointer, leadingDimensionB,
						(double*)C.pointer, leadingDimensionC,
						&beta,
						(double*)A.pointer, leadingDimensionB);
			}
			default:
				return -1;
			}
		}

		EXPORT int _Dot(MemoryBuffer y, const MemoryTile A, const MemoryBuffer x, const MatrixOperation aOperation, const double alpha)
		{
			const cublasHandle_t& handle = detail::CublasHandle();
			switch (A.mathDomain)
			{
			case MathDomain::Float:
			{
				const float _alpha = (float)alpha;
				const float beta = 0.0f;
				return cublasSgemv(handle, cublasOperation[aOperation],
					A.nRows, A.nCols,
					&_alpha,
					(float*)A.pointer, A.nRows,
					(float*)x.pointer, 1,
					&beta,
					(float*)y.pointer, 1);
			}
			case MathDomain::Double:
			{
				const double beta = 0.0;
				return cublasDgemv(handle, cublasOperation[aOperation],
					A.nRows, A.nCols,
					&alpha,
					(double*)A.pointer, A.nRows,
					(double*)x.pointer, 1,
					&beta,
					(double*)y.pointer, 1);
			}
			default:
				return -1;
			}
		}

		EXPORT int _CumulativeRowSum(MemoryTile A)
		{
			const cublasHandle_t& handle = detail::CublasHandle();
			int err = -1;

			switch(A.mathDomain)
			{
			case MathDomain::Float: 
			{
				float *onesPtr = nullptr;
				if (cudaMalloc((void **)&onesPtr, A.nRows * A.nCols * sizeof(float)))
					return -1;
				MemoryTile ones((ptr_t)onesPtr, A.nRows, A.nCols, A.memorySpace, A.mathDomain);
				_OnesUpperTriangular(ones);

				float *buffer = nullptr;
				if (cudaMalloc((void **)&buffer, A.nRows * A.nCols * sizeof(float)))
					return -1;

				cudaMemcpy(buffer, (void*)A.pointer, A.nRows * A.nCols * sizeof(float), cudaMemcpyDeviceToDevice);

				float alpha = 1.0f, beta = 0.0f;
				err = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
					A.nRows, A.nCols, A.nRows,
					&alpha,
					buffer, A.nRows,
					(float*)ones.pointer, A.nRows,
					&beta,
					(float*)A.pointer, A.nRows);

				cudaFree((void*)ones.pointer);
				cudaFree(buffer);
				break;
			}
			case MathDomain::Double: 
			{
				double *onesPtr = nullptr;
				if (cudaMalloc((void **)&onesPtr, A.nRows * A.nCols * sizeof(double)))
					return -1;
				MemoryTile ones((ptr_t)onesPtr, A.nRows, A.nCols, A.memorySpace, A.mathDomain);
				_OnesUpperTriangular(ones);

				double *buffer = nullptr;
				if (cudaMalloc((void **)&buffer, A.nRows * A.nCols * sizeof(double)))
					return -1;

				cudaMemcpy(buffer, (void*)A.pointer, A.nRows * A.nCols * sizeof(double), cudaMemcpyDeviceToDevice);

				double alpha = 1.0, beta = 0.0;
				err = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
					A.nRows, A.nCols, A.nRows,
					&alpha,
					buffer, A.nRows,
					(double*)ones.pointer, A.nRows,
					&beta,
					(double*)A.pointer, A.nRows);

				cudaFree((void*)ones.pointer);
				cudaFree(buffer);
				break;
			}
			default: 
				return -1;
			}

			return err;
		}

		/**
		* X such that A * X = b by means of LU factorization
		*/
		EXPORT int _Solve(const MemoryTile A, MemoryTile B, const MatrixOperation aOperation)
		{
			const cusolverDnHandle_t& handle = detail::CuSolverHandle();
			int err = -1;
			switch (A.mathDomain)
			{
			case MathDomain::Float:
			{
				// Need to copy A, as it will be overwritten by its factorization
				float *aPtr = nullptr;
				if (cudaMalloc(&aPtr, A.nRows * A.nRows * sizeof(float)))
					return -1;
				cudaMemcpy(aPtr, (float*)A.pointer, A.nRows * A.nRows * sizeof(float), cudaMemcpyDeviceToDevice);

				// calculate buffer size required by the solver
				int bufferSize = 0;
				if (cusolverDnSgetrf_bufferSize(handle, A.nRows, A.nRows, aPtr, A.nRows, &bufferSize))
					return -1;
				float *buffer = nullptr;
				if (cudaMalloc(&buffer, bufferSize * sizeof(float)))
					return -1;

				// allocate memory for pivoting
				int *ipiv = nullptr;
				if (cudaMalloc(&ipiv, A.nRows * sizeof(int)))
					return -1;

				// Initializes auxliary value for solver
				int *info = nullptr;
				if (cudaMalloc(&info, sizeof(int)))
					return -1;
				if (cudaMemset(info, 0, sizeof(int)))
					return -1;

				// Factorize A (and overwrite it with L)
				if (cusolverDnSgetrf(handle, A.nRows, A.nRows, aPtr, A.nRows, buffer, ipiv, info))
					return -1;
				// Solve
				err = cusolverDnSgetrs(handle, cublasOperation[aOperation], A.nRows, B.nCols, aPtr, A.nRows, ipiv, (float*)B.pointer, A.nRows, info);
				cudaDeviceSynchronize();

				// free memory
				cudaFree(info);
				cudaFree(buffer);
				cudaFree(ipiv);
				break;
			}
			case MathDomain::Double:
			{
				// Need to copy A, as it will be overwritten by its factorization
				double *aPtr = nullptr;
				if (cudaMalloc(&aPtr, A.nRows * A.nRows * sizeof(double)))
					return -1;
				cudaMemcpy(aPtr, (float*)A.pointer, A.nRows * A.nRows * sizeof(double), cudaMemcpyDeviceToDevice);

				// calculate buffer size required by the solver
				int bufferSize = 0;
				if (cusolverDnDgetrf_bufferSize(handle, A.nRows, A.nRows, aPtr, A.nRows, &bufferSize))
					return -1;
				double *buffer = nullptr;
				if (cudaMalloc(&buffer, bufferSize * sizeof(double)))
					return -1;

				// allocate memory for pivoting
				int *ipiv = nullptr;
				if (cudaMalloc(&ipiv, A.nRows * sizeof(int)))
					return -1;

				// Initializes auxliary value for solver
				int *info = nullptr;
				if (cudaMalloc(&info, sizeof(int)))
					return -1;
				if (cudaMemset(info, 0, sizeof(int)))
					return -1;

				// Factorize A (and overwrite it with L)
				if (cusolverDnDgetrf(handle, A.nRows, A.nRows, aPtr, A.nRows, buffer, ipiv, info))
					return -1;
				// Solve
				err = cusolverDnDgetrs(handle, CUBLAS_OP_N, A.nRows, B.nCols, aPtr, A.nRows, ipiv, (double*)B.pointer, A.nRows, info);
				cudaDeviceSynchronize();

				// free memory
				cudaFree(info);
				cudaFree(buffer);
				cudaFree(ipiv);
				break;
			};
			default: 
				return -1;
			}

			return err;
		}

		/**
		* A = A^(-1) by means of Cholesky
		*/
		EXPORT int _Invert(MemoryTile A, const MatrixOperation aOperation)
		{
			float* eyePtr = nullptr;
			if (cudaMalloc(&eyePtr, A.TotalSize()))
				return -1;
			MemoryTile eye((ptr_t)eyePtr, A.nRows, A.nRows, A.memorySpace, A.mathDomain);
			if (_Eye(eye))
				return -1;
			int err = _Solve(A, eye, aOperation);

			// This might not be the fastest implementation, but it's general enough
			switch (A.mathDomain)
			{
			case MathDomain::Float:
			{
				int err2 = cudaMemcpy((float*)A.pointer, (float*)eye.pointer, A.TotalSize(), cudaMemcpyDefault);
				if (err2)
					return err2;
				break;
			}
			case MathDomain::Double:
			{
				int err2 = cudaMemcpy((double*)A.pointer, (double*)eye.pointer, A.TotalSize(), cudaMemcpyDefault);
				if (err2)
					return err2;
				break;
			}
			default:
				return -1;
			}

			cudaFree((void*)eye.pointer);
			return err;
		}
	}
}