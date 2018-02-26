#pragma once

#include "Common.cuh"
#include "Flags.cuh"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse.h>

namespace detail
{
	static int currentDevice;
}

EXTERN_C
{
	EXPORT int _GetDevice(int& dev);

	EXPORT int _GetDeviceCount(int& count);

	EXPORT int _ThreadSynchronize();

	EXPORT int _SetDevice(const int dev);

	EXPORT int _GetDeviceStatus();

	EXPORT int _GetBestDevice(int& dev);

	EXPORT int _GetDeviceProperties(cudaDeviceProp& prop, const int dev);
}

namespace detail
{
	const cublasHandle_t& CublasHandle();
	const cusolverDnHandle_t& CuSolverHandle();
	const cusparseHandle_t& CuSparseHandle();
	const cusparseMatDescr_t& CsrMatrixDescription();

	void GetBestDimension(dim3& block, dim3& grid, const unsigned nBlocks, const unsigned problemDimension);
}
