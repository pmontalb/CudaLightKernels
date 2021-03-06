#include "DeviceManager.cuh"
#include <stdio.h>

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct
	{
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] =
	{
		{ 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192 }, // Kepler Generation (SM 3.0) GK10x class
		{ 0x32, 192 }, // Kepler Generation (SM 3.2) GK10x class
		{ 0x35, 192 }, // Kepler Generation (SM 3.5) GK11x class
		{ 0x37, 192 }, // Kepler Generation (SM 3.7) GK21x class
		{ 0x50, 128 }, // Maxwell Generation (SM 5.0) GM10x class
		{ 0x52, 128 }, // Maxwell Generation (SM 5.2) GM20x class
		{ 0x53, 128 }, // Maxwell Generation (SM 5.3) GM20x class
		{ 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
		{ 0x61, 128 }, // Pascal Generation (SM 6.1) GP10x class
		{ 0x62, 128 }, // Pascal Generation (SM 6.2) GP10x class
		{ -1, -1 }
	};

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1)
	{
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
		{
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one to run properly
	printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index - 1].Cores);
	return nGpuArchCoresPerSM[index - 1].Cores;
}

EXTERN_C
{
	EXPORT int _ThreadSynchronize()
	{
		return cudaDeviceSynchronize();
	}

	EXPORT int _SetDevice(const int dev)
	{
		detail::currentDevice = dev;
		return cudaSetDevice(dev);
	}

	EXPORT int _GetDevice(int& dev)
	{
		return cudaGetDevice(&dev);
	}

	EXPORT int _GetDeviceCount(int& count)
	{
		return cudaGetDeviceCount(&count);
	}

	EXPORT int _GetDeviceProperties(cudaDeviceProp& ret, const int dev)
	{
		return cudaGetDeviceProperties(&ret, dev);
	}

	EXPORT int _GetDeviceStatus()
	{
		return cudaGetLastError();
	}

	EXPORT int _GetBestDevice(int& dev)
	{
		#ifndef MAX
		#define MAX(a,b) (a > b ? a : b)
		#endif

		int current_device = 0, sm_per_multiproc = 0;
		int max_perf_device = 0;
		int device_count = 0, best_SM_arch = 0;
		int devices_prohibited = 0;

		unsigned long long max_compute_perf = 0;
		cudaDeviceProp deviceProp;
		cudaGetDeviceCount(&device_count);

		cudaGetDeviceCount(&device_count);

		if (device_count == 0)
		{
			fprintf(stderr, "gpuGetMaxGflopsDeviceId() CUDA error: no devices supporting CUDA.\n");
			exit(EXIT_FAILURE);
		}

		// Find the best major SM Architecture GPU device
		while (current_device < device_count)
		{
			cudaGetDeviceProperties(&deviceProp, current_device);

			// If this GPU is not running on Compute Mode prohibited, then we can add it to the list
			if (deviceProp.computeMode != cudaComputeModeProhibited)
			{
				if (deviceProp.major > 0 && deviceProp.major < 9999)
				{
					best_SM_arch = MAX(best_SM_arch, deviceProp.major);
				}
			}
			else
			{
				devices_prohibited++;
			}

			current_device++;
		}

		if (devices_prohibited == device_count)
		{
			fprintf(stderr, "gpuGetMaxGflopsDeviceId() CUDA error: all devices have compute mode prohibited.\n");
			exit(EXIT_FAILURE);
		}

		// Find the best CUDA capable GPU device
		current_device = 0;

		while (current_device < device_count)
		{
			cudaGetDeviceProperties(&deviceProp, current_device);

			// If this GPU is not running on Compute Mode prohibited, then we can add it to the list
			if (deviceProp.computeMode != cudaComputeModeProhibited)
			{
				if (deviceProp.major == 9999 && deviceProp.minor == 9999)
				{
					sm_per_multiproc = 1;
				}
				else
				{
					sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
				}

				unsigned long long compute_perf = (unsigned long long) deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;

				if (compute_perf > max_compute_perf)
				{
					// If we find GPU with SM major > 2, search only these
					if (best_SM_arch > 2)
					{
						// If our device==dest_SM_arch, choose this, or else pass
						if (deviceProp.major == best_SM_arch)
						{
							max_compute_perf = compute_perf;
							max_perf_device = current_device;
						}
					}
					else
					{
						max_compute_perf = compute_perf;
						max_perf_device = current_device;
					}
				}
			}

			++current_device;
		}

		dev = max_perf_device;

		return cudaGetLastError();
	}
}

namespace detail
{
	void GetBestDimension(dim3& block, dim3& grid, const unsigned nBlocks, const unsigned problemDimension)
	{
		// Get device properties
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, detail::currentDevice);

		// Determine how to divide the work between cores
		block.x = nBlocks;
		grid.x = (problemDimension + nBlocks - 1) / nBlocks;

		// Aim to launch around ten or more times as many blocks as there
		// are multiprocessors on the target device.
		const unsigned blocksPerSM = 10;
		const int& numSMs = deviceProperties.multiProcessorCount;

		while (grid.x > 2 * blocksPerSM * numSMs)
			grid.x >>= 1;
	}

	const cublasHandle_t& CublasHandle()
	{
		static bool hasBeenInitialised = false;
		static cublasHandle_t cublasHandle[6];
		if (!hasBeenInitialised)
		{
			for (int i = 0; i < 6; ++i)
				cublasCreate(&cublasHandle[i]);
			hasBeenInitialised = true;
		}
		return cublasHandle[currentDevice];
	}

	const cusolverDnHandle_t& CuSolverHandle()
	{
		static bool hasBeenInitialised = false;
		static cusolverDnHandle_t cusolverHandle[6];
		if (!hasBeenInitialised)
		{
			for (int i = 0; i < 6; ++i)
				cusolverDnCreate(&cusolverHandle[i]);
			hasBeenInitialised = true;
		}
		return cusolverHandle[currentDevice];
	}

	const cusparseHandle_t& CuSparseHandle()
	{
		static bool hasBeenInitialised = false;
		static cusparseHandle_t cusparseHandle[6];
		if (!hasBeenInitialised)
		{
			for (int i = 0; i < 6; ++i)
				cusparseCreate(&cusparseHandle[i]);
			hasBeenInitialised = true;
		}
		return cusparseHandle[currentDevice];
	}

	const cusparseMatDescr_t& CsrMatrixDescription()
	{
		static bool hasBeenInitialised = false;
		static cusparseMatDescr_t descr;
		if (!hasBeenInitialised)
		{
			cusparseCreateMatDescr(&descr);
			cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
			cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
			hasBeenInitialised = true;
		}
		return descr;
	}

	const cusolverSpHandle_t& CuSolverSparseHandle()
	{
		static bool hasBeenInitialised = false;
		static cusolverSpHandle_t cusparseSpHandle[6];
		if (!hasBeenInitialised)
		{
			for (int i = 0; i < 6; ++i)
				cusolverSpCreate(&cusparseSpHandle[i]);
			hasBeenInitialised = true;
		}
		return cusparseSpHandle[currentDevice];
	}
}