#include "BufferInitializer.cuh"
#include "DeviceManager.cuh"

EXTERN_C
{
	EXPORT int _Zero(MemoryBuffer& buf)
	{
		cudaMemset((void*)(buf.pointer), 0, buf.TotalSize());
		return cudaGetLastError();
	}
	EXPORT int _ZeroRaw(const ptr_t pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer buf(pointer, size, memorySpace, mathDomain);
		return _Zero(buf);
	}

	EXPORT int _Initialize(MemoryBuffer& buf, const double value)
	{
		switch (buf.mathDomain)
		{
		case MathDomain::Float:
			CUDA_CALL_SINGLE(__Initialize__<float>, (float*)buf.pointer, buf.size, (float)value);
			break;
		case MathDomain::Double:
			CUDA_CALL_DOUBLE(__Initialize__<double>, (double*)buf.pointer, buf.size, value);
			break;
		case MathDomain::Int:
			CUDA_CALL_SINGLE(__Initialize__<int>, (int*)buf.pointer, buf.size, (int)value);
			break;
		default:
			return CudaKernelException::_NotImplementedException;
		}

		return cudaGetLastError();
	}
	EXPORT int _InitializeRaw(const ptr_t pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const double value)
	{
		MemoryBuffer buf(pointer, size, memorySpace, mathDomain);
		return _Initialize(buf, value);
    }

	EXPORT int _Reciprocal(MemoryBuffer& buf)
{
	switch (buf.mathDomain)
	{
		case MathDomain::Float:
			CUDA_CALL_SINGLE(__Reciprocal__<float>, (float*)buf.pointer, buf.size);
			break;
		case MathDomain::Double:
			CUDA_CALL_DOUBLE(__Reciprocal__<double>, (double*)buf.pointer, buf.size);
			break;
		case MathDomain::Int:
			CUDA_CALL_SINGLE(__Reciprocal__<int>, (int*)buf.pointer, buf.size);
			break;
		default:
			return CudaKernelException::_NotImplementedException;
	}
	
	return cudaGetLastError();
}
	EXPORT int _ReciprocalRaw(const ptr_t pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
{
	MemoryBuffer buf(pointer, size, memorySpace, mathDomain);
	return _Reciprocal(buf);
}

	EXPORT int _LinSpace(MemoryBuffer& buf, const double x0, const double x1)
	{
		const double dx = (x1 - x0) / (buf.size - 1);

		switch (buf.mathDomain)
		{
		case MathDomain::Float:
			CUDA_CALL_SINGLE(__LinSpace__<float>, (float*)buf.pointer, buf.size, (float)x0, (float)dx);
			break;
		case MathDomain::Double:
			CUDA_CALL_DOUBLE(__LinSpace__<double>, (double*)buf.pointer, buf.size, (double)x0, (double)dx);
			break;
		case MathDomain::Int:
			CUDA_CALL_SINGLE(__LinSpace__<int>, (int*)buf.pointer, buf.size, (int)x0, (int)dx);
			break;
		default:
			return CudaKernelException::_NotImplementedException;
		}

		return cudaGetLastError();
	}
	EXPORT int _LinSpaceRaw(const ptr_t pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const double x0, const double x1)
	{
		MemoryBuffer buf(pointer, size, memorySpace, mathDomain);
		return _LinSpace(buf, x0, x1);
	}

	EXPORT int _RandUniform(MemoryBuffer& buf, const unsigned seed)
	{
		dim3 block, grid;
		const unsigned halfSz = (buf.size + 1) >> 1;
		detail::GetBestDimension(block, grid, N_BLOCKS_SINGLE, halfSz);

		curandState *states = 0;
		int err = cudaMalloc((void **)&states, grid.x * block.x * sizeof(curandState));
		if (err)
			return err;
		CUDA_CALL_XY(__SetupCuRand__, grid, block, states, halfSz, seed);

		switch (buf.mathDomain)
		{
		case MathDomain::Float:
			CUDA_CALL_XYZ(__RandUniform__<float>, grid, block, block.x * sizeof(unsigned int), (float*)buf.pointer, states, halfSz, buf.size);
			break;
		case MathDomain::Double:
			CUDA_CALL_XYZ(__RandUniform__<double>, grid, block, block.x * sizeof(unsigned int), (double*)buf.pointer, states, halfSz, buf.size);
			break;
		case MathDomain::Int:
			CUDA_CALL_XYZ(__RandUniform__<int>, grid, block, block.x * sizeof(unsigned int), (int*)buf.pointer, states, halfSz, buf.size);
			break;
		default:
			return CudaKernelException::_NotImplementedException;
		}

		cudaFree(states);

		return cudaGetLastError();
	}
	EXPORT int _RandUniformRaw(const ptr_t pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const unsigned seed)
	{
		MemoryBuffer buf(pointer, size, memorySpace, mathDomain);
		return _RandUniform(buf,seed);
	}

	EXPORT int _RandNormal(MemoryBuffer& buf, const unsigned seed)
	{
		dim3 block, grid;
		const unsigned halfSz = buf.size >> 1;
		detail::GetBestDimension(block, grid, N_BLOCKS_SINGLE, halfSz);

		curandState *states = 0;
		int err = cudaMalloc((void **)&states, grid.x * block.x * sizeof(curandState));
		if (err)
			return err;
		CUDA_CALL_XY(__SetupCuRand__, grid, block, states, halfSz, seed);

		switch (buf.mathDomain)
		{
		case MathDomain::Float:
			CUDA_CALL_XYZ(__RandNormal__<float>, grid, block, block.x * sizeof(unsigned int), (float*)buf.pointer, states, halfSz, buf.size);
			break;
		case MathDomain::Double:
			CUDA_CALL_XYZ(__RandNormal__<double>, grid, block, block.x * sizeof(unsigned int), (double*)buf.pointer, states, halfSz, buf.size);
			break;
		case MathDomain::Int:
			CUDA_CALL_XYZ(__RandNormal__<int>, grid, block, block.x * sizeof(unsigned int), (int*)buf.pointer, states, halfSz, buf.size);
			break;
		default:
			return CudaKernelException::_NotImplementedException;
		}

		cudaFree(states);

		return cudaGetLastError();
	}
	EXPORT int _RandNormalRaw(const ptr_t pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const unsigned seed)
	{
		MemoryBuffer buf(pointer, size, memorySpace, mathDomain);
		return _RandNormal(buf, seed);
	}

	EXPORT int _Eye(MemoryTile& buf)
	{
		dim3 blockDim(16, 16);
		dim3 gridDim((buf.nRows + 15) / 16, (buf.nRows + 15) / 16);

		switch (buf.mathDomain)
		{
		case MathDomain::Float:
			CUDA_CALL_XY(__Eye__<float>, gridDim, blockDim, (float*)buf.pointer, buf.nRows);
			break;
		case MathDomain::Double:
			CUDA_CALL_XY(__Eye__<double>, gridDim, blockDim, (double*)buf.pointer, buf.nRows);
			break;
		default:
			return CudaKernelException::_NotImplementedException;
		}
			
		return cudaGetLastError();
	}
	EXPORT int _EyeRaw(const ptr_t pointer, const unsigned nRows, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryTile tile(pointer, nRows, nRows, memorySpace, mathDomain);
		return _Eye(tile);
	}

	EXPORT int _OnesUpperTriangular(MemoryTile& buf)
	{
		dim3 blockDim(16, 16);
		dim3 gridDim((buf.nRows + 15) / 16, (buf.nRows + 15) / 16);

		switch (buf.mathDomain)
		{
		case MathDomain::Float:
			CUDA_CALL_XY(__OnesUpperTriangular__<float>, gridDim, blockDim, (float*)buf.pointer, buf.nRows);
			break;
		case MathDomain::Double:
			CUDA_CALL_XY(__OnesUpperTriangular__<double>, gridDim, blockDim, (double*)buf.pointer, buf.nRows);
			break;
		default:
			return -1;
		}

		return cudaGetLastError();
	}
	EXPORT int _OnesUpperTriangularRaw(const ptr_t pointer, const unsigned nRows, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryTile tile(pointer, nRows, nRows, memorySpace, mathDomain);
		return _OnesUpperTriangular(tile);
	}

	EXPORT int _RandShuffle(MemoryBuffer& buf, const unsigned seed)
	{
		dim3 block, grid;
		detail::GetBestDimension(block, grid, N_BLOCKS_SINGLE, buf.size);
		
		curandState *states = 0;
		int err = cudaMalloc((void **)&states, grid.x * block.x * sizeof(curandState));
		if (err)
			return err;
		CUDA_CALL_XY(__SetupCuRand__, grid, block, states, buf.size, seed);
		
		// TODO: vectorize it!
		switch (buf.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_XY(__RandShuffle__<float>, 1, 1, (float*)buf.pointer, states, buf.size);
				break;
			case MathDomain::Double:
				CUDA_CALL_XY(__RandShuffle__<double>, 1, 1, (double*)buf.pointer, states, buf.size);
				break;
			case MathDomain::Int:
				CUDA_CALL_XY(__RandShuffle__<int>, 1, 1, (int*)buf.pointer, states, buf.size);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		
		cudaFree(states);
		
		return cudaGetLastError();
	}
	EXPORT int _RandShuffleRaw(const ptr_t pointer, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const unsigned seed)
	{
		MemoryBuffer buf(pointer, size, memorySpace, mathDomain);
		return _RandShuffle(buf, seed);
	}

	EXPORT int _RandShufflePair(MemoryBuffer& buf1, MemoryBuffer& buf2, const unsigned seed)
	{
		dim3 block, grid;
		detail::GetBestDimension(block, grid, N_BLOCKS_SINGLE, buf1.size);
		
		curandState *states = 0;
		int err = cudaMalloc((void **)&states, grid.x * block.x * sizeof(curandState));
		if (err)
			return err;
		CUDA_CALL_XY(__SetupCuRand__, grid, block, states, buf1.size, seed);
		
		// TODO: vectorize it!
		switch (buf1.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_XY(__RandShufflePair__<float>, 1, 1, (float*)buf1.pointer, (float*)buf2.pointer, states, buf1.size);
				break;
			case MathDomain::Double:
				CUDA_CALL_XY(__RandShufflePair__<double>, 1, 1, (double*)buf1.pointer, (double*)buf2.pointer, states, buf1.size);
				break;
			case MathDomain::Int:
				CUDA_CALL_XY(__RandShufflePair__<int>, 1, 1, (int*)buf1.pointer, (int*)buf2.pointer, states, buf1.size);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		
		cudaFree(states);
		
		return cudaGetLastError();
	}
	EXPORT int _RandShufflePairRaw(const ptr_t p1, const ptr_t p2, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain, const unsigned seed)
	{
		MemoryBuffer buf1(p1, size, memorySpace, mathDomain);
		MemoryBuffer buf2(p2, size, memorySpace, mathDomain);
		return _RandShufflePair(buf1, buf2, seed);
	}

	EXPORT int _RandShuffleColumns(MemoryTile& buf, const unsigned seed)
	{
		dim3 block, grid;
		detail::GetBestDimension(block, grid, N_BLOCKS_SINGLE, buf.nCols);
		
		curandState *states = 0;
		int err = cudaMalloc((void **)&states, grid.x * block.x * sizeof(curandState));
		if (err)
			return err;
		CUDA_CALL_XY(__SetupCuRand__, grid, block, states, buf.nCols, seed);
		
		// parallelized on rows, not on colums
		switch (buf.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_XY(__RandShuffleColumns__<float>, GRID_1D_X_SINGLE, BLOCK_1D_X_SINGLE, (float*)buf.pointer, states, buf.nRows, buf.nCols);
				break;
			case MathDomain::Double:
				CUDA_CALL_XY(__RandShuffleColumns__<double>, GRID_1D_X_DOUBLE, BLOCK_1D_X_DOUBLE, (double*)buf.pointer, states, buf.nRows, buf.nCols);
				break;
			case MathDomain::Int:
				CUDA_CALL_XY(__RandShuffleColumns__<int>, GRID_1D_X_SINGLE, BLOCK_1D_X_SINGLE, (int*)buf.pointer, states, buf.nRows, buf.nCols);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		
		cudaFree(states);
		
		return cudaGetLastError();
	}
	EXPORT int _RandShuffleColumnsRaw(const ptr_t pointer, const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain, const unsigned seed)
	{
		MemoryTile buf(pointer, nRows, nCols, memorySpace, mathDomain);
		return _RandShuffleColumns(buf, seed);
	}

	EXPORT int _RandShuffleColumnsPair(MemoryTile& buf1, MemoryTile& buf2, const unsigned seed)
	{
		assert(buf1.nCols == buf2.nCols);
		dim3 block, grid;
		detail::GetBestDimension(block, grid, N_BLOCKS_SINGLE, buf1.nCols);
		
		curandState *states = 0;
		int err = cudaMalloc((void **)&states, grid.x * block.x * sizeof(curandState));
		if (err)
			return err;
		CUDA_CALL_XY(__SetupCuRand__, grid, block, states, buf1.nCols, seed);
		
		// parallelized on rows, not on colums
		switch (buf1.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_XY(__RandShuffleColumnsPair__<float>, GRID_1D_X_SINGLE, BLOCK_1D_X_SINGLE, (float*)buf1.pointer, (float*)buf2.pointer, states, buf1.nRows, buf2.nRows, buf1.nCols);
				break;
			case MathDomain::Double:
				CUDA_CALL_XY(__RandShuffleColumnsPair__<double>, GRID_1D_X_DOUBLE, BLOCK_1D_X_DOUBLE, (double*)buf1.pointer, (double*)buf2.pointer, states, buf1.nRows, buf2.nRows, buf1.nCols);
				break;
			case MathDomain::Int:
				CUDA_CALL_XY(__RandShuffleColumnsPair__<int>, GRID_1D_X_SINGLE, BLOCK_1D_X_SINGLE, (int*)buf1.pointer, (int*)buf2.pointer, states, buf1.nRows, buf2.nRows, buf1.nCols);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		
		cudaFree(states);
		
		return cudaGetLastError();
	}
	EXPORT int _RandShuffleColumnsPairRaw(const ptr_t p1, const ptr_t p2, const unsigned nRows1, const unsigned nRows2, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain, const unsigned seed)
	{
		MemoryTile buf1(p1, nRows1, nCols, memorySpace, mathDomain);
		MemoryTile buf2(p2, nRows2, nCols, memorySpace, mathDomain);
		return _RandShuffleColumnsPair(buf1, buf2, seed);
	}
}

template <typename T>
GLOBAL void __Initialize__(T* RESTRICT ptr, const ptr_t sz, const T value)
{
	CUDA_FUNCTION_PROLOGUE
	CUDA_FOR_LOOP_PROLOGUE

		ptr[i] = value;

	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __Reciprocal__(T* RESTRICT ptr, const ptr_t sz)
{
	CUDA_FUNCTION_PROLOGUE
	CUDA_FOR_LOOP_PROLOGUE
		
		ptr[i] = static_cast<T>(1.0) / ptr[i];
	
	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __LinSpace__(T* RESTRICT ptr, const ptr_t sz, const T x0, const T dx)
{
	CUDA_FUNCTION_PROLOGUE
	CUDA_FOR_LOOP_PROLOGUE

		ptr[i] = x0 + i * dx;

	CUDA_FOR_LOOP_EPILOGUE
}

GLOBAL void __SetupCuRand__(CURAND_STATE_PTR states, const ptr_t sz, const unsigned seed)
{
	// Determine thread ID
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Initialise the RNG
	curand_init(seed, tid, 0, &states[tid]);
}

template <typename T>
GLOBAL void __RandUniform__(T* RESTRICT ptr, CURAND_STATE_PTR states, const unsigned sz, const unsigned fullSz)
{
	CUDA_FUNCTION_PROLOGUE

	curandState localState = states[tid];

	CUDA_FOR_LOOP_PROLOGUE

		ptr[2 * i] = static_cast<T>(curand_uniform(&localState));
	    if (2 * i + 1 < fullSz)
			ptr[2 * i + 1] = static_cast<T>(1.0) - ptr[2 * i];

	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __RandNormal__(T* RESTRICT ptr, CURAND_STATE_PTR states, const unsigned sz, const unsigned fullSz)
{
	CUDA_FUNCTION_PROLOGUE

	curandState localState = states[tid];

	CUDA_FOR_LOOP_PROLOGUE

		ptr[2 * i] = static_cast<T>(curand_normal(&localState));
	    if (2 * i + 1 < fullSz)
		    ptr[2 * i + 1] = -ptr[2 * i];

	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __Eye__(T* RESTRICT A, const size_t sz)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (j < sz && i < sz)
	{
		if (i == j)
			A[i + sz * j] = static_cast<T>(1.0);
		else
			A[i + sz * j] = static_cast<T>(0.0);
	}
}

template <typename T>
GLOBAL void __OnesUpperTriangular__(T* RESTRICT A, size_t sz)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (j < sz && i < sz)
	{
		if (i <= j)
			A[i + sz * j] = static_cast<T>(1.0);
		else
			A[i + sz * j] = static_cast<T>(0.0);
	}
}

template <typename T>
DEVICE void Shuffle(T* RESTRICT ptr, const size_t i, const size_t j)
{
	const T tmp = ptr[i];
	ptr[i] = ptr[j];
	ptr[j] = tmp;
}

template <typename T>
GLOBAL void __RandShuffle__(T* RESTRICT ptr, CURAND_STATE_PTR states, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE
	
	curandState localState = states[tid];
	
	CUDA_FOR_LOOP_PROLOGUE
		
		// generate rand numbers from 0 to i
		const int newIndex = static_cast<int>(curand_normal(&localState)) % (i + 1);
		Shuffle(ptr, i, newIndex);
	
	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __RandShufflePair__(T* RESTRICT p1, T* RESTRICT p2, CURAND_STATE_PTR states, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE
	
	curandState localState = states[tid];
	
	CUDA_FOR_LOOP_PROLOGUE
		
		// generate rand numbers from 0 to i
		const int newIndex = static_cast<int>(curand_normal(&localState)) % (i + 1);
		Shuffle(p1, i, newIndex);
		Shuffle(p2, i, newIndex);
	
	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __RandShuffleColumns__(T* RESTRICT ptr, CURAND_STATE_PTR states, const unsigned nRows, const unsigned nCols)
{
	unsigned int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stepX = gridDim.x * blockDim.x;
	
	unsigned int tidY = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int stepY = gridDim.y * blockDim.y;
	curandState localState = states[tidY];
 
	for (size_t j = tidY; j < nCols; j += stepY)
    {
	    // generate rand numbers from 0 to j
	    const int newIndex = static_cast<int>(curand_normal(&localState)) % (j + 1);
	    const size_t offset = j * nRows;
	    const size_t newOffset = newIndex * nRows;
	    
	    for (size_t i = tidX; i < nRows; i += stepX)
		    Shuffle(ptr, i + offset, i + newOffset);
	}
}

template <typename T>
GLOBAL void __RandShuffleColumnsPair__(T* RESTRICT ptr1, T* RESTRICT ptr2, CURAND_STATE_PTR states, const unsigned nRows1, const unsigned nRows2, const unsigned nCols)
{
	unsigned int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stepX = gridDim.x * blockDim.x;
	
	unsigned int tidY = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int stepY = gridDim.y * blockDim.y;
	curandState localState = states[tidY];
	
	for (size_t j = tidY; j < nCols; j += stepY)
	{
		// generate rand numbers from 0 to j
		const int newIndex = static_cast<int>(curand_normal(&localState)) % (j + 1);
		
		size_t offset = j * nRows1;
		size_t newOffset = newIndex * nRows1;
		for (size_t i = tidX; i < nRows1; i += stepX)
			Shuffle(ptr1, i + offset, i + newOffset);
		
		offset = j * nRows2;
		newOffset = newIndex * nRows2;
		for (size_t i = tidX; i < nRows2; i += stepX)
			Shuffle(ptr2, i + offset, i + newOffset);
	}
}