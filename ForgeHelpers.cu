#include "ForgeHelpers.cuh"

// this assumes the in array has been normalised in [0-1]
template <typename T, typename V>
GLOBAL void __MakeRgbaJetColorMap__(T* RESTRICT out, const V* RESTRICT in, const size_t sz)
{
    CUDA_FUNCTION_PROLOGUE

    CUDA_FOR_LOOP_PROLOGUE
        float r = 0.0f, g = 0.0f, b = 0.0f;

        if (in[i] < 0.25f)
        {
            r = b = 0.0f;
            g = 4.0f * in[i];
        }
        else if (in[i] < 0.5f)
        {
            r = g = 0.0f;
            b = 1.0f + 4.0f * (.25f - in[i]);
        }
        else if (in[i] < 0.75)
        {
            r = 4.0f * (in[i] - 0.5f);
            b = g = 0.0f;
        }
        else
        {
            g = 1.0f + 4.0f * (0.75f - in[i]);
            b = r = 0.0f;
        }

        out[i * 4 + 0] = static_cast<T>(r);
        out[i * 4 + 1] = static_cast<T>(b);
        out[i * 4 + 2] = static_cast<T>(g);
        out[i * 4 + 3] = 1.0;
    CUDA_FOR_LOOP_EPILOGUE
}

// this assumes the in array has been normalised in [0-1]
template <typename V>
GLOBAL void  __MakeRgbaJetColorMap__(unsigned char* RESTRICT out, const V* RESTRICT in, const size_t sz)
{
    CUDA_FUNCTION_PROLOGUE

    CUDA_FOR_LOOP_PROLOGUE
        float r = 0.0f, g = 0.0f, b = 0.0f;

        if (in[i] < 0.25f)
        {
            r = b = 0.0f;
            g = 4.0f * in[i];
        }
        else if (in[i] < 0.5f)
        {
            r = g = 0.0f;
            b = 1.0f + 4.0f * (.25f - in[i]);
        }
        else if (in[i] < 0.75)
        {
            r = 4.0f * (in[i] - 0.5f);
            b = g = 0.0f;
        }
        else
        {
            g = 1.0f + 4.0f * (0.75f - in[i]);
            b = r = 0.0f;
        }

        out[i * 4 + 0] = static_cast<unsigned char>(255 * r);
        out[i * 4 + 1] = static_cast<unsigned char>(255 * b);
        out[i * 4 + 2] = static_cast<unsigned char>(255 * g);
        out[i * 4 + 3] = 255;
    CUDA_FOR_LOOP_EPILOGUE
}

EXTERN_C
{
	EXPORT int _MakePair(MemoryBuffer z, const MemoryBuffer x, const MemoryBuffer y)
	{
		assert(z.mathDomain == MathDomain::Float);
		assert(x.size == y.size);
		assert(z.size == 2 * x.size);
		switch (z.mathDomain)
		{
			case MathDomain::Float:
			{
				CUDA_CALL_SINGLE(__MakePair__, (float*)z.pointer, (float*)x.pointer, (float*)y.pointer, x.size);
				break;
			}
			case MathDomain::Double:
			{
				CUDA_CALL_DOUBLE(__MakePair__, (float*)z.pointer, (double*)x.pointer, (double*)y.pointer, x.size);
				break;
			}
			case MathDomain::Int:
			{
				CUDA_CALL_SINGLE(__MakePair__, (float*)z.pointer, (int*)x.pointer, (int*)y.pointer, x.size);
				break;
			}
			default:
				return CudaKernelException::_NotImplementedException;
		}

		return cudaGetLastError();
	}

	EXPORT int _MakeTriple(MemoryBuffer v, const MemoryBuffer x, const MemoryBuffer y, const MemoryBuffer z)
	{
		assert(v.mathDomain == MathDomain::Float);
		assert(x.size * y.size == z.size);
		assert(v.size == 3 * z.size);
		switch (z.mathDomain)
		{
			case MathDomain::Float:
			{
				CUDA_CALL_SINGLE(__MakeTriple__, (float*)v.pointer, (float*)x.pointer, (float*)y.pointer, (float*)z.pointer, x.size, y.size);
				break;
			}
			case MathDomain::Double:
			{
				CUDA_CALL_DOUBLE(__MakeTriple__, (float*)v.pointer, (double*)x.pointer, (double*)y.pointer, (double*)z.pointer, x.size, y.size);
				break;
			}
			case MathDomain::Int:
			{
				CUDA_CALL_SINGLE(__MakeTriple__, (float*)v.pointer, (int*)x.pointer, (int*)y.pointer, (int*)z.pointer, x.size, y.size);
				break;
			}
			default:
				return CudaKernelException::_NotImplementedException;
		}

		return cudaGetLastError();
	}

	EXPORT int _MakeRgbaJetColorMap(MemoryBuffer out, const MemoryBuffer in)
	{
		assert(4 * in.size == out.size);

#define CALL(T)\
		switch (in.mathDomain)\
		{\
			case MathDomain::Float:\
			{\
				CUDA_CALL_SINGLE(__MakeRgbaJetColorMap__, (T*)out.pointer, (float*)in.pointer, in.size);\
				break;\
			}\
			case MathDomain::Double:\
			{\
				CUDA_CALL_DOUBLE(__MakeRgbaJetColorMap__, (T*)out.pointer, (double*)in.pointer, in.size);\
				break;\
			}\
			case MathDomain::Int:\
			{\
				CUDA_CALL_SINGLE(__MakeRgbaJetColorMap__, (T*)out.pointer, (int*)in.pointer, in.size);\
				break;\
			}\
			default:\
				return CudaKernelException::_NotImplementedException;\
		}

		switch (out.mathDomain)
		{
			case MathDomain::Float:
			{
				CALL(float);
				break;
			}
			case MathDomain::Double:
			{
				CALL(double);
				break;
			}
			case MathDomain::Int:
			{
				CALL(int);
				break;
			}
			case MathDomain::UnsignedChar:
			{
				CALL(unsigned char);
				break;
			}
			default:
				return CudaKernelException::_NotImplementedException;
		}

#undef CALL
		return cudaGetLastError();
	}
}

template <typename T>
GLOBAL void __MakePair__(float* RESTRICT z, const T* RESTRICT x, const T* RESTRICT y, const size_t sz)
{
	CUDA_FUNCTION_PROLOGUE

	CUDA_FOR_LOOP_PROLOGUE

		const size_t idx = i << 1;
		z[idx    ] = static_cast<float>(x[i]);
		z[idx + 1] = static_cast<float>(y[i]);

	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __MakeTriple__(float* RESTRICT v, const T* RESTRICT x, const T* RESTRICT y, const T* RESTRICT z, const size_t nRows, const size_t nCols)
{
	const int tidX = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int stepX = gridDim.x * blockDim.x;
	const int tidY = blockDim.y * blockIdx.y + threadIdx.y;
	const unsigned int stepY = gridDim.y * blockDim.y;

	for (size_t i = tidX; i < nRows; i += stepX)
	{
		for (size_t j = tidY; j < nCols; j += stepY)
		{
			const size_t offset = j + i * nCols;
			v[3 * offset    ] = x[i];
			v[3 * offset + 1] = y[j];
			v[3 * offset + 2] = z[i + j * nRows];
		}
	}
}