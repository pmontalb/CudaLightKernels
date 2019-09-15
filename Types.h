#pragma once

#include "Flags.cuh"
#include <assert.h>

#define MAKE_DEFAULT_CONSTRUCTORS(CLASS)\
	virtual ~CLASS() noexcept = default;\
	CLASS(const CLASS& rhs) noexcept = default;\
	CLASS(CLASS&& rhs) noexcept = default;\
	CLASS& operator=(const CLASS& rhs) noexcept = default;\
	CLASS& operator=(CLASS&& rhs) noexcept = default\

#ifdef __CUDACC__
#include <cublas_v2.h>
static const cublasOperation_t cublasOperation[] = { CUBLAS_OP_N, CUBLAS_OP_T };

#include <cusparse_v2.h>
static const cusparseOperation_t cusparseOperation[] = { CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE };
#endif

EXTERN_C
{
	enum CudaKernelException
	{
		_NotImplementedException = -1,
		_NotSupportedException = -2,
		_ExpectedEvenSizeException = -3,
		_InternalException = -4
	};

	enum class MemorySpace
	{
		Null,
		Host,
		Device
	};

	enum class MathDomain
	{
		Null,
		Int,
		Float,
		Double,
		UnsignedChar,
	};

	enum class MatrixOperation : unsigned int
	{
		None = 0,
		Transpose = 1
	};

	class MemoryBuffer
	{
	public:
		ptr_t pointer;
		MemorySpace memorySpace;
		MathDomain mathDomain;
		unsigned size;

		size_t ElementarySize() const noexcept
		{
			switch (mathDomain)
			{
			case MathDomain::Double:
				return sizeof(double);
			case MathDomain::Float:
				return sizeof(float);
			case MathDomain::Int:
				return sizeof(int);
			case MathDomain::UnsignedChar:
				return sizeof(unsigned char);
			default:
				return 0;
			}
		}

		virtual size_t TotalSize() const noexcept
		{
			return size * ElementarySize();
		}

		MemoryBuffer(const ptr_t pointer_ = 0,
			const unsigned size_ = 0,
			const MemorySpace memorySpace_ = MemorySpace::Null,
			const MathDomain mathDomain_ = MathDomain::Null)
			: pointer(pointer_), memorySpace(memorySpace_), mathDomain(mathDomain_), size(size_)
		{
				
		}

		MAKE_DEFAULT_CONSTRUCTORS(MemoryBuffer);
	};

	class MemoryTile : public MemoryBuffer
	{
	public:
		unsigned nRows;
		unsigned nCols;

		explicit MemoryTile(const ptr_t pointer_ = 0,
			const unsigned nRows_ = 0,
			const unsigned nCols_ = 0,
			const MemorySpace memorySpace_ = MemorySpace::Null,
			const MathDomain mathDomain_ = MathDomain::Null) noexcept
			: MemoryBuffer(pointer_, nRows_ * nCols_, memorySpace_, mathDomain_), nRows(nRows_), nCols(nCols_)
		{

		}

		explicit MemoryTile(const MemoryBuffer& buffer) noexcept
			: MemoryBuffer(buffer), nRows(buffer.size), nCols(1)
		{

		}

		MAKE_DEFAULT_CONSTRUCTORS(MemoryTile);

	protected:
		explicit MemoryTile(const ptr_t pointer_,
			const unsigned nRows_, const unsigned nCols_, const unsigned size_,
			const MemorySpace memorySpace_, const MathDomain mathDomain_)
			: MemoryBuffer(pointer_, size_, memorySpace_, mathDomain_), nRows(nRows_), nCols(nCols_)
		{

		}
	};

	class MemoryCube : public MemoryTile
	{
	public:
		unsigned nCubes;

		explicit MemoryCube(const ptr_t pointer_ = 0,
			const unsigned nRows_ = 0,
			const unsigned nCols_ = 0,
			const unsigned nCubes_ = 0,
			const MemorySpace memorySpace_ = MemorySpace::Null,
			const MathDomain mathDomain_ = MathDomain::Null) noexcept
			: MemoryTile(pointer_, nRows_, nCols_, nRows_ * nCols_ * nCubes_, memorySpace_, mathDomain_), nCubes(nCubes_)
		{

		}

		explicit MemoryCube(const MemoryTile& tile) noexcept
			: MemoryTile(tile), nCubes(1)
		{

		}
		
		MAKE_DEFAULT_CONSTRUCTORS(MemoryCube);
	};

	class SparseMemoryBuffer : public MemoryBuffer
	{
	public:
		ptr_t indices;

		explicit SparseMemoryBuffer(const ptr_t pointer_ = 0,
			const unsigned nNonZeros_ = 0,
			const ptr_t indices_ = 0,
			const MemorySpace memorySpace_ = MemorySpace::Null,
			const MathDomain mathDomain_ = MathDomain::Null)
			: MemoryBuffer(pointer_, nNonZeros_, memorySpace_, mathDomain_), indices(indices_)
		{
				
		}

		MAKE_DEFAULT_CONSTRUCTORS(SparseMemoryBuffer);
	};

	/**
	* CSR Matrix representation
	*/
	class SparseMemoryTile : public MemoryBuffer
	{
	public:
		ptr_t nonZeroColumnIndices;
		ptr_t nNonZeroRows;
		unsigned nRows;
		unsigned nCols;

		explicit SparseMemoryTile(const ptr_t pointer_ = 0,
			const unsigned nNonZeros_ = 0,
			const ptr_t nonZeroColumnIndices_ = 0,
		    const ptr_t nNonZeroRows_ = 0,
			const unsigned nRows_ = 0,
			const unsigned nCols_ = 0,
			const MemorySpace memorySpace_ = MemorySpace::Null,
			const MathDomain mathDomain_ = MathDomain::Null)
			: MemoryBuffer(pointer_, nNonZeros_, memorySpace_, mathDomain_),
				nonZeroColumnIndices(nonZeroColumnIndices_), nNonZeroRows(nNonZeroRows_), nRows(nRows_), nCols(nCols_)
		{
				
		}

		MAKE_DEFAULT_CONSTRUCTORS(SparseMemoryTile);
	};

	static size_t GetElementarySize(const MemoryBuffer& memoryBuffer) noexcept
	{
		return memoryBuffer.ElementarySize();
	}

	static size_t GetTotalSize(const MemoryBuffer& memoryBuffer) noexcept
	{
		return memoryBuffer.TotalSize();
	}

	static void ExtractColumnBufferFromMatrix(MemoryBuffer& out, const MemoryTile& rhs, const unsigned column)
	{
		assert(column < rhs.nCols);
		out = MemoryBuffer(rhs.pointer + column * rhs.nRows * rhs.ElementarySize(), rhs.nRows, rhs.memorySpace, rhs.mathDomain);
	}

	static void ExtractColumnBufferFromCube(MemoryBuffer& out, const MemoryCube& rhs, const unsigned matrix, const unsigned column)
	{
		assert(matrix < rhs.nCubes);
		assert(column < rhs.nCols);
		out = MemoryBuffer(rhs.pointer + rhs.nRows * (matrix * rhs.nCols + column)* rhs.ElementarySize(), 
						 rhs.nRows,
						 rhs.memorySpace, 
						 rhs.mathDomain);
	}

	static void ExtractMatrixBufferFromCube(MemoryTile& out, const MemoryCube& rhs, const unsigned matrix)
	{
		assert(matrix < rhs.nCubes);
		out = MemoryTile(rhs.pointer + matrix * rhs.nRows * rhs.nCols * rhs.ElementarySize(),
					   rhs.nRows, rhs.nCols,
					   rhs.memorySpace,
					   rhs.mathDomain);
	}
}

#undef MAKE_DEFAULT_CONSTRUCTORS

