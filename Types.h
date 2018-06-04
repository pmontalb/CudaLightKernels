#pragma once

#include "Flags.cuh"

#define MAKE_DEFAULT_CONSTRUCTORS(CLASS)\
	virtual ~CLASS() noexcept = default;\
	CLASS(const CLASS& rhs) noexcept = default;\
	CLASS(CLASS&& rhs) noexcept = default;\
	CLASS& operator=(const CLASS& rhs) noexcept = default;\
	CLASS& operator=(CLASS&& rhs) noexcept = default;\

extern "C"
{
#ifdef __CUDACC__
	#include <cublas_v2.h>
	static const cublasOperation_t cublasOperation[] = { CUBLAS_OP_N, CUBLAS_OP_T };

	#include <cusparse_v2.h>
	static const cusparseOperation_t cusparseOperation[] = { CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE };
#endif

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
		Double
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
			default:
				return 0;
			}
		}

		virtual size_t TotalSize() const noexcept
		{
			return size * ElementarySize();
		};

		MemoryBuffer(const ptr_t pointer = 0,
			const unsigned size = 0,
			const MemorySpace memorySpace = MemorySpace::Null,
			const MathDomain mathDomain = MathDomain::Null)
			: pointer(pointer), memorySpace(memorySpace), mathDomain(mathDomain), size(size)
		{
				
		}

		MAKE_DEFAULT_CONSTRUCTORS(MemoryBuffer);
	};

	class MemoryTile : public MemoryBuffer
	{
	public:
		unsigned nRows;
		unsigned nCols;

		explicit MemoryTile(const ptr_t pointer = 0,
			const unsigned nRows = 0,
			const unsigned nCols = 0,
			const MemorySpace memorySpace = MemorySpace::Null,
			const MathDomain mathDomain = MathDomain::Null) noexcept
			: MemoryBuffer(pointer, nRows * nCols, memorySpace, mathDomain), nRows(nRows), nCols(nCols)
		{

		}

		MemoryTile(const MemoryBuffer& buffer) noexcept
			: MemoryBuffer(buffer), nRows(nRows), nCols(1)
		{

		}

		MAKE_DEFAULT_CONSTRUCTORS(MemoryTile);

	protected:
		explicit MemoryTile(const ptr_t pointer = 0,
			const unsigned nRows = 0,
			const unsigned nCols = 0,
			const unsigned size = 0,
			const MemorySpace memorySpace = MemorySpace::Null,
			const MathDomain mathDomain = MathDomain::Null)
			: MemoryBuffer(pointer, size, memorySpace, mathDomain), nRows(nRows), nCols(nCols)
		{

		}
	};

	class MemoryCube : public MemoryTile
	{
	public:
		unsigned nCubes;

		explicit MemoryCube(const ptr_t pointer = 0,
			const unsigned nRows = 0,
			const unsigned nCols = 0,
			const unsigned nCubes = 0,
			const MemorySpace memorySpace = MemorySpace::Null,
			const MathDomain mathDomain = MathDomain::Null) noexcept
			: MemoryTile(pointer, nRows, nCols, nRows * nCols * nCubes, memorySpace, mathDomain), nCubes(nCubes)
		{

		}

		MemoryCube(const MemoryTile& tile) noexcept
			: MemoryTile(tile), nCubes(1)
		{

		}

		MAKE_DEFAULT_CONSTRUCTORS(MemoryCube);
	};

	class SparseMemoryBuffer : public MemoryBuffer
	{
	public:
		ptr_t indices;

		SparseMemoryBuffer(const ptr_t pointer = 0,
			const unsigned nNonZeros = 0,
			const ptr_t indices = 0,
			const MemorySpace memorySpace = MemorySpace::Null,
			const MathDomain mathDomain = MathDomain::Null)
			: MemoryBuffer(pointer, nNonZeros, memorySpace, mathDomain), indices(indices)
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

		SparseMemoryTile(const ptr_t pointer = 0,
			const unsigned nNonZeros = 0,
			const ptr_t nonZeroColumnIndices = 0,
		    const ptr_t nNonZeroRows = 0,
			const unsigned nRows = 0,
			const unsigned nCols = 0,
			const MemorySpace memorySpace = MemorySpace::Null,
			const MathDomain mathDomain = MathDomain::Null)
			: MemoryBuffer(pointer, nNonZeros, memorySpace, mathDomain),
				nonZeroColumnIndices(nonZeroColumnIndices), nNonZeroRows(nNonZeroRows), nRows(nRows), nCols(nCols)
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
}

#undef MAKE_DEFAULT_CONSTRUCTORS

