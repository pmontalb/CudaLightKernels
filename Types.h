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

	enum MatrixOperation
	{
		None = 0,
		Transpose = 1
	};
#ifdef __CUDACC__
	static const cublasOperation_t cublasOperation[] = { CUBLAS_OP_N, CUBLAS_OP_T };
#endif

	class IMemoryBuffer
	{
	public:
		ptr_t pointer;
		MemorySpace memorySpace;
		MathDomain mathDomain;

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

		virtual size_t TotalSize() const noexcept = 0;

		IMemoryBuffer(const ptr_t pointer = 0,
			const MemorySpace memorySpace = MemorySpace::Null,
			const MathDomain mathDomain = MathDomain::Null)
			: pointer(pointer), memorySpace(memorySpace), mathDomain(mathDomain)
		{
				
		}

		MAKE_DEFAULT_CONSTRUCTORS(IMemoryBuffer);
	};

	class MemoryBuffer : public IMemoryBuffer
	{
	public:
		unsigned size;

		size_t TotalSize() const noexcept override
		{
			return size * ElementarySize();
		}

		explicit MemoryBuffer(const ptr_t pointer = 0,
			const unsigned size = 0,
			const MemorySpace memorySpace = MemorySpace::Null,
			const MathDomain mathDomain = MathDomain::Null) noexcept
			: IMemoryBuffer(pointer, memorySpace, mathDomain), size(size)
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

	class ISparseMemoryBuffer : public IMemoryBuffer
	{
	public:
		unsigned nNonZeros;

		ISparseMemoryBuffer(const ptr_t pointer = 0,
			const MemorySpace memorySpace = MemorySpace::Null,
			const MathDomain mathDomain = MathDomain::Null,
			unsigned nNonZeros = 0)
			: IMemoryBuffer(pointer, memorySpace, mathDomain), nNonZeros(nNonZeros)
		{
				
		}

		size_t TotalSize() const noexcept override
		{
			return nNonZeros * ElementarySize();
		}

		MAKE_DEFAULT_CONSTRUCTORS(ISparseMemoryBuffer);
	};

	class SparseMemoryBuffer : public ISparseMemoryBuffer
	{
	public:
		ptr_t indices;

		SparseMemoryBuffer(const ptr_t pointer = 0,
			const ptr_t indices = 0,
			const MemorySpace memorySpace = MemorySpace::Null,
			const MathDomain mathDomain = MathDomain::Null,
			unsigned nNonZeros = 0)
			: ISparseMemoryBuffer(pointer, memorySpace, mathDomain, nNonZeros), indices(indices)
		{
				
		}

		MAKE_DEFAULT_CONSTRUCTORS(SparseMemoryBuffer);
	};

	/**
		* CSR Matrix representation
		*/
	class SparseMemoryTile : public ISparseMemoryBuffer
	{
	public:
		ptr_t nonZeroColumnIndices;
		unsigned nNonZeroRows;
		unsigned nRows;
		unsigned nCols;

		SparseMemoryTile(const ptr_t pointer = 0,
			const ptr_t nonZeroColumnIndices = 0,
			unsigned nNonZeroRows = 0,
			unsigned nRows = 0,
			unsigned nCols = 0,
			const MemorySpace memorySpace = MemorySpace::Null,
			const MathDomain mathDomain = MathDomain::Null,
			unsigned nNonZeros = 0)
			: ISparseMemoryBuffer(pointer, memorySpace, mathDomain, nNonZeros), 
				nonZeroColumnIndices(nonZeroColumnIndices), nNonZeroRows(nNonZeroRows), nRows(nRows), nCols(nCols)
		{
				
		}

		MAKE_DEFAULT_CONSTRUCTORS(SparseMemoryTile);
	};

	static size_t GetElementarySize(const IMemoryBuffer& memoryBuffer) noexcept
	{
		return memoryBuffer.ElementarySize();
	}

	static size_t GetTotalSize(const IMemoryBuffer& memoryBuffer) noexcept
	{
		return memoryBuffer.TotalSize();
	}
}

#undef MAKE_DEFAULT_CONSTRUCTORS;
