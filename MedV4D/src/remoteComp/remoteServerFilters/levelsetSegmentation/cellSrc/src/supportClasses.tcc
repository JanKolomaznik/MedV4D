
#ifndef SUPPORTCLASSES_H_
#error File supportClasses.tcc cannot be included directly!
#else

namespace M4D
{
namespace Cell
{

///////////////////////////////////////////////////////////////////////////////

template<typename TRadius, typename TOffset, uint8 Dim>
SparseFieldCityBlockNeighborList<TRadius, TOffset, Dim>
::SparseFieldCityBlockNeighborList()
{
	unsigned int i, nCenter;
	int d;
	TOffset zero_offset;

	m_size = SIZE;
	
	D_COMMAND( memset((void*)m_ArrayIndex, 0, SIZE * sizeof(unsigned int)); )
	D_COMMAND( memset((void*)m_NeighborhoodOffset, 0, SIZE * sizeof(TOffset)); )

	TSize size;

	for (i = 0; i < Dim; ++i)
	{
		m_Radius[i] = 1;
		zero_offset[i] = 0;
		size[i] = 3;
	}

	nCenter = (Dim * Dim * Dim) / 2;

	for (i = 0; i < SIZE; ++i)
	{
		m_NeighborhoodOffset[i] = zero_offset;
	}

	ComputeStridesFromSize<TSize, TStrides>(size, m_StrideTable);

	uint32 pushIndex = 0;

	for (d = Dim - 1, i = 0; d >= 0; --d, ++i)
	{
		m_ArrayIndex[pushIndex++] = nCenter - m_StrideTable[d];
		m_NeighborhoodOffset[i][d] = -1;
	}
	for (d = 0; d < static_cast<int>(Dim); ++d, ++i)
	{
		m_ArrayIndex[pushIndex++] = nCenter + m_StrideTable[d];
		m_NeighborhoodOffset[i][d] = 1;
	}
}

///////////////////////////////////////////////////////////////////////////////

template<typename TRadius, typename TOffset, uint8 Dim>
void
SparseFieldCityBlockNeighborList<TRadius, TOffset, Dim>
::Print(std::ostream &os) const
{
	os << "SparseFieldCityBlockNeighborList: " << std::endl;
	for (unsigned i = 0; i < SIZE; ++i)
	{
		os << "m_ArrayIndex[" << i << "]: " << m_ArrayIndex[i] << std::endl;
		//os << "m_NeighborhoodOffset[" << i << "]: " << m_NeighborhoodOffset[i]
		//<< std::endl;
	}
}

///////////////////////////////////////////////////////////////////////////////

}
}

#endif
