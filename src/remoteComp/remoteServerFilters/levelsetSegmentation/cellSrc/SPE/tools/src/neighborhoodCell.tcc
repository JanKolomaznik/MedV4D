#ifndef NEIGHBORHOODCELL_H_
#error File neighborhoodCell.tcc cannot be included directly!
#else

namespace M4D {
namespace Cell {

#include <string.h>

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
NeighborhoodCell<PixelType>::NeighborhoodCell(TImageProperties<PixelType> *props)
		: m_imageProps(props)
{
	// set size
	for (unsigned int i=0; i<DIM; ++i)
		{ m_radiusSize[i] = SIZEIN1DIM; }
	
	// count size of buffer in linear manner
	m_size = NEIGHBOURHOOD_SIZE;
	
	ComputeStridesFromSize(m_radiusSize, m_radiusStrides);
	ComputeStridesFromSize(m_imageProps->region.size, m_imageStrides);
}

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
PixelType *
NeighborhoodCell<PixelType>
::ComputeImageDataPointer(const TIndex &pos)
{
	PixelType *pointer = m_imageProps->imageData;
	for(uint8 i=0; i<DIM; i++)
	{
		pointer += pos[i] * m_imageStrides[i];
	}
	return pointer;
}

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
void
NeighborhoodCell<PixelType>
::LoadData(PixelType *src, PixelType *dest, size_t size)
{
	// copy the memory
	memcpy((void*)dest, (void*)src, size);
}

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
bool
NeighborhoodCell<PixelType>
::IsWithinImage(const TIndex &pos)
{
	for(uint32 i=0; i<DIM; i++)
	{
		if((pos[i] < m_imageProps->region.offset[i]) 
				|| (pos[i] >= (m_imageProps->region.offset[i] + m_imageProps->region.size[i]) ) )
			return false;
	}
	return true;
}

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
void
NeighborhoodCell<PixelType>
::LoadSlice(TIndex posm, uint8 dim, PixelType *dest)
{
	posm[dim] -= RADIUS;
	if(dim == 0)
	{		
		PixelType *begin = ComputeImageDataPointer(posm);
		LoadData(begin, dest, m_radiusSize[dim] * sizeof(PixelType));
	}
	else
	{
		TIndex iteratingIndex(posm);
		for(uint32 i=0; i<m_radiusSize[dim]; i++)
		{
			if(IsWithinImage(iteratingIndex))
				LoadSlice(iteratingIndex, dim-1, dest + (i * m_radiusStrides[dim]));
			iteratingIndex[dim] += 1;	// move in iteration  direction
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
void
NeighborhoodCell<PixelType>::SetCenterPixel(PixelType val)
{
	PixelType *begin = ComputeImageDataPointer(m_currIndex);
	*begin = val;
}

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
void
NeighborhoodCell<PixelType>
	::SetPosition(const TIndex &pos)
{
	m_currIndex = pos;
	
#define DEFAULT_VAL 0
	// fill the buff
	memset((void*)m_buf, DEFAULT_VAL, m_size * sizeof(PixelType));
	
	TIndex iteratingIndex(pos);
	iteratingIndex[DIM-1] -= RADIUS;
	for(uint32 i=0; i<m_radiusSize[DIM-1]; i++)
	{
		if(IsWithinImage(iteratingIndex))
			LoadSlice(iteratingIndex, DIM-2, m_buf + (i * m_radiusStrides[DIM-1]));
		iteratingIndex[DIM-1] += 1;	// move in iteration  direction
	}
}

///////////////////////////////////////////////////////////////////////////////

//template<typename T1, typename T2>
//T1 operator+ ( const T1 &v1, const T2 &v2 )
//{
//	T1 tmp(v1);
//	for(uint32 i=0; i<T1::Dimension; i++)
//	{
//		tmp[i] += v2[i];
//	}
//	return tmp;
//}
///////////////////////////////////////////////////////////////////////////////


//void
//NeighborhoodCell::Print(std::ostream &stream)
//{
//	stream << "m_currIndex: " << m_currIndex << std::endl;
//	
//	TOffset iteratingIndex;
//	for(uint32 i=0; i<DIM; i++)
//	{
//		iteratingIndex[i] = -m_radius[i];
//	}
//	TOffset begin(iteratingIndex);
//	
//	
//	for(uint32 i=0; i<m_radiusSize[2]; i++)
//	{
//		iteratingIndex[1] = begin[1];
//		for(uint32 j=0; j<m_radiusSize[1]; j++)
//		{
//			iteratingIndex[0] = begin[0];
//			for(uint32 k=0; k<m_radiusSize[0]; k++)
//			{
//				stream << "(" << (iteratingIndex + m_currIndex) << ") = " << GetPixel(GetNeighborhoodIndex(iteratingIndex)) << std::endl;
//				iteratingIndex[0] += 1;
//			}
//			iteratingIndex[1] += 1;
//		}
//		iteratingIndex[2] += 1;
//	}
//}

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
uint32 
NeighborhoodCell<PixelType>::GetNeighborhoodIndex(const TOffset &o) const
{
  uint32 idx = (m_size/2);
  for (unsigned i = 0; i < DIM; ++i)
    {      idx+=o[i] * static_cast<long>(m_radiusStrides[i]);    }
  return idx;
}

///////////////////////////////////////////////////////////////////////////////

}
}
#endif
