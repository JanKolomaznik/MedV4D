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
				|| (pos[i] >= (m_imageProps->region.offset[i] + (int32)m_imageProps->region.size[i]) ) )
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
	// change the buffer as well
	m_buf[static_cast<uint32>(m_size/2)] = val;
}

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
void
NeighborhoodCell<PixelType>::SetPixel(PixelType val, TOffset pos)
{
	TIndex i = m_currIndex + pos;
	if(IsWithinImage(i))
	{
		PixelType *begin = ComputeImageDataPointer(i);
		*begin = val;
	}
	// change the buffer as well !!!!!!!!
	//std::cout << "setting" << i << "=" << val << std::endl;
	m_buf[GetNeighborhoodIndex(pos)] = val;
}



// xxxxxxxxxxxxxxxxxxxxxxxxxxx
//template<typename PixelType>
//PixelType
//NeighborhoodCell<PixelType>::DebugGetImagePixel(TOffset off)
//{
//	TIndex i = m_currIndex + off;
//	if(IsWithinImage(i))
//		{
//		return *ComputeImageDataPointer(i);
//		}
//}
//	
//template<typename PixelType>
//void
//NeighborhoodCell<PixelType>::DebugSetImagePixel(TOffset off, PixelType val)
//{
//	TIndex i = m_currIndex + off;
//	if(IsWithinImage(i))
//	{
//		PixelType *begin = ComputeImageDataPointer(i);
//		*begin = val;
//	}
//}
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

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


template<typename PixelType>
std::ostream & operator<<(std::ostream &stream, NeighborhoodCell<PixelType> &n)
{
	stream << "m_currIndex: " << n.m_currIndex << std::endl;
	
	TOffset iteratingIndex;
	for(uint32 i=0; i<DIM; i++)
	{
		iteratingIndex[i] = -RADIUS;
	}
	TOffset begin(iteratingIndex);
	PixelType val;
	
	for(uint32 i=0; i<SIZEIN1DIM; i++)
	{
		iteratingIndex[1] = begin[1];
		for(uint32 j=0; j<SIZEIN1DIM; j++)
		{
			iteratingIndex[0] = begin[0];
			for(uint32 k=0; k<SIZEIN1DIM; k++)
			{
				val = n.GetPixel(n.GetNeighborhoodIndex(iteratingIndex));
				stream << (iteratingIndex) << "=" << (int32) val << std::endl;
				iteratingIndex[0] += 1;
			}
			iteratingIndex[1] += 1;
		}
		iteratingIndex[2] += 1;
	}
	return stream;
}

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
void
NeighborhoodCell<PixelType>::PrintImage(std::ostream &s)
{
	//s << "image: " << std::endl;
	s << "size: " << m_imageProps->region.size[0] << "," << m_imageProps->region.size[1] << "," << m_imageProps->region.size[2] << std::endl;
	PixelType *data;
	TIndex ind;
	
	for(uint32 i=0; i<m_imageProps->region.size[0]; i++)
	{
		for(uint32 j=0; j<m_imageProps->region.size[1]; j++)
		{
			for(uint32 k=0; k<m_imageProps->region.size[2]; k++)
			{
				ind[0] = i; ind[1] = j; ind[2] = k;
				data = ComputeImageDataPointer(ind);
				s << "[" << ind[0] << "," << ind[1] << "," << ind[2] << "]"  << "= " << ((int32)*data) << std::endl;
//				data++;
//				i[0]++;
			}
//			i[1]++;
//			i[0] = m_imageProps->region.offset[0];
		}
//		i[2]++;
//		i[1] = m_imageProps->region.offset[1];
//		i[0] = m_imageProps->region.offset[0];
	}
}

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

template<typename PixelType>
void
NeighborhoodCell<PixelType>
::HowMuchCrossesBoundary(TOffset &howMuch)
{
	TIndex pos = m_currIndex + howMuch;
	for(uint32 i=0; i<DIM; i++)
	{
		if(pos[i] < m_imageProps->region.offset[i])
			howMuch[i] = m_imageProps->region.offset[i] - pos[i];
		else if(pos[i] >= (m_imageProps->region.offset[i] + (int32)m_imageProps->region.size[i]) )
			howMuch[i] = (m_imageProps->region.offset[i] + (int32)m_imageProps->region.size[i] - 1) - pos[i];
		else
			howMuch[i] = 0;
	}
}

///////////////////////////////////////////////////////////////////////////////

}
}
#endif
