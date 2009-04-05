#ifndef NEIGHBORHOODCELL_H_
#error File neighborhoodCell.tcc cannot be included directly!
#else

///////////////////////////////////////////////////////////////////////////////

template< typename TPixel, uint8 Dim>
NeighborhoodCell<TPixel, Dim>
	::NeighborhoodCell(const RadiusType &radius, TImageProperties *props)
		: m_radius(radius), m_imageProps(props)
{
	// check validity of given radius
	for(uint8 i=0; i < Dim; i++)
	{
		if(radius[i] <= 0)
			;// TODO throw exception
	}
	
	// set size
	for (unsigned int i=0; i<Dim; ++i)
		{ m_radiusSize[i] = m_radius[i]*2+1; }
	
	// count size of buffer in linear manner
	m_size = m_radiusSize[0];
	for(uint8 i=1; i < Dim; i++)
		m_size *= m_radiusSize[i];
	
	m_buf = new TPixel[m_size];
	
	ComputeStridesFromSize(m_radiusSize, m_radiusStrides);
	ComputeStridesFromSize(m_imageProps->region.size, m_imageStrides);
}

///////////////////////////////////////////////////////////////////////////////

template< typename TPixel, uint8 Dim>
void
NeighborhoodCell<TPixel, Dim>
::ComputeStridesFromSize(const TSize &size, TStrides &strides)
{
  unsigned int accum;

  accum = 1;
  strides[0] = 1;
  for (unsigned int dim = 1; dim < Dim; ++dim)
    {
	  accum *= size[dim-1];
	  strides[dim] = accum;
	  }
}

///////////////////////////////////////////////////////////////////////////////

template< typename TPixel, uint8 Dim>
TPixel *
NeighborhoodCell<TPixel, Dim>
::ComputeImageDataPointer(const TIndex &pos)
{
	TPixel *pointer = m_imageProps->imageData;
	for(uint8 i=0; i<Dim; i++)
	{
		pointer += pos[i] * m_imageStrides[i];
	}
	return pointer;
}

///////////////////////////////////////////////////////////////////////////////

template< typename TPixel, uint8 Dim>
void
NeighborhoodCell<TPixel, Dim>
::LoadData(TPixel *src, TPixel *dest, size_t size)
{
	// copy the memory
	memcpy((void*)dest, (void*)src, size);
}

///////////////////////////////////////////////////////////////////////////////

template< typename TPixel, uint8 Dim>
bool
NeighborhoodCell<TPixel, Dim>
::IsWithinImage(const TIndex &pos)
{
	for(uint32 i=0; i<Dim; i++)
	{
		if((pos[i] < m_imageProps->region.offset[i]) 
				|| (pos[i] >= (m_imageProps->region.offset[i] + m_imageProps->region.size[i]) ) )
			return false;
	}
	return true;
}

///////////////////////////////////////////////////////////////////////////////

template< typename TPixel, uint8 Dim>
void
NeighborhoodCell<TPixel, Dim>
::LoadSlice(TIndex posm, uint8 dim, TPixel *dest)
{
	posm[dim] -= m_radius[dim];
	if(dim == 0)
	{		
		TPixel *begin = ComputeImageDataPointer(posm);
		LoadData(begin, dest, m_radiusSize[dim] * sizeof(TPixel));
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

template< typename TPixel, uint8 Dim>
void
NeighborhoodCell<TPixel, Dim>
	::SetPosition(const TIndex &pos)
{
	m_currIndex = pos;
	
#define DEFAULT_VAL 0
	// fill the buff
	memset((void*)m_buf, DEFAULT_VAL, m_size * sizeof(TPixel));
	
	TIndex iteratingIndex(pos);
	iteratingIndex[Dim-1] -= m_radius[Dim-1];
	for(uint32 i=0; i<m_radiusSize[Dim-1]; i++)
	{
		if(IsWithinImage(iteratingIndex))
			LoadSlice(iteratingIndex, Dim-2, m_buf + (i * m_radiusStrides[Dim-1]));
		iteratingIndex[Dim-1] += 1;	// move in iteration  direction
	}
//	const Iterator _end = Superclass::End();
//	  InternalPixelType * Iit;
//	  ImageType *ptr = const_cast<ImageType *>(m_ConstImage.GetPointer());
//	  const SizeType size = this->GetSize();
//	  const OffsetValueType *OffsetTable = m_ConstImage->GetOffsetTable();
//	  const SizeType radius = this->GetRadius();
//
//	  unsigned int i;
//	  Iterator Nit;
//	  SizeType loop;
//	  for (i=0; i<Dimension; ++i) loop[i]=0;
//
//	  // Find first "upper-left-corner"  pixel address of neighborhood
//	  Iit = ptr->GetBufferPointer() + ptr->ComputeOffset(pos);
//
//	  for (i = 0; i<Dimension; ++i)
//	    {
//	    Iit -= radius[i] * OffsetTable[i];
//	    }
//
//	  // Compute the rest of the pixel addresses
//	  for (Nit = Superclass::Begin(); Nit != _end; ++Nit)
//	    {
//	    *Nit = Iit;
//	    ++Iit;
//	    for (i = 0; i <Dimension; ++i)
//	      {
//	      loop[i]++;
//	      if ( loop[i] == size[i] )
//	        {
//	        if (i==Dimension-1) break;
//	        Iit += OffsetTable[i+1] - OffsetTable[i] * static_cast<long>(size[i]);
//	        loop[i]= 0;
//	        }
//	      else break;
//	      }
//	    }
}

///////////////////////////////////////////////////////////////////////////////

template<typename T1, typename T2>
T1 operator+ ( const T1 &v1, const T2 &v2 )
{
	T1 tmp(v1);
	for(uint32 i=0; i<T1::Dimension; i++)
	{
		tmp[i] += v2[i];
	}
	return tmp;
}
///////////////////////////////////////////////////////////////////////////////

template< typename TPixel, uint8 Dim>
void
NeighborhoodCell<TPixel, Dim>::Print(std::ostream &stream)
{
	stream << "m_currIndex: " << m_currIndex << std::endl;
	
	TOffset iteratingIndex;
	for(uint32 i=0; i<Dim; i++)
	{
		iteratingIndex[i] = -m_radius[i];
	}
	TOffset begin(iteratingIndex);
	
	
	for(uint32 i=0; i<m_radiusSize[2]; i++)
	{
		iteratingIndex[1] = begin[1];
		for(uint32 j=0; j<m_radiusSize[1]; j++)
		{
			iteratingIndex[0] = begin[0];
			for(uint32 k=0; k<m_radiusSize[0]; k++)
			{
				stream << "(" << (iteratingIndex + m_currIndex) << ") = " << GetPixel(GetNeighborhoodIndex(iteratingIndex)) << std::endl;
				iteratingIndex[0] += 1;				
			}
			iteratingIndex[1] += 1;
		}
		iteratingIndex[2] += 1;
	}
}

///////////////////////////////////////////////////////////////////////////////

template< typename TPixel, uint8 Dim>
const uint32 
NeighborhoodCell<TPixel, Dim>::GetNeighborhoodIndex(const TOffset &o) const
{
  uint32 idx = (m_size/2);
  for (unsigned i = 0; i < Dim; ++i)
    {      idx+=o[i] * static_cast<long>(m_radiusStrides[i]);    }
  return idx;
}

///////////////////////////////////////////////////////////////////////////////

#endif
