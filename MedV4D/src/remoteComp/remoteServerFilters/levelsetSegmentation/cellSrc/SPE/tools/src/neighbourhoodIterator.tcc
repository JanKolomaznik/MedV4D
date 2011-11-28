#ifndef NEIGHBOURHOODITERATOR_H_
#error File neighbourhoodIterator.tcc cannot be included directly!
#else

namespace M4D {
namespace Cell {

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
NeighbourIteratorCell<PixelType>
::NeighbourIteratorCell()
	: m_neighbourhood(0) 
{
}

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
NeighbourIteratorCell<PixelType>
::NeighbourIteratorCell(NeighborhoodType *neiborhood)
	: m_neighbourhood(neiborhood)
{
ComputeNeighborhoodOffsetTable();
}

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
NeighbourIteratorCell<PixelType>
::~NeighbourIteratorCell()
{
	
}

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
void
NeighbourIteratorCell<PixelType>::ComputeNeighborhoodOffsetTable()
{
  TOffset o;
  unsigned int i, j;
  for (j = 0; j < DIM; j++)
    {
    o[j] = -(static_cast<long>(RADIUS));
    }

  for (i = 0; i < NEIGHBOURHOOD_SIZE; ++i)
    {
	  m_OffsetTable[i] = o;
    for (j= 0; j< DIM; j++)
      {
      o[j] = o[j] + 1;
      if (o[j] > static_cast<long>(RADIUS))
        {
        o[j] = -(static_cast<long>(RADIUS));
        }
      else break;
      }
    }
  
//  std::cout << "Offset table" << std::endl;
//  for(i=0; i<NEIGHBOURHOOD_SIZE; i++)
//  {
//	  std::cout << m_OffsetTable[i][0] << "," << m_OffsetTable[i][1] << "," << m_OffsetTable[i][2] << std::endl; 
//  }
} 

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
PixelType
NeighbourIteratorCell<PixelType>::GetPixel(uint32 pos, bool &isWithin)
{
	if(m_neighbourhood->IsWithinImage(m_neighbourhood->m_currIndex + m_OffsetTable[pos]) )
	{
		isWithin = true;
		return m_neighbourhood->GetPixel(pos);
	}
	else
	{
		isWithin = false;
		return OnBehindBoundary(m_OffsetTable[pos]);
	}
}

///////////////////////////////////////////////////////////////////////////////
template<typename PixelType>
PixelType
NeighbourIteratorCell<PixelType>
::OnBehindBoundary(const TOffset &off) const
{
	TOffset o = off;
	m_neighbourhood->HowMuchCrossesBoundary(o);
	        
	uint32 linear_index = NEIGHBOURHOOD_SIZE / 2;
	
	TStrides strides = m_neighbourhood->GetStrides();

  // Return the value of the pixel at the closest boundary point.
  for (uint32 i = 0; i < DIM; ++i)
    {
    linear_index += (off[i] + o[i]) * strides[i];
    }
  
  return m_neighbourhood->GetPixel(linear_index);
}

///////////////////////////////////////////////////////////////////////////////

}
}

#endif
