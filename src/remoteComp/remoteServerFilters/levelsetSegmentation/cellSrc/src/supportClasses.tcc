

#ifndef SUPPORTCLASSES_H_
#error File supportClasses.tcc cannot be included directly!
#else

namespace itk {


template <class TNeighborhoodType>
SparseFieldCityBlockNeighborList<TNeighborhoodType>
::SparseFieldCityBlockNeighborList()
{
  typedef typename NeighborhoodType::ImageType ImageType;
  typename ImageType::Pointer dummy_image = ImageType::New();

  unsigned int i, nCenter;
  int d;
  OffsetType zero_offset;
    
  for (i = 0; i < Dimension; ++i)
    {
    m_Radius[i] = 1;
    zero_offset[i] = 0;
    }
  NeighborhoodType it(m_Radius, dummy_image, dummy_image->GetRequestedRegion());
  nCenter = it.Size() / 2;

  m_Size = 2 * Dimension;
  m_ArrayIndex.reserve(m_Size);
  m_NeighborhoodOffset.reserve(m_Size);

  for (i = 0; i < m_Size; ++i)
    {
    m_NeighborhoodOffset.push_back(zero_offset);
    }

  for (d = Dimension - 1, i = 0; d >= 0; --d, ++i)
    {
    m_ArrayIndex.push_back( nCenter - it.GetStride(d) );
    m_NeighborhoodOffset[i][d] = -1;
    }
  for (d = 0; d < static_cast<int>(Dimension); ++d, ++i)
    {
    m_ArrayIndex.push_back( nCenter + it.GetStride(d) );
    m_NeighborhoodOffset[i][d] = 1;
    }

  for (i = 0; i < Dimension; ++i)
    {
    m_StrideTable[i] = it.GetStride(i);
    }
}

template <class TNeighborhoodType>
void
SparseFieldCityBlockNeighborList<TNeighborhoodType>
::Print(std::ostream &os) const
{
  os << "SparseFieldCityBlockNeighborList: " << std::endl;
  for (unsigned i = 0; i < this->GetSize(); ++i)
    {
    os << "m_ArrayIndex[" << i << "]: " << m_ArrayIndex[i] << std::endl;
    os << "m_NeighborhoodOffset[" << i << "]: " << m_NeighborhoodOffset[i]
       << std::endl;
    }
}

}