#ifndef SUPPORTCLASSES_H_
#define SUPPORTCLASSES_H_

namespace itk {

template <class TNeighborhoodType>
class SparseFieldCityBlockNeighborList
{
public:
  typedef TNeighborhoodType                     NeighborhoodType;
  typedef typename NeighborhoodType::OffsetType OffsetType;
  typedef typename NeighborhoodType::RadiusType RadiusType;
  itkStaticConstMacro(Dimension, unsigned int,
                      NeighborhoodType::Dimension );

  const RadiusType &GetRadius() const
    { return m_Radius; }
  
  const unsigned int &GetArrayIndex(unsigned int i) const
    { return m_ArrayIndex[i]; }

  const OffsetType &GetNeighborhoodOffset(unsigned int i) const
    { return m_NeighborhoodOffset[i]; }

  const unsigned int &GetSize() const
    { return m_Size; }

  int GetStride(unsigned int i)
    { return m_StrideTable[i]; }
  
  SparseFieldCityBlockNeighborList();
  ~SparseFieldCityBlockNeighborList() {}

  void Print(std::ostream &os) const;
  
private:
  unsigned int              m_Size;
  RadiusType                m_Radius;
  std::vector<unsigned int> m_ArrayIndex;
  std::vector<OffsetType>   m_NeighborhoodOffset;

  /** An internal table for keeping track of stride lengths in a neighborhood,
      i.e. the memory offsets between pixels along each dimensional axis. */
  unsigned m_StrideTable[Dimension];
};



}

//include implementation
#include "src/supportClasses.tcc"

#endif /*SUPPORTCLASSES_H_*/
