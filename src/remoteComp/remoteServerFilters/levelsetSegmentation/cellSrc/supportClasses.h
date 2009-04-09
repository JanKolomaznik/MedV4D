#ifndef SUPPORTCLASSES_H_
#define SUPPORTCLASSES_H_

#include "SPE/commonTypes.h"

namespace itk {

template <class TIndex>
class SparseFieldLevelSetNode
{
public:
	TIndex               m_Value;
	SparseFieldLevelSetNode *Next;
	SparseFieldLevelSetNode *Previous;
};

template<typename TRadius, typename TOffset, uint8 Dim>
class SparseFieldCityBlockNeighborList
{
public:
	
#define SIZE (2 * Dim)

  const TRadius &GetRadius() const
    { return m_Radius; }
  
  const unsigned int &GetArrayIndex(unsigned int i) const
    { return m_ArrayIndex[i]; }

  const TOffset &GetNeighborhoodOffset(unsigned int i) const
    { return m_NeighborhoodOffset[i]; }

  const unsigned int &GetSize() const
    { return m_size; }

  int GetStride(unsigned int i)
    { return m_StrideTable[i]; }
  
  SparseFieldCityBlockNeighborList();
  ~SparseFieldCityBlockNeighborList() {}

  void Print(std::ostream &os) const;
  
private:
	uint32 m_size;
  TRadius                m_Radius;
  unsigned int m_ArrayIndex[SIZE];
  TOffset   m_NeighborhoodOffset[SIZE];

  /** An internal table for keeping track of stride lengths in a neighborhood,
      i.e. the memory offsets between pixels along each dimensional axis. */
  M4D::Cell::TStrides m_StrideTable;
};



}

//include implementation
#include "src/supportClasses.tcc"

#endif /*SUPPORTCLASSES_H_*/
