#ifndef SUPPORTCLASSES_H_
#define SUPPORTCLASSES_H_

#include "SPE/commonTypes.h"

namespace M4D
{
namespace Cell
{

///////////////////////////////////////////////////////////////////////////////
template<typename SizeType, typename StridesType>
void 
ComputeStridesFromSize(const SizeType &size, StridesType &strides)
{
  unsigned int accum;

  accum = 1;
  strides[0] = 1;
  for (unsigned int dim = 1; dim < DIM; ++dim)
  {
	  accum *= size[dim-1];
	  strides[dim] = accum;
  }
}

///////////////////////////////////////////////////////////////////////////////

// support functions
template<typename ImageType, typename RegionType>
RegionType ConvertRegion(const ImageType &image)
{
	// convert values
	typename ImageType::RegionType imageRegion;

	imageRegion = image.GetLargestPossibleRegion();
	RegionType reg;

	for(uint8 i=0; i<ImageType::ImageDimension; i++)
	{
		reg.offset[i] = imageRegion.GetIndex()[i];
		reg.size[i] = imageRegion.GetSize()[i];
	}

	return reg;
}

template<typename ImageType>
void 
PrintITKImage(const ImageType &image, std::ostream &s)
{
	//image.Print( s);
	    
	typename ImageType::RegionType::IndexType index;
	typename ImageType::RegionType::SizeType size = 
    	image.GetLargestPossibleRegion().GetSize();
	
	typename ImageType::PixelType pixel;
    
    s << "size: " << size[0] << "," << size[1] << "," << size[2] << std::endl;
    
    for( unsigned int i=0; i<size[0]; i++)
    {
    	for( unsigned int j=0; j<size[1]; j++)
    	{
    		for( unsigned int k=0; k< size[2]; k++)
    		{
    			index[0] = i;
    			index[1] = j;
    			index[2] = k;
    			
    			s << "[" << i << "," << j << "," << k << "]= ";
    			pixel = image.GetPixel(index);
    			s << ((int32) pixel) << std::endl;
    		}
    	}
    }
}

template<typename T1, typename T2>
T1 ConvertIncompatibleVectors(const T2 &v2)
{
	T1 tmp;
	for(uint32 i=0; i<DIM; i++)
		tmp[i] = v2[i];

	return tmp;
}

///////////////////////////////////////////////////////////////////////////////

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
  TStrides m_StrideTable;
};


}
}

//include implementation
#include "src/supportClasses.tcc"

#endif /*SUPPORTCLASSES_H_*/
