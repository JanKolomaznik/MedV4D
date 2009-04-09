#ifndef NEIGHBORHOODCELL_H_
#define NEIGHBORHOODCELL_H_

#include "commonTypes.h"

#if( defined(COMPILE_FOR_CELL) || defined(COMPILE_ON_CELL) )
#include <spu_mfcio.h>
#endif

namespace M4D {
namespace Cell {

#define RADIUS 1
#define SIZEIN1DIM ((RADIUS * 2) + 1)
#define NEIGHBOURHOOD_SIZE (SIZEIN1DIM * SIZEIN1DIM * SIZEIN1DIM)

class NeighborhoodCell
{
public:
	
	//ctor
	NeighborhoodCell(TImageProperties *props);	
	void SetPosition(const TIndex &pos);
	
	inline TPixelValue GetPixel(uint32 pos) { return m_buf[pos]; }
	inline TPixelValue *GetPixelPointer(uint32 pos) { return &m_buf[pos]; }
	
	TStrides GetStrides() { return m_radiusStrides; }
	uint32 GetStride(const uint32 axis)
	  {     return m_radiusStrides[axis];  }
	
	uint32 GetNeighborhoodIndex(const TOffset &) const;
	uint32 GetCenterNeighborhoodIndex() const
		{ return  static_cast<uint32>(m_size/2); }
	size_t GetSize() { return m_size; }
	
	void SetImageProperties(TImageProperties *props) { m_imageProps = props; }
	
	//void Print(std::ostream &stream);
protected:
	void ComputeStridesFromSize(const TSize &size, TStrides &strides);
	
	TPixelValue *ComputeImageDataPointer(const TIndex &pos);
	void LoadData(TPixelValue *src, TPixelValue *dest, size_t size);
	void LoadSlice(TIndex posm, uint8 dim, TPixelValue *dest);
	bool IsWithinImage(const TIndex &pos);
	
	TStrides m_radiusStrides;
	TSize m_radiusSize;
	
	TIndex m_currIndex;
	
	TImageProperties *m_imageProps;
	TStrides m_imageStrides;
	
	TPixelValue m_buf[NEIGHBOURHOOD_SIZE];
	size_t m_size;
};

}  // namespace
} // namespace

#endif /*NEIGHBORHOODCELL_H_*/
