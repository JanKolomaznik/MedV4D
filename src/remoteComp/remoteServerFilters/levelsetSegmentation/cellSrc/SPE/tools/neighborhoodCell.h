#ifndef NEIGHBORHOODCELL_H_
#define NEIGHBORHOODCELL_H_

#include "DMAGate.h"
#include "../../supportClasses.h"

namespace M4D {
namespace Cell {

#define RADIUS 1
#define SIZEIN1DIM ((RADIUS * 2) + 1)
#define NEIGHBOURHOOD_SIZE (SIZEIN1DIM * SIZEIN1DIM * SIZEIN1DIM)

template<typename PixelType>
class NeighborhoodCell
{
public:
	
	typedef PixelType TPixel;
	typedef TImageProperties<PixelType> TImageProps;
	
	//ctor
	NeighborhoodCell();
	
	void SetPosition(const TIndex &pos);
	
	inline PixelType GetPixel(uint32 pos) { return m_buf[pos]; }
	
	void SetPixel(PixelType val, TOffset pos);
	void SetCenterPixel(PixelType val);
	inline PixelType *GetPixelPointer(uint32 pos) { return &m_buf[pos]; }
	
	TStrides GetStrides() { return m_radiusStrides; }
	uint32 GetStride(const uint32 axis)
	  {     return m_radiusStrides[axis];  }
	
	uint32 GetNeighborhoodIndex(const TOffset &) const;
	uint32 GetCenterNeighborhoodIndex() const
		{ return  static_cast<uint32>(m_size/2); }
	size_t GetSize() { return m_size; }
	
	void SetImageProperties(TImageProperties<PixelType> *props) { 
		m_imageProps = props;
		ComputeStridesFromSize<TSize, TStrides>(m_imageProps->region.size, m_imageStrides);
		}
	bool IsWithinImage(const TIndex &pos);
	TIndex m_currIndex;
	void PrintImage(std::ostream &s);
	
	void HowMuchCrossesBoundary(TOffset &howMuch);
	
protected:
	
	PixelType *ComputeImageDataPointer(const TIndex &pos);
	void LoadData(PixelType *src, PixelType *dest, size_t size);
	void LoadSlice(TIndex posm, uint8 dim, PixelType *dest);
	
	TStrides m_radiusStrides;
	TSize m_radiusSize;
	TImageProperties<PixelType> *m_imageProps;
	TStrides m_imageStrides;
	
	PixelType m_buf[NEIGHBOURHOOD_SIZE];
	size_t m_size;
};

template<typename PixelType>
std::ostream & operator<<(std::ostream &stream, NeighborhoodCell<PixelType> &n);

}  // namespace
} // namespace

//include implementation
#include "src/neighborhoodCell.tcc"

#endif /*NEIGHBORHOODCELL_H_*/
