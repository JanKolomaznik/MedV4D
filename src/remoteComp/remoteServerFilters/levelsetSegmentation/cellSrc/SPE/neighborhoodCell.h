#ifndef NEIGHBORHOODCELL_H_
#define NEIGHBORHOODCELL_H_

#include "common/Vector.h"

namespace M4D {
namespace Cell {

template< typename TPixel, uint8 Dim>
class NeighborhoodCell
{
public:
	typedef Vector<uint32, Dim> RadiusType;
	typedef Vector<uint32, Dim> TSize;
	typedef Vector<int32, Dim> TOffset;
	typedef Vector<uint32, Dim> TStrides;
	typedef Vector<uint32, Dim> TIndex;
	typedef Vector<float32, Dim> ContinuousIndexType;
	typedef Vector<float32, Dim> TSpacing;
	
	struct TRegion
	{
		typedef TIndex OffsetType;
		typedef TSize SizeType;
		TIndex offset;
		TSize size;
		TRegion() {}
		TRegion(TIndex offset_, TSize size_) : offset(offset_), size(size_) {}
	};
	struct TImageProperties
	{
		typedef TRegion RegionType;
		typedef TSpacing SpacingType;
		TRegion region;
		SpacingType spacing;
		TPixel *imageData;
		TImageProperties() {}
		TImageProperties(TRegion region_, TPixel *data_) 
			: region(region_), imageData(data_) {}
	};
	
	//ctor
	NeighborhoodCell(const RadiusType &radius, TImageProperties *props);
	
	
	void SetPosition(const TIndex &pos);
	
	inline const TPixel GetPixel(uint32 pos) { return m_buf[pos]; }
	inline TPixel *GetPixelPointer(uint32 pos) { return &m_buf[pos]; }
	
	TStrides GetStrides() { return m_radiusStrides; }
	uint32 GetStride(const uint32 axis) const
	  {     return m_radiusStrides[axis];  }
	
	const uint32 GetNeighborhoodIndex(const TOffset &) const;
	const uint32 GetCenterNeighborhoodIndex() const
		{ return  static_cast<uint32>(m_size/2); }
	size_t GetSize() { return m_size; }
	
	void SetImageProperties(TImageProperties *props) { m_imageProps = props; }
	
	void Print(std::ostream &stream);
protected:
	void ComputeStridesFromSize(const TSize &size, TStrides &strides);
	
	TPixel *ComputeImageDataPointer(const TIndex &pos);
	void LoadData(TPixel *src, TPixel *dest, size_t size);
	void LoadSlice(TIndex posm, uint8 dim, TPixel *dest);
	bool IsWithinImage(const TIndex &pos);
	
	RadiusType m_radius;
	TStrides m_radiusStrides;
	TSize m_radiusSize;
	
	TIndex m_currIndex;
	
	TImageProperties *m_imageProps;
	TStrides m_imageStrides;
	
	TPixel *m_buf;
	size_t m_size;
};

//include implementation
#include "src/neighborhoodCell.tcc"

}  // namespace
} // namespace

#endif /*NEIGHBORHOODCELL_H_*/
