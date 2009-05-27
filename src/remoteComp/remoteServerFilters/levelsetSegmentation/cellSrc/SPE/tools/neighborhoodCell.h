#ifndef NEIGHBORHOODCELL_H_
#define NEIGHBORHOODCELL_H_

#include "DMAGate.h"
#include "../../supportClasses.h"

namespace M4D 
{
namespace Cell 
{

#define RADIUS 1
#define SIZEIN1DIM ((RADIUS * 2) + 1)
#define NEIGHBOURHOOD_SLICE_SIZE (SIZEIN1DIM * SIZEIN1DIM)
#define NEIGHBOURHOOD_SIZE (NEIGHBOURHOOD_SLICE_SIZE * SIZEIN1DIM)

template<typename PixelType>
class NeighborhoodCell
{
public:
	
	typedef PixelType TPixel;
	typedef TImageProperties<PixelType> TImageProps;
	
	//ctor
	NeighborhoodCell();
	
	void SetPosition(const TIndex &pos);
	
	inline PixelType GetPixel(uint32 pos) { return m_buf[traslationTable_[pos]]; }
	
	void SetPixel(PixelType val, TOffset pos);
	void SetCenterPixel(PixelType val);
	inline PixelType *GetPixelPointer(uint32 pos) { return &m_buf[traslationTable_[pos]]; }
	
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
		//ComputeAlignStrides();
		}
	bool IsWithinImage(const TIndex &pos);
	TIndex m_currIndex;
	void PrintImage(std::ostream &s);
	
	void HowMuchCrossesBoundary(TOffset &howMuch);
	
	void SaveChanges();
	
protected:
	
	Address ComputeImageDataPointer(const TIndex &pos);
	void LoadData(PixelType *src, PixelType *dest, size_t size);
	void LoadSlice(TIndex posm, uint8 dim, PixelType *dest);
	
	TStrides m_radiusStrides;
	TSize m_radiusSize;
	TImageProperties<PixelType> *m_imageProps;
	TStrides m_imageStrides;
	
	int32 traslationTable_[NEIGHBOURHOOD_SIZE];
	int8 transIdxIter_;
	
	uint32 _dirtyElems;
	
#define DMA_LIST_SET_SIZE (SIZEIN1DIM*SIZEIN1DIM)	// maximal count
#define LIST_SET_NUM (16 / sizeof(PixelType))
#define BUFFER_SIZE (DMA_LIST_SET_SIZE * LIST_SET_NUM)
	
	PixelType m_buf[BUFFER_SIZE] __attribute__ ((aligned (128)));
	size_t m_size;
	
	TOffset OffsetFromPos(uint32 pos);
	
#ifdef FOR_CELL
	/* here we reserve space for the dma list.
	 * This array is aligned on 16 byte boundary */	
	mfc_list_element_t dma_list[LIST_SET_NUM][DMA_LIST_SET_SIZE] __attribute__ ((aligned (16)));
	
	uint32 _dmaListIter[LIST_SET_NUM];
	
	void PutIntoList(uint64 address, uint32 size);
#endif
	
private:
	//void ComputeAlignStrides();
};

template<typename PixelType>
std::ostream & operator<<(std::ostream &stream, NeighborhoodCell<PixelType> &n);

}  // namespace
} // namespace

//include implementation
#include "src/neighborhoodCell.tcc"

#endif /*NEIGHBORHOODCELL_H_*/
