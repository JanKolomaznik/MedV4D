#ifndef NEIGHBOURHOODITERATOR_H_
#define NEIGHBOURHOODITERATOR_H_

#include "neighborhoodCell.h"
#include "common/Debug.h"

namespace M4D
{
namespace Cell
{

template<typename PixelType> class NeighbourIteratorCell
{
public:

	/** Standard class typedefs. */
	typedef NeighbourIteratorCell Self;
	typedef NeighborhoodCell<PixelType> NeighborhoodType;

	NeighbourIteratorCell();
	NeighbourIteratorCell(NeighborhoodType *neiborhood);
	~NeighbourIteratorCell();

	NeighborhoodType &GetNeighborhood() const
	{
		return *m_neighbourhood;
	}

	inline void SetNeighbourhood(NeighborhoodType *neiborhood)
	{
		m_neighbourhood = neiborhood;
		ComputeNeighborhoodOffsetTable();
	}

	void SetCenterPixel(PixelType val)
	{
		m_neighbourhood->SetCenterPixel(val);
	}

	void SetPixel(TOffset pos, PixelType val)
	{
		m_neighbourhood->SetPixel(val, pos);
		DOUT << "Setting pixel " << pos[0] << "," << pos[1] << "," << pos[2] << " = " << val << std::endl;
	}

	void SetPixel(uint32 idx, PixelType val)
	{
		TOffset o = m_OffsetTable[idx];
		TIndex soucet = m_neighbourhood->m_currIndex + o;
		m_neighbourhood->SetPixel(val, o);
		DOUT << "Setting pixel 2" << soucet[0] << "," << soucet[1] << "," << soucet[2] << " = " << val << std::endl;
	}

	/** Returns the pointer to the center pixel of the neighborhood. */
	const PixelType *GetCenterPointer() const
	{
		return m_neighbourhood->GetPixelPointer(m_neighbourhood->GetSize()>>1);
	}

	/** Returns the pixel referenced at the center of the 
	 *  ConstNeighborhoodIterator. */
	PixelType GetCenterPixel() const
	{
		return m_neighbourhood->GetPixel(m_neighbourhood->GetCenterNeighborhoodIndex() );
	}

	PixelType GetPixel(const unsigned i) const
	{
		return m_neighbourhood->GetPixel(i);
	}
	PixelType GetPixel(const TOffset &o) const
	{
		return m_neighbourhood->GetPixel(m_neighbourhood->GetNeighborhoodIndex(o) );
	}

	PixelType GetPixel(uint32 pos, bool &isWithin);
	PixelType OnBehindBoundary(const TOffset &off);

	/** Returns the pixel value located i pixels distant from the neighborhood 
	 *  center in the positive specified ``axis'' direction. No bounds checking 
	 *  is done on the size of the neighborhood. */
	PixelType GetNext(const unsigned axis, const unsigned i) const
	{
		return (GetPixel(m_neighbourhood->GetCenterNeighborhoodIndex() + (i
				* m_neighbourhood->GetStride(axis))));
	}

	/** Returns the pixel value located one pixel distant from the neighborhood
	 *  center in the specifed positive axis direction. No bounds checking is 
	 *  done on the size of the neighborhood. */
	PixelType GetNext(const unsigned axis) const
	{
		return (GetPixel(m_neighbourhood->GetCenterNeighborhoodIndex()
				+ m_neighbourhood->GetStride(axis)));
	}

	/** Returns the pixel value located i pixels distant from the neighborhood 
	 *  center in the negative specified ``axis'' direction. No bounds checking 
	 *  is done on the size of the neighborhood. */
	PixelType GetPrevious(const unsigned axis, const unsigned i) const
	{
		return (GetPixel(m_neighbourhood->GetCenterNeighborhoodIndex() - (i
				* m_neighbourhood->GetStride(axis))));
	}

	/** Returns the pixel value located one pixel distant from the neighborhood 
	 *  center in the specifed negative axis direction. No bounds checking is 
	 *  done on the size of the neighborhood. */
	PixelType GetPrevious(const unsigned axis) const
	{
		return (GetPixel(m_neighbourhood->GetCenterNeighborhoodIndex()
				- m_neighbourhood->GetStride(axis)));
	}

	/** Returns the N-dimensional index of the iterator's position in
	 * the image. */
	const TIndex& GetIndex(void) const
	{
		return m_neighbourhood->m_currIndex;
	}

	void SetLocation(const TIndex& position)
	{
		m_neighbourhood->SetPosition(position);
	}

	TOffset GetOffset(unsigned int i) const
	{
		return m_OffsetTable[i];
	}

protected:

	void ComputeNeighborhoodOffsetTable();

	TOffset m_OffsetTable[NEIGHBOURHOOD_SIZE];
	NeighborhoodType *m_neighbourhood;
};

}
} // namespace

//include implementation
#include "src/neighbourhoodIterator.tcc"

#endif /*NEIGHBOURHOODITERATOR_H_*/
