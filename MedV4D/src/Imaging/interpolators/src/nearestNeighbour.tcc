
#ifndef NEARESTNEIGHBOR_H_
#error File nearestNeighbour.tcc cannot be included directly!
#else

#if WIN32
#define round(X)	floor((X) + 0.5)
#endif

namespace M4D
{
namespace Imaging
{

///////////////////////////////////////////////////////////////////////////////

template< typename ImageType >
typename ImageType::Element
NearestNeighborInterpolator<ImageType>
	::Get(CoordType &coords)
{

	// calculate the interpolated value and return
	return *(this->m_dataPointer + (
			((uint32)round(coords[0]) * this->m_strides[0]) + 
			((uint32)round(coords[1]) * this->m_strides[1]) +
			((uint32)round(coords[2]) * this->m_strides[2])
			) );
}

///////////////////////////////////////////////////////////////////////////////

} /*namespace Imaging*/
} /*namespace M4D*/


#endif

