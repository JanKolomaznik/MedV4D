
#ifndef NEARESTNEIGHBOR_H_
#error File nearestNeighbour.tcc cannot be included directly!
#else

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

