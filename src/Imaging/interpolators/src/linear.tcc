
#ifndef LINEAR_H_
#error File trilinear.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{

///////////////////////////////////////////////////////////////////////////////

template< typename ImageType >
typename ImageType::Element
LinearInterpolator<ImageType>
	::Get(CoordType &coords)
{

	typename ImageType::PointType strides;
	double temp;

	double cx = 0;
	strides[0] = 0;
	if ( coords.Dimension > 0 )
	{
		cx = std::modf((double)coords[0],&temp);
		strides[0] = this->m_strides[0];
	}

	double cy = 0;
	strides[1] = 0;
	if ( coords.Dimension > 1 )
	{
		cy = std::modf((double)coords[1],&temp);
		strides[1] = this->m_strides[1];
	}

	double cz = 0;
	strides[2] = 0;
	if ( coords.Dimension > 2 )
	{
		cz = std::modf((double)coords[2],&temp);
		strides[2] = this->m_strides[2];
	}

	double w[8];

	w[0] = (1 - cx) * (1 - cy) * (1 - cz);
	w[1] = cx * (1 - cy) * (1 - cz);
	w[2] = (1 - cx) * cy * (1 - cz);
	w[3] = (1 - cx) * (1 - cy) * cz;
	w[4] = cx * (1 - cy) * cz;
	w[5] = (1 - cx) * cy * cz;
	w[6] = cx * cy * (1 - cz);
	w[7] = cx * cy * cz;

	return  (typename ImageType::Element)(
		*(this->m_dataPointer + (
			((uint32)std::ceil(coords[0]) * strides[0]) + 
			((uint32)std::ceil(coords[1]) * strides[1]) +
			((uint32)std::ceil(coords[2]) * strides[2])
			) ) * w[0]

		+

		*(this->m_dataPointer + (
			((uint32)std::floor(coords[0]) * strides[0]) + 
			((uint32)std::ceil(coords[1]) * strides[1]) +
			((uint32)std::ceil(coords[2]) * strides[2])
			) ) * w[1]

		+

		*(this->m_dataPointer + (
			((uint32)std::ceil(coords[0]) * strides[0]) + 
			((uint32)std::floor(coords[1]) * strides[1]) +
			((uint32)std::ceil(coords[2]) * strides[2])
			) ) * w[2]

		+

		*(this->m_dataPointer + (
			((uint32)std::ceil(coords[0]) * strides[0]) + 
			((uint32)std::ceil(coords[1]) * strides[1]) +
			((uint32)std::floor(coords[2]) * strides[2])
			) ) * w[3]

		+

		*(this->m_dataPointer + (
			((uint32)std::floor(coords[0]) * strides[0]) + 
			((uint32)std::ceil(coords[1]) * strides[1]) +
			((uint32)std::floor(coords[2]) * strides[2])
			) ) * w[4]

		+

		*(this->m_dataPointer + (
			((uint32)std::ceil(coords[0]) * strides[0]) + 
			((uint32)std::floor(coords[1]) * strides[1]) +
			((uint32)std::floor(coords[2]) * strides[2])
			) ) * w[5]

		+

		*(this->m_dataPointer + (
			((uint32)std::floor(coords[0]) * strides[0]) + 
			((uint32)std::floor(coords[1]) * strides[1]) +
			((uint32)std::ceil(coords[2]) * strides[2])
			) ) * w[6]

		+

		*(this->m_dataPointer + (
			((uint32)std::floor(coords[0]) * strides[0]) + 
			((uint32)std::floor(coords[1]) * strides[1]) +
			((uint32)std::floor(coords[2]) * strides[2])
			) ) * w[7]

		);

		
}

///////////////////////////////////////////////////////////////////////////////

} /*namespace Imaging*/
} /*namespace M4D*/


#endif
