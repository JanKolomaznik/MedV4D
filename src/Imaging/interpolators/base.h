#ifndef BASE_H_
#define BASE_H_

#include "Vector.h"

namespace M4D
{
namespace Imaging
{

template< typename ImageType >
class InterpolatorBase
{
public:
	typedef Vector<float32, ImageType::Dimension> CoordType;
	typedef typename ImageType::Element ElementType;
	
	InterpolatorBase(const ImageType *image) : m_image(image) 
	{
		m_dataPointer = m_image->GetPointer(m_size, m_strides);
	}
	
protected:
	const ImageType *m_image;
	typename ImageType::PointType m_strides;
	typename ImageType::SizeType m_size;
	typename ImageType::Element *m_dataPointer;
};

}
}

#endif /*BASE_H_*/
