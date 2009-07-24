#ifndef BASE_H_
#define BASE_H_

#include "common/Common.h"
#include "common/Vector.h"

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

	InterpolatorBase() : m_image( NULL ), m_dataPointer( NULL )
	{}
	
	InterpolatorBase(const ImageType *image) : m_image(image) 
	{
		m_dataPointer = m_image->GetPointer(m_size, m_strides);
	}

	void SetImage(const ImageType *image)
        {
		m_image = image;
                m_dataPointer = m_image->GetPointer(m_size, m_strides);
        }

	virtual typename ImageType::Element Get(CoordType &coords)=0;
	
protected:
	const ImageType *m_image;
	typename ImageType::PointType m_strides;
	typename ImageType::SizeType m_size;
	typename ImageType::Element *m_dataPointer;
};

}
}

#endif /*BASE_H_*/
