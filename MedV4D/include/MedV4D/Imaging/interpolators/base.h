#ifndef BASE_H_
#define BASE_H_

#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/Vector.h"

namespace M4D
{
namespace Imaging
{

/**
 * Abstract base class for interpolators
 */
template< typename ImageType >
class InterpolatorBase
{
public:
	typedef Vector<float32, ImageType::Dimension> CoordType;
	typedef typename ImageType::Element ElementType;

	/**
         * Constructor
         */
	InterpolatorBase() : m_image( NULL ), m_dataPointer( NULL )
	{}

	/**
         * Constructor
         *  @param image pointer to the image according to which interpolation is required
         */
	InterpolatorBase(const ImageType *image) : m_image(image) 
	{
		m_dataPointer = m_image->GetPointer(m_size, m_strides);
	}

	/**
         * Set the interpolated image
         *  @param image pointer to the image according to which interpolation is required
         */
	void SetImage(const ImageType *image)
        {
		m_image = image;
                m_dataPointer = m_image->GetPointer(m_size, m_strides);
        }

	/**
         * Get the interpolated value
         *  @param coord the coordinates where to calculate the interpolated value
	 *  @return the interpolated value
         */
	virtual typename ImageType::Element Get(CoordType &coords)=0;
	
protected:

	/**
	 * Interpolated image
	 */
	const ImageType *m_image;

	/**
	 * Interpolated image strides
	 */
	typename ImageType::PointType m_strides;

	/**
	 * Interpolated image sizes
	 */
	typename ImageType::SizeType m_size;

	/**
	 * Pointer to the voxel values of the interpolated image
	 */
	typename ImageType::Element *m_dataPointer;
};

}
}

#endif /*BASE_H_*/
