/**
 * @ingroup imaging 
 * @author Szabolcs Grof
 * @file ImageRegistration.h 
 * @{ 
 **/

#ifndef IMAGE_REGISTRATION_H_
#define IMAGE_REGISTRATION_H_

#include "Imaging/filters/ImageTransform.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{

template< typename ElementType, uint32 dim >
class ImageRegistration
	: public ImageTransform< ElementType, dim >
{
public:
	typedef Image< ElementType, dim >				ImageType;
	typedef ImageTransform< ElementType, dim >		 	PredecessorType;

	struct Properties : public PredecessorType::Properties
	{
		Properties() {}

	};

	ImageRegistration( Properties  * prop );
	ImageRegistration();

protected:
	bool
	ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype );

private:
	GET_PROPERTIES_DEFINITION_MACRO;

};

	
} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

//include implementation
#include "Imaging/filters/ImageRegistration.tcc"

#endif /*IMAGE_REGISTRATION_H_*/

/** @} */

