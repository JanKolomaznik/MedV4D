#ifndef _IMAGE_FACTORY_H
#define _IMAGE_FACTORY_H

#include "ImageDataTemplate.h"

//#include "dicomConn/M4DDICOMObject.h"


namespace M4D
{
namespace Images
{

class ImageFactory
{
public:
	/**
	 * Method for custom empty image creation.
	 **/
	template< typename ElementType >
	static AbstractImage::Ptr CreateEmptyImage2D( 
			size_t		width, 
			size_t		height
			);

	template< typename ElementType >
	static AbstractImage::Ptr CreateEmptyImage3D( 
			size_t		width, 
			size_t		height, 
			size_t		depth
			);
protected:

private:
	/**
	 * Not implemented - cannot instantiate this class.
	 **/
	ImageFactory();

};


} /*namespace Images*/
} /*namespace M4D*/

//Including template implementation
#include "ImageFactory.tcc"

#endif /*_IMAGE_FACTORY_H*/
