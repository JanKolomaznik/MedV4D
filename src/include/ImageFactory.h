#ifndef _IMAGE_FACTORY_H
#define _IMAGE_FACTORY_H

#include "ImageDataTemplate.h"

#include "dicomConn/M4DDICOMObject.h"


namespace Images
{

class ImageFactory
{
public:
	/**
	 * Method for custom empty image creation.
	 **/
	static AbstractImage::Ptr CreateEmptyImage( ... );
protected:

private:
	/**
	 * Not implemented - cannot instantiate this class.
	 **/
	ImageFactory();

};


} /*namespace Images*/

#endif /*_IMAGE_FACTORY_H*/
