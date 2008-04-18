#ifndef _IMAGE_FACTORY_H
#define _IMAGE_FACTORY_H

#include "ImageDataTemplate.h"

#include "dicomConn/M4DDICOMServiceProvider.h"

#include "M4DCommon.h"

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
	static AbstractImage::APtr 
	CreateEmptyImage2D( 
			size_t		width, 
			size_t		height
			);

	template< typename ElementType >
	static AbstractImage::APtr 
	CreateEmptyImage3D( 
			size_t		width, 
			size_t		height, 
			size_t		depth
			);

	static AbstractImage::APtr 
	CreateImageFromDICOM( M4DDicom::DcmProvider::DicomObjSetPtr dicomObjects );
protected:

private:
	/**
	 * Not implemented - cannot instantiate this class.
	 **/
	ImageFactory();

	/**
	 * @param pixelSize Enum, which sais how many bits is used per pixel.
	 * @param imageSize How many elements of size 'pixelSize' will be stored in array.
	 * @param dataArray Output parameter - allocated array is returned in this.
	 * @param elementSize Output parametr - sais how many bytes is used by one element.
	 **/
	static void
	PrepareElementArrayFromPixelSize( 
		M4DDicom::DcmProvider::DicomObj::PixelSize 	pixelSize, 
		size_t 						imageSize, 
		uint8						*& dataArray, 
		unsigned short 					&  elementSize 
		);

	/**
	 * @param pixelSize Enum, which sais how many bits is used per pixel.
	 * @param imageSize How many elements of size 'pixelSize' will be stored in array.
	 * @param dataArray Filled array of image elements.
	 * @param info Filled dimension info array.
	 **/
	static AbstractImage*
	CreateImageFromDataAndPixelType(
		M4DDicom::DcmProvider::DicomObj::PixelSize 	pixelSize, 
		size_t 						imageSize, 
		uint8						* dataArray, 
		DimensionInfo					* info
		);

};


} /*namespace Images*/
} /*namespace M4D*/

//Including template implementation
#include "ImageFactory.tcc"

#endif /*_IMAGE_FACTORY_H*/
