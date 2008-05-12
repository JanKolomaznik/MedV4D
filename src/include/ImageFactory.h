#ifndef _IMAGE_FACTORY_H
#define _IMAGE_FACTORY_H

#include "AbstractImage.h"
#include "ImageDataTemplate.h"

#include "dicomConn/M4DDICOMServiceProvider.h"

#include "Common.h"

namespace M4D
{
namespace Images
{

/**
 * Factory class which takes care of allocation of image 
 * datasets. It is able to create empty image, or fill 
 * it with DICOM data.
 **/
class ImageFactory
{
public:
	/**
	 * Method for custom empty 2D image creation.
	 * @param ElementType Type of elements which will be contained 
	 * in created dataset.
	 * @param width Width of desided image.
	 * @param height Height of desided image.
	 * @return Smart pointer to abstract ancestor of created image.
	 **/
	template< typename ElementType >
	static AbstractImage::APtr 
	CreateEmptyImage2D( 
			size_t		width, 
			size_t		height
			);

	/**
	 * Method for custom empty 2D image creation. Difference from 
	 * previous method is in return value.
	 * @param ElementType Type of elements which will be contained 
	 * in created dataset.
	 * @param width Width of desided image.
	 * @param height Height of desided image.
	 * @return Smart pointer to created image.
	 **/
	template< typename ElementType >
	static typename ImageDataTemplate< ElementType >::Ptr
	CreateEmptyImage2DTyped( 
			size_t		width, 
			size_t		height
			);

	/**
	 * Method for custom empty 3D image creation.
	 * @param ElementType Type of elements which will be contained 
	 * in created dataset.
	 * @param width Width of desided image.
	 * @param height Height of desided image.
	 * @param depth Depth of desided image.
	 * @return Smart pointer to abstract ancestor of created image.
	 **/
	template< typename ElementType >
	static AbstractImage::APtr 
	CreateEmptyImage3D( 
			size_t		width, 
			size_t		height, 
			size_t		depth
			);

	/**
	 * Method for custom empty 3D image creation. Difference from 
	 * previous method is in return value.
	 * @param ElementType Type of elements which will be contained 
	 * in created dataset.
	 * @param width Width of desided image.
	 * @param height Height of desided image.
	 * @param depth Depth of desided image.
	 * @return Smart pointer to created image.
	 **/
	template< typename ElementType >
	static typename ImageDataTemplate< ElementType >::Ptr 
	CreateEmptyImage3DTyped( 
			size_t		width, 
			size_t		height, 
			size_t		depth
			);


	/**
	 * Creates image from given dicomObject set.
	 * @param dicomObjects Given set of dicom objects.
	 * @return Smart pointer to created image.
	 **/
	static AbstractImage::APtr 
	CreateImageFromDICOM( Dicom::DcmProvider::DicomObjSetPtr dicomObjects );
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
		Dicom::DcmProvider::DicomObj::PixelSize 	pixelSize, 
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
		Dicom::DcmProvider::DicomObj::PixelSize 	pixelSize, 
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
