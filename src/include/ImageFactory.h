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
	class EWrongPointer;
	class EEmptyDicomObjSet;
	class EWrongArrayForFlush;
	class EUnknowDataType;


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
	 * @exception ImageFactory::EWrongPointer Thrown when passed pointer isn't valid.
	 * @exception ImageFactory::EEmptyDicomObjSet Thrown when empty set passed.
	 **/
	static AbstractImage::APtr 
	CreateImageFromDICOM( Dicom::DcmProvider::DicomObjSetPtr dicomObjects );
protected:

private:
	/**
	 * Not implemented - cannot instantiate this class.
	 **/
	ImageFactory();
	ImageFactory( const ImageFactory& );
	void operator=( const ImageFactory& );

	/**
	 * @param typeId Id of type used for storing pixel value.
	 * @param imageSize How many elements of size 'pixelSize' will be stored in array.
	 * @param dataArray Output parameter - allocated array is returned in this.
	 **/
	static void
	PrepareElementArrayFromTypeID( 
		int 			typeId, 
		size_t 			imageSize, 
		uint8			*& dataArray  
		);

	/**
	 * @param typeId Id of type used for storing pixel value.
	 * @param imageSize How many elements of size 'pixelSize' will be stored in array.
	 * @param dataArray Filled array of image elements.
	 * @param info Filled dimension info array.
	 **/
	static AbstractImage*
	CreateImageFromDataAndTypeID(
		int 			typeId,
		size_t 			imageSize, 
		uint8			* dataArray, 
		DimensionInfo		* info
		);

	//TODO - make this function asynchronous. Add locking of array in image.
	/**
	 * @param pixelSize How many bytes is used per pixel.
	 * @param imageSize How many elements of size 'pixelSize' can be stored in array.
 	 * @param stride Number of bytes used per one object flush (size of one layer in bytes).
	 * @param dataArray Array to be filled from dicom objects. Must be allocated!!!
	 * @exception EWrongArrayForFlush Thrown when NULL array passed, or imageSize is less than
	 * space needed for flushing all dicom objects.
	 **/
	static void
	FlushDicomObjects(
		Dicom::DcmProvider::DicomObjSetPtr dicomObjects,
		uint8 			pixelSize, 
		size_t 			imageSize,
		size_t			stride,
		uint8			* dataArray
		);
	

};

/**
 * Exception class, which is thrown from CreateImageFromDICOM(), when
 * wrong pointer was passed to function.
 **/
 class ImageFactory::EWrongPointer
{
public:
	EWrongPointer(){}
//TODO
};

/**
 * Exception class, which is thrown from CreateImageFromDICOM(), when
 * empty set of DICOM objects was passed to function.
 **/
class ImageFactory::EEmptyDicomObjSet
{
public:
	EEmptyDicomObjSet(){}
//TODO
};

/**
 * Exception class, which is thrown from FlushDicomObjects(), when
 * array passed to function is wrong - NULL or too small.
 **/
class ImageFactory::EWrongArrayForFlush
{
public:
	EWrongArrayForFlush(){}
//TODO
};

/**
 * Exception class, which is thrown from CreateImageFromDICOM(), when
 * no type exists for information obtained from dicom object.
 **/
class ImageFactory::EUnknowDataType
{
public:
	EUnknowDataType( uint8 size, bool sign ){}
//TODO
};

/**
 * Exception class, which is thrown from PrepareElementArray<>(), when
 * array couldn't be allocated ( not enough memory ).
 **/
class EFailedArrayAllocation
{
public:
	EFailedArrayAllocation(){}
//TODO
};

} /*namespace Images*/
} /*namespace M4D*/

//Including template implementation
#include "ImageFactory.tcc"

#endif /*_IMAGE_FACTORY_H*/
