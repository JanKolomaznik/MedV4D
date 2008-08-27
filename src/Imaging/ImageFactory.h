/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ImageFactory.h 
 * @{ 
 **/

#ifndef _IMAGE_FACTORY_H
#define _IMAGE_FACTORY_H

#include "Imaging/AbstractImageData.h"
#include "Imaging/ImageDataTemplate.h"
#include "Imaging/Image.h"

#include "dicomConn/DICOMServiceProvider.h"

#include "Common.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

//TODO check comments

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
	class EWrongDICOMObjIndex;

	/**
	 * Create image according to passed information.
	 * \param dim Dimesnion of desired image.
	 * \param minimums Minimum coordinates of elements in image (for each dimension).
	 * \param maximums Coordinates of first element out of image (for each dimension).
	 * \param elementExtents Proportion of elements in each dimension
	 * @return Smart pointer to abstract ancestor of created image.
	 **/
	template< typename ElementType >
	static AbstractImage::AImagePtr 
	CreateEmptyImageFromExtents( 
			uint32		dim,
			int32		minimums[], 
			int32		maximums[],
			float32		elementExtents[]
			);

	/**
	 * Method for custom empty 2D image creation.
	 * ElementType is type of elements which will be contained 
	 * in created dataset.
	 * @param width Width of desided image.
	 * @param height Height of desided image.
	 * @param elementWidth Width of each element.
	 * @param elementHeight Height of each element.
	 * @return Smart pointer to abstract ancestor of created image.
	 **/
	template< typename ElementType >
	static AbstractImage::AImagePtr 
	CreateEmptyImage2D( 
			uint32		width, 
			uint32		height,
			float32		elementWidth = 1.0,
			float32		elementHeight = 1.0
			);

	/**
	 * Method for custom empty 2D image creation. Difference from 
	 * previous method is in return value.
	 * ElementType is type of elements which will be contained 
	 * in created dataset.
	 * @param width Width of desided image.
	 * @param height Height of desided image.
	 * @param elementWidth Width of each element.
	 * @param elementHeight Height of each element.
	 * @return Smart pointer to created image.
	 **/
	template< typename ElementType >
	static typename Image< ElementType, 2 >::Ptr
	CreateEmptyImage2DTyped( 
			uint32		width, 
			uint32		height,
			float32		elementWidth = 1.0,
			float32		elementHeight = 1.0
			);

	template< typename ElementType >
	static void
	ReallocateImage2DData(
			Image< ElementType, 2 >	&image,
			uint32			width, 
			uint32			height,
			float32			elementWidth = 1.0,
			float32			elementHeight = 1.0
			);

	/**
	 * Method for custom empty 3D image creation.
	 * ElementType is type of elements which will be contained 
	 * in created dataset.
	 * @param width Width of desided image.
	 * @param height Height of desided image.
	 * @param depth Depth of desided image.
	 * @param elementWidth Width of each element.
	 * @param elementHeight Height of each element.
	 * @param elementDepth Depth of each element.
	 * @return Smart pointer to abstract ancestor of created image.
	 **/
	template< typename ElementType >
	static AbstractImage::AImagePtr 
	CreateEmptyImage3D( 
			uint32		width, 
			uint32		height, 
			uint32		depth,
			float32		elementWidth = 1.0,
			float32		elementHeight = 1.0,
			float32		elementDepth = 1.0
			);

	/**
	 * Method for custom empty 3D image creation. Difference from 
	 * previous method is in return value.
	 * ElementType is type of elements which will be contained 
	 * in created dataset.
	 * @param width Width of desided image.
	 * @param height Height of desided image.
	 * @param depth Depth of desided image.
	 * @param elementWidth Width of each element.
	 * @param elementHeight Height of each element.
	 * @param elementDepth Depth of each element.
	 * @return Smart pointer to created image.
	 **/
	template< typename ElementType >
	static typename Image< ElementType, 3 >::Ptr 
	CreateEmptyImage3DTyped( 
			uint32		width, 
			uint32		height, 
			uint32		depth,
			float32		elementWidth = 1.0,
			float32		elementHeight = 1.0,
			float32		elementDepth = 1.0
			);

	template< typename ElementType >
	static void
	ReallocateImage3DData(
			Image< ElementType, 3 >	&image,
			uint32			width, 
			uint32			height,
			uint32			depth,
			float32			elementWidth = 1.0,
			float32			elementHeight = 1.0,
			float32			elementDepth = 1.0
			);

	/**
	 * Method for custom empty 2D image buffer creation.
	 * ElementType is type of elements which will be contained 
	 * in created dataset.
	 * @param width Width of desided image.
	 * @param height Height of desided image.
	 * @param elementWidth Width of each element.
	 * @param elementHeight Height of each element.
	 * @return Smart pointer to abstract ancestor of created image buffer.
	 **/
	template< typename ElementType >
	static AbstractImageData::APtr 
	CreateEmptyImageData2D( 
			uint32		width, 
			uint32		height,
			float32		elementWidth = 1.0,
			float32		elementHeight = 1.0
			);

	/**
	 * Method for custom empty 2D image creation. Difference from 
	 * previous method is in return value.
	 * ElementType is type of elements which will be contained 
	 * in created dataset.
	 * @param width Width of desided image.
	 * @param height Height of desided image.
	 * @param elementWidth Width of each element.
	 * @param elementHeight Height of each element.
	 * @return Smart pointer to created image.
	 **/
	template< typename ElementType >
	static typename ImageDataTemplate< ElementType >::Ptr
	CreateEmptyImageData2DTyped( 
			uint32		width, 
			uint32		height,
			float32		elementWidth = 1.0,
			float32		elementHeight = 1.0
			);

	/**
	 * Method for custom empty 3D image creation.
	 * ElementType is type of elements which will be contained 
	 * in created dataset.
	 * @param width Width of desided image.
	 * @param height Height of desided image.
	 * @param depth Depth of desided image.
	 * @param elementWidth Width of each element.
	 * @param elementHeight Height of each element.
	 * @param elementDepth Depth of each element.
	 * @return Smart pointer to abstract ancestor of created image.
	 **/
	template< typename ElementType >
	static AbstractImageData::APtr 
	CreateEmptyImageData3D( 
			uint32		width, 
			uint32		height, 
			uint32		depth,
			float32		elementWidth = 1.0,
			float32		elementHeight = 1.0,
			float32		elementDepth = 1.0
			);

	/**
	 * Method for custom empty 3D image creation. Difference from 
	 * previous method is in return value.
	 * ElementType is type of elements which will be contained 
	 * in created dataset.
	 * @param width Width of desided image.
	 * @param height Height of desided image.
	 * @param depth Depth of desided image.
	 * @param elementWidth Width of each element.
	 * @param elementHeight Height of each element.
	 * @param elementDepth Depth of each element.
	 * @return Smart pointer to created image.
	 **/
	template< typename ElementType >
	static typename ImageDataTemplate< ElementType >::Ptr 
	CreateEmptyImageData3DTyped( 
			uint32		width, 
			uint32		height, 
			uint32		depth,
			float32		elementWidth = 1.0,
			float32		elementHeight = 1.0,
			float32		elementDepth = 1.0
			);

	/**
	 * Creates image from given dicomObject set.
	 * @param dicomObjects Given set of dicom objects.
	 * @return Smart pointer to created image.
	 * @exception ImageFactory::EWrongPointer Thrown when passed pointer isn't valid.
	 * @exception ImageFactory::EEmptyDicomObjSet Thrown when empty set passed.
	 * @exception ImageFactory::EUnknowDataType Thrown when type for element with 
	 * parameters from dicomObject doesn't exist.
	 **/
	static AbstractImage::AImagePtr 
	CreateImageFromDICOM( M4D::Dicom::DcmProvider::DicomObjSetPtr dicomObjects );

	/**
	 * Creates image from given dicomObject set.
	 * @param dicomObjects Given set of dicom objects.
	 * @return Smart pointer to created image.
	 * @exception ImageFactory::EWrongPointer Thrown when passed pointer isn't valid.
	 * @exception ImageFactory::EEmptyDicomObjSet Thrown when empty set passed.
	 * @exception ImageFactory::EUnknowDataType Thrown when type for element with 
	 * parameters from dicomObject doesn't exist.
	 **/
	static AbstractImageData::APtr 
	CreateImageDataFromDICOM( M4D::Dicom::DcmProvider::DicomObjSetPtr dicomObjects );
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
		uint32 			imageSize, 
		uint8			*& dataArray  
		);

	/**
	 * @param typeId Id of type used for storing pixel value.
	 * @param imageSize How many elements of size 'pixelSize' will be stored in array.
	 * @param dataArray Filled array of image elements.
	 * @param info Filled dimension info array.
	 **/
	static AbstractImageData*
	CreateImageFromDataAndTypeID(
		int 			typeId,
		uint32 			imageSize, 
		uint8			* dataArray, 
		DimensionInfo		* info
		);

	//TODO - make this function asynchronous. Add locking of array in image.
	/**
	 * @param dicomObjects Set of dicom objects, which will be flushed into array.
	 * @param elementTypeID Type of stored elements.
	 * @param imageSize How many elements of size 'pixelSize' can be stored in array.
 	 * @param stride Number of BYTES!!! used per one object flush (size of one layer in bytes).
	 * @param dataArray Array to be filled from dicom objects. Must be allocated!!!
	 * @exception EWrongArrayForFlush Thrown when NULL array passed, or imageSize is less than
	 * space needed for flushing all dicom objects.
	 **/
	static void
	FlushDicomObjects(
		M4D::Dicom::DcmProvider::DicomObjSetPtr	&dicomObjects,
		int		 			elementTypeID, 
		uint32 					imageSize,
		uint32					stride,
		uint8					* dataArray
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
	EUnknowDataType( uint8 size, bool sign ): _size( size ), _sign( sign ) {}

	uint8
	GetSize()const 
		{ return _size; }
	uint8
	GetSign()const 
		{ return _sign; }
private:
	uint8 _size;
	bool _sign;
};

class ImageFactory::EWrongDICOMObjIndex
{
public:
	EWrongDICOMObjIndex() {}

	//TODO
private:

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

} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

//Including template implementation
#include "ImageFactory.tcc"

#endif /*_IMAGE_FACTORY_H*/

/** @} */

