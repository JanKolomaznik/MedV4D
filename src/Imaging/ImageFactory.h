#ifndef _IMAGE_FACTORY_H
#define _IMAGE_FACTORY_H

#include "Imaging/AbstractImageData.h"
#include "Imaging/ImageDataTemplate.h"
#include "Imaging/Image.h"

#include "common/Common.h"
#include "common/Vector.h"

#include <iostream>
#include <fstream>
#include <string>

/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ImageFactory.h 
 * @{ 
 **/

namespace M4D
{

/**
 * Factory class which takes care of allocation of image 
 * datasets. It is able to create empty image, or fill 
 * it with DICOM data.
 **/

namespace Imaging
{

//TODO check comments

class ImageFactory
{
public:
	class EWrongPointer;
	class EEmptyDicomObjSet;
	class EWrongArrayForFlush;
	class EUnknowDataType;
	class EWrongDICOMObjIndex;

	class EWrongStreamBeginning;
	class EWrongFormatVersion;
	class EWrongHeader;
	class EWrongDatasetTypeIdentification;
	
	static AbstractImage::Ptr 
	DeserializeImage(M4D::IO::InStream &stream);

	static AbstractImage::Ptr 
	DeserializeImageFromStream(M4D::IO::InStream &stream);

	static void 
	SerializeImage( M4D::IO::OutStream &stream, const AbstractImage &image );

	template< typename ElementType, unsigned Dimension >
	static void 
	SerializeImage( M4D::IO::OutStream &stream, const Image< ElementType, Dimension > &image );
	/**
	 * Create image according to passed information.
	 * \param dim Dimesnion of desired image.
	 * \param minimums Minimum coordinates of elements in image (for each dimension).
	 * \param maximums Vector of first element out of image (for each dimension).
	 * \param elementExtents Proportion of elements in each dimension
	 * @return Smart pointer to abstract ancestor of created image.
	 **/
	template< typename ElementType >
	static AbstractImage::Ptr 
	CreateEmptyImageFromExtents( 
			uint32		dim,
			int32		minimums[], 
			int32		maximums[],
			float32		elementExtents[]
			);

	template< typename ElementType, unsigned Dim >
	static typename Image< ElementType, Dim >::Ptr 
	CreateEmptyImageFromExtents( 
			Vector< int32, Dim >	minimum, 
			Vector< int32, Dim >	maximum,
			Vector< float32, Dim >	elementExtents
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
	static AbstractImage::Ptr 
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
	static AbstractImage::Ptr 
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

	template< typename ElementType, unsigned Dim >
	static typename ImageDataTemplate< ElementType >::Ptr 
	CreateEmptyImageDataTyped( 
			Vector< int32, Dim > 	size,
			Vector< float32, Dim >	elementExtents
			);
	
	template< typename ElementType, unsigned Dim  >
	static void	AssignNewDataToImage( 
			ElementType *pointer, Image<ElementType, Dim> &image,
			Vector< int32, Dim > 	&size, 
			Vector< float32, Dim >	&elementSize);

	
	template< unsigned Dim >
	static void
	ChangeImageSize( 
				AbstractImage			&image,
				Vector< int32, Dim > 	minimum,
				Vector< int32, Dim > 	maximum,
				Vector< float32, Dim >	elementExtents
			    );
	
	template< typename ElementType, unsigned Dim >
	static void
	ChangeImageSize( 
				Image< ElementType, Dim >	&image,
				Vector< int32, Dim > 	minimum,
				Vector< int32, Dim > 	maximum,
				Vector< float32, Dim >	elementExtents
			    );
	

	template< typename ElementType, uint32 Dimension >
	static void
	DumpImage( std::ostream &stream, const Image< ElementType, Dimension > & image );

	template< typename ElementType, uint32 Dimension >
	static void
	DumpImage( std::string filename, const Image< ElementType, Dimension > & image );

	static void
	DumpImage( std::string filename, const AbstractImage & image );

	static AbstractImage::Ptr
	LoadDumpedImage( std::istream &stream );

	static AbstractImage::Ptr
	LoadDumpedImage( std::string filename );
	
	/*template< typename ElementType, unsigned Dim >
	static void AllocateDataAccordingProperties(Image<ElementType, Dim> &image);*/
	
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
		
protected:

private:
	/**
	 * Not implemented - cannot instantiate this class.
	 **/
	ImageFactory();
	ImageFactory( const ImageFactory& );
	void operator=( const ImageFactory& );

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


class ImageFactory::EWrongStreamBeginning
{
public:
	EWrongStreamBeginning() {}

	//TODO
};

class ImageFactory::EWrongFormatVersion
{
public:
	EWrongFormatVersion() {}

	//TODO
};

class ImageFactory::EWrongHeader
{
public:
	EWrongHeader() {}

	//TODO
};

class ImageFactory::EWrongDatasetTypeIdentification
{
public:
	EWrongDatasetTypeIdentification() {}

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



} /*namespace Imaging*/
/** @} */
} /*namespace M4D*/


//Including template implementation
#include "ImageFactory.tcc"

#endif /*_IMAGE_FACTORY_H*/


