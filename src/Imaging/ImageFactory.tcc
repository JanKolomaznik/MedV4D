#ifndef _IMAGE_FACTORY_H
#error File ImageDataTemplate.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{


/**
 * Function creating templated array of desired size.
 * @param ElementType 
 * @param size Size of required array.
 * @exception EFailedArrayAllocation If array couldn't be allocated.
 **/
template< typename ElementType >
ElementType*
PrepareElementArray( uint32 size )
{
	try
	{
		ElementType *array = NULL;
		array = new ElementType[size];
		 
		return array;
	}
	catch( ... )
	{
		throw EFailedArrayAllocation();
	}
}

template< typename ElementType >
AbstractImage::AImagePtr 
ImageFactory::CreateEmptyImageFromExtents( 
		uint32		dim,
		int32		minimums[], 
		int32		maximums[],
		float32		elementExtents[]
		)
{
	//TODO fix
	switch( dim ) {
	case 2:		
		return CreateEmptyImage2D< ElementType >( 
				(uint32) maximums[0]-minimums[0], 
				(uint32) maximums[1]-minimums[1],
				elementExtents[0],
				elementExtents[1]
				);
	case 3:
		return CreateEmptyImage3D< ElementType >( 
				(uint32) maximums[0]-minimums[0], 
				(uint32) maximums[1]-minimums[1],
				(uint32) maximums[2]-minimums[2],
				elementExtents[0],
				elementExtents[1],
				elementExtents[2]
				);
	default:
		ASSERT( false );
		return AbstractImage::AImagePtr();
	}
}

template< typename ElementType >
AbstractImage::AImagePtr 
ImageFactory::CreateEmptyImage2D( 
			uint32		width, 
			uint32		height,
			float32		elementWidth,
			float32		elementHeight
			)
{
	//TODO exceptions
	typename Image< ElementType, 2 >::Ptr ptr = 
		ImageFactory::CreateEmptyImage2DTyped< ElementType >( width, height, elementWidth, elementHeight );

	AbstractImage::AImagePtr aptr = 
		boost::static_pointer_cast
		< AbstractImage, Image<ElementType, 2 > >( ptr );
	return aptr; 
}

template< typename ElementType >
typename Image< ElementType, 2 >::Ptr 
ImageFactory::CreateEmptyImage2DTyped( 
			uint32		width, 
			uint32		height,
			float32		elementWidth,
			float32		elementHeight
			)
{
	//TODO exceptions
	typename ImageDataTemplate< ElementType >::Ptr ptr = 
		ImageFactory::CreateEmptyImageData2DTyped< ElementType >( width, height, elementWidth, elementHeight );

	Image< ElementType, 2 > *img = new Image< ElementType, 2 >( ptr );

	return typename Image< ElementType, 2 >::Ptr( img );
}

template< typename ElementType >
void
ImageFactory::ReallocateImage2DData(
		Image< ElementType, 2 >	&image,
		uint32			width, 
		uint32			height,
		float32			elementWidth,
		float32			elementHeight
		)
{
	//TODO exceptions
	typename ImageDataTemplate< ElementType >::Ptr ptr = 
		ImageFactory::CreateEmptyImageData2DTyped< ElementType >( width, height, elementWidth, elementHeight );

	image.ReallocateData( ptr );
}

template< typename ElementType >
AbstractImage::AImagePtr 
ImageFactory::CreateEmptyImage3D( 
			uint32		width, 
			uint32		height,
			uint32		depth,
			float32		elementWidth,
			float32		elementHeight,
			float32		elementDepth
			)
{
	//TODO exceptions
	typename Image< ElementType, 3 >::Ptr ptr = 
		ImageFactory::CreateEmptyImage3DTyped< ElementType >( width, height, depth, elementWidth, elementHeight, elementDepth );

	AbstractImage::AImagePtr aptr = 
		boost::static_pointer_cast
		< AbstractImage, Image<ElementType, 3 > >( ptr );
	return aptr; 
}

template< typename ElementType >
typename Image< ElementType, 3 >::Ptr 
ImageFactory::CreateEmptyImage3DTyped( 
			uint32		width, 
			uint32		height,
			uint32		depth,
			float32		elementWidth,
			float32		elementHeight,
			float32		elementDepth
			)
{
	//TODO exceptions
	typename ImageDataTemplate< ElementType >::Ptr ptr = 
		ImageFactory::CreateEmptyImageData3DTyped< ElementType >( width, height, depth, elementWidth, elementHeight, elementDepth );

	Image< ElementType, 3 > *img = new Image< ElementType, 3 >( ptr );

	return typename Image< ElementType, 3 >::Ptr( img );
}

template< typename ElementType >
void
ImageFactory::ReallocateImage3DData(
		Image< ElementType, 3 >	&image,
		uint32			width, 
		uint32			height,
		uint32			depth,
		float32			elementWidth,
		float32			elementHeight,
		float32			elementDepth
		)
{
	//TODO exceptions
	typename ImageDataTemplate< ElementType >::Ptr ptr = 
		ImageFactory::CreateEmptyImageData3DTyped< ElementType >( width, height, depth, elementWidth, elementHeight, elementDepth );

	image.ReallocateData( ptr );
}

//**********************************************************************

template< typename ElementType >
AbstractImageData::APtr 
ImageFactory::CreateEmptyImageData2D( 
			uint32		width, 
			uint32		height,
			float32		elementWidth,
			float32		elementHeight
			)
{
	typename ImageDataTemplate< ElementType >::Ptr ptr = 
		ImageFactory::CreateEmptyImageData2DTyped< ElementType >( width, height, elementWidth, elementHeight );

	AbstractImageData::APtr aptr = 
		boost::static_pointer_cast
		< AbstractImageData, ImageDataTemplate<ElementType> >( ptr );

	return aptr;
}

template< typename ElementType >
typename ImageDataTemplate< ElementType >::Ptr 
ImageFactory::CreateEmptyImageData2DTyped( 
			uint32		width, 
			uint32		height,
			float32		elementWidth,
			float32		elementHeight
			)
{
	ImageDataTemplate< ElementType > *newImage;
	try
	{
		uint32 size = width * height;
		
		//Preparing informations about dimensionality.
		DimensionInfo *info = new DimensionInfo[ 2 ];
		info[0].Set( width, 1, elementWidth );
		info[1].Set( height, width, elementHeight );

		//Creating place for data storage.
		ElementType *array = PrepareElementArray< ElementType >( size );
		
		//Creating new image, which is using allocated data storage.
		newImage = new ImageDataTemplate< ElementType >( array, info, 2, size );
	}
	catch( ... )
	{
		//TODO exception handling
		throw;
	}

	//Returning smart pointer to abstract image class.
	return typename ImageDataTemplate< ElementType >::Ptr( newImage );
}

template< typename ElementType >
AbstractImageData::APtr 
ImageFactory::CreateEmptyImageData3D( 
			uint32		width, 
			uint32		height, 
			uint32		depth,
			float32		elementWidth,
			float32		elementHeight,
			float32		elementDepth
			)
{
	typename ImageDataTemplate< ElementType >::Ptr ptr = 
		ImageFactory::CreateEmptyImageData3DTyped< ElementType >( width, height, depth, elementWidth, elementHeight, elementDepth );

	AbstractImageData::APtr aptr = 
		boost::static_pointer_cast
		< AbstractImageData, ImageDataTemplate<ElementType> >( ptr );

	return aptr;
}


template< typename ElementType >
typename ImageDataTemplate< ElementType >::Ptr 
ImageFactory::CreateEmptyImageData3DTyped( 
			uint32		width, 
			uint32		height, 
			uint32		depth,
			float32		elementWidth,
			float32		elementHeight,
			float32		elementDepth
			)
{
	//TODO exception handling
	uint32 size = width * height * depth;
	
	//Preparing informations about dimensionality.
	DimensionInfo *info = new DimensionInfo[ 3 ];
	info[0].Set( width, 1, elementWidth );
	info[1].Set( height, width, elementHeight );
	info[2].Set( depth, (width * height), elementDepth );

	//Creating place for data storage.
	ElementType *array = PrepareElementArray< ElementType >( size );
	
	//Creating new image, which is using allocated data storage.
	ImageDataTemplate< ElementType > *newImage = 
		new ImageDataTemplate< ElementType >( array, info, 3, size );

	//Returning smart pointer to abstract image class.
	return typename ImageDataTemplate< ElementType >::Ptr( newImage );
}


} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_IMAGE_FACTORY_H*/

