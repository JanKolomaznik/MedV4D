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
PrepareElementArray( size_t size )
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
ImageFactory::CreateEmptyImage2D( 
			size_t		width, 
			size_t		height
			)
{
	//TODO exceptions
	typename Image< ElementType, 2 >::Ptr ptr = 
		ImageFactory::CreateEmptyImage2DTyped< ElementType >( width, height );

	AbstractImage::AImagePtr aptr = 
		boost::static_pointer_cast
		< AbstractImage, Image<ElementType, 2 > >( ptr );
	return aptr; 
}

template< typename ElementType >
typename Image< ElementType, 2 >::Ptr 
ImageFactory::CreateEmptyImage2DTyped( 
			size_t		width, 
			size_t		height
			)
{
	//TODO exceptions
	typename ImageDataTemplate< ElementType >::Ptr ptr = 
		ImageFactory::CreateEmptyImageData2DTyped< ElementType >( width, height );

	Image< ElementType, 2 > *img = new Image< ElementType, 2 >( ptr );

	return typename Image< ElementType, 2 >::Ptr( img );
}

template< typename ElementType >
AbstractImage::AImagePtr 
ImageFactory::CreateEmptyImage3D( 
			size_t		width, 
			size_t		height,
			size_t		depth
			)
{
	//TODO exceptions
	typename Image< ElementType, 3 >::Ptr ptr = 
		ImageFactory::CreateEmptyImage3DTyped< ElementType >( width, height, depth );

	AbstractImage::AImagePtr aptr = 
		boost::static_pointer_cast
		< AbstractImage, Image<ElementType, 3 > >( ptr );
	return aptr; 
}

template< typename ElementType >
typename Image< ElementType, 3 >::Ptr 
ImageFactory::CreateEmptyImage3DTyped( 
			size_t		width, 
			size_t		height,
			size_t		depth
			)
{
	//TODO exceptions
	typename ImageDataTemplate< ElementType >::Ptr ptr = 
		ImageFactory::CreateEmptyImageData3DTyped< ElementType >( width, height, depth );

	Image< ElementType, 3 > *img = new Image< ElementType, 3 >( ptr );

	return typename Image< ElementType, 3 >::Ptr( img );
}

//**********************************************************************

template< typename ElementType >
AbstractImageData::APtr 
ImageFactory::CreateEmptyImageData2D( 
			size_t		width, 
			size_t		height
			)
{
	typename ImageDataTemplate< ElementType >::Ptr ptr = 
		ImageFactory::CreateEmptyImageData2DTyped< ElementType >( width, height );

	AbstractImageData::APtr aptr = 
		boost::static_pointer_cast
		< AbstractImageData, ImageDataTemplate<ElementType> >( ptr );

	return aptr;
}

template< typename ElementType >
typename ImageDataTemplate< ElementType >::Ptr 
ImageFactory::CreateEmptyImageData2DTyped( 
			size_t		width, 
			size_t		height
			)
{
	ImageDataTemplate< ElementType > *newImage;
	try
	{
		size_t size = width * height;
		
		//Preparing informations about dimensionality.
		DimensionInfo *info = new DimensionInfo[ 2 ];
		info[0].Set( width, 1, 1.0 );
		info[1].Set( height, width, 1.0 );

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
			size_t		width, 
			size_t		height, 
			size_t		depth
			)
{
	typename ImageDataTemplate< ElementType >::Ptr ptr = 
		ImageFactory::CreateEmptyImageData3DTyped< ElementType >( width, height, depth );

	AbstractImageData::APtr aptr = 
		boost::static_pointer_cast
		< AbstractImageData, ImageDataTemplate<ElementType> >( ptr );

	return aptr;
}


template< typename ElementType >
typename ImageDataTemplate< ElementType >::Ptr 
ImageFactory::CreateEmptyImageData3DTyped( 
			size_t		width, 
			size_t		height, 
			size_t		depth
			)
{
	//TODO exception handling
	size_t size = width * height * depth;
	
	//Preparing informations about dimensionality.
	DimensionInfo *info = new DimensionInfo[ 3 ];
	info[0].Set( width, 1, 1.0 );
	info[1].Set( height, width, 1.0 );
	info[2].Set( depth, width * height, 1.0 );

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

