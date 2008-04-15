#ifndef _IMAGE_FACTORY_H
#error File ImageDataTemplate.tcc cannot be included directly!
#else

namespace M4D
{
namespace Images
{

template< typename ElementType >
ElementType*
PrepareElementArray( size_t size )
{
	//TODO exception handling
	ElementType *array = new ElementType[size];

	return array;
}

template< typename ElementType >
AbstractImage::APtr 
ImageFactory::CreateEmptyImage2D( 
			size_t		width, 
			size_t		height
			)
{
	//TODO exception handling
	size_t size = width * height;
	
	//Preparing informations about dimensionality.
	DimensionInfo *info = new DimensionInfo[ 2 ];
	info[0].Set( width, 1 );
	info[1].Set( height, width );

	//Creating place for data storage.
	ElementType *array = PrepareElementArray< ElementType >( size );
	
	//Creating new image, which is using allocated data storage.
	ImageDataTemplate< ElementType > *newImage = 
		new ImageDataTemplate< ElementType >( array, info, 2, size );

	//Returning smart pointer to abstract image class.
	return AbstractImage::APtr( newImage );
}

template< typename ElementType >
AbstractImage::APtr 
ImageFactory::CreateEmptyImage3D( 
			size_t		width, 
			size_t		height, 
			size_t		depth
			)
{
	//TODO exception handling
	size_t size = width * height * depth;
	
	//Preparing informations about dimensionality.
	DimensionInfo *info = new DimensionInfo[ 3 ];
	info[0].Set( width, 1 );
	info[1].Set( height, width );
	info[2].Set( depth, width * height );

	//Creating place for data storage.
	ElementType *array = PrepareElementArray< ElementType >( size );
	
	//Creating new image, which is using allocated data storage.
	ImageDataTemplate< ElementType > *newImage = 
		new ImageDataTemplate< ElementType >( array, info, 3, size );

	//Returning smart pointer to abstract image class.
	return AbstractImage::APtr( newImage );
}

} /*namespace Images*/
} /*namespace M4D*/

#endif /*_IMAGE_FACTORY_H*/

