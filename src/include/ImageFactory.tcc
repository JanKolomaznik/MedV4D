#ifndef _IMAGE_FACTORY_H
#error File ImageDataTemplate.tcc cannot be included directly!
#elif

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
static AbstractImage::Ptr 
ImageFactory::CreateEmptyImage2D( 
			size_t		width, 
			size_t		height
			)
{
	size_t size = width * height;

	ElementType *array = PrepareElementArray< ElementType >( size );

}

template< typename ElementType >
static AbstractImage::Ptr 
ImageFactory::CreateEmptyImage3D( 
			size_t		width, 
			size_t		height, 
			size_t		depth
			)
{
	size_t size = width * height * depth;

}

} /*namespace Images*/

#endif /*_IMAGE_FACTORY_H*/

