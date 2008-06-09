#ifndef _IMAGE__H
#error File Image.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{

template< typename ElementType >
Image< ElementType, 2 >::Image( AbstractImageData::APtr imageData )
{

}

template< typename ElementType >
Image< ElementType, 2 >::Image( typename ImageDataTemplate< ElementType >::Ptr imageData )
{

}
	
template< typename ElementType >
Image< ElementType, 2 >::~Image()
{

}

template< typename ElementType >
ElementType &
Image< ElementType, 2 >::GetElement( size_t x, size_t y )
{

}

template< typename ElementType >
const ElementType &
Image< ElementType, 2 >::GetElement( size_t x, size_t y )const
{

}

template< typename ElementType >
typename Image< ElementType, 2 >::Ptr
Image< ElementType, 2 >::GetRestricted2DImage( 
			size_t x1, 
			size_t y1, 
			size_t x2, 
			size_t y2 
			)
{

}

//*****************************************************************************

template< typename ElementType >
Image< ElementType, 3 >::Image( AbstractImageData::APtr imageData )
{

}

template< typename ElementType >
Image< ElementType, 3 >::Image( typename ImageDataTemplate< ElementType >::Ptr imageData )
{

}

template< typename ElementType >
Image< ElementType, 3 >::~Image()
{

}

template< typename ElementType >
ElementType &
Image< ElementType, 3 >::GetElement( size_t x, size_t y, size_t z )
{

}

template< typename ElementType >
const ElementType &
Image< ElementType, 3 >::GetElement( size_t x, size_t y, size_t z )const
{

}

template< typename ElementType >
typename Image< ElementType, 2 >::Ptr
Image< ElementType, 3 >::GetRestricted2DImage( 
		size_t x1, 
		size_t y1, 
		size_t z1, 
		size_t x2, 
		size_t y2, 
		size_t z2 
		)
{

}

template< typename ElementType >
typename Image< ElementType, 3 >::Ptr
Image< ElementType, 3 >::GetRestricted3DImage( 
		size_t x1, 
		size_t y1, 
		size_t z1, 
		size_t x2, 
		size_t y2, 
		size_t z2 
		)
{

}

//*****************************************************************************

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_IMAGE__H*/
