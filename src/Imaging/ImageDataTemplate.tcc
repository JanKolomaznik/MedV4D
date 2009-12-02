/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ImageDataTemplate.tcc 
 * @{ 
 **/

#ifndef _IMAGE_DATA_TEMPLATE_H
#error File ImageDataTemplate.tcc cannot be included directly!
#else

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

template < typename ElementType >
ImageDataTemplate< ElementType >::ImageDataTemplate( 
			ElementType 		*data, 
			DimensionInfo		*parameters,
			unsigned short		dimension,
			size_t			elementCount
			) 
	: AImageData( parameters, dimension, elementCount ), _data( data ), _arrayPointer( data, data )
{
	if ( _data == NULL ) {
		//TODO handle problem
	}
}

template < typename ElementType >
ImageDataTemplate< ElementType >::ImageDataTemplate( 
			AlignedArrayPointer< ElementType >	data, 
			DimensionInfo		*parameters,
			unsigned short		dimension,
			size_t			elementCount
			) 
	: AImageData( parameters, dimension, elementCount ), _data( data.aligned ), _arrayPointer( data ) 	
{
	if ( _data == NULL ) {
		//TODO handle problem
	}
}

template < typename ElementType >
ImageDataTemplate< ElementType >::~ImageDataTemplate()
{
	if ( _data ) {
		delete[] _arrayPointer.original;
	}
	if ( _parameters ) {
		delete[] _parameters;
	}
}

template < typename ElementType >
ElementType
ImageDataTemplate< ElementType >::Get( size_t index )const
{
	if ( index >= _elementCount ) {
		_THROW_ EIndexOutOfBounds( index );
	}

	return _data[ index ];
}

template < typename ElementType >
ElementType&
ImageDataTemplate< ElementType >::Get( size_t index )
{
	if ( index >= _elementCount ) {
		_THROW_ EIndexOutOfBounds( index );
	}

	return _data[ index ];
}
//2D------------------------------------------------
template < typename ElementType >
inline ElementType
ImageDataTemplate< ElementType >::Get( size_t x, size_t y )const
{
	size_t index;

	index = x * _parameters[0].stride +
		y * _parameters[1].stride;
	return Get( index );
}

template < typename ElementType >
inline ElementType&
ImageDataTemplate< ElementType >::Get( size_t x, size_t y )
{
	size_t index;

	index = x * _parameters[0].stride +
		y * _parameters[1].stride;
	return Get( index );
}
//2D end--------------------------------------------
//3D------------------------------------------------
template < typename ElementType >
inline ElementType
ImageDataTemplate< ElementType >::Get( size_t x, size_t y, size_t z )const
{
	size_t index;

	index = x * _parameters[0].stride +
		y * _parameters[1].stride +
		z * _parameters[2].stride;
	return Get( index );
}

template < typename ElementType >
inline ElementType&
ImageDataTemplate< ElementType >::Get( size_t x, size_t y, size_t z )
{
	size_t index;

	index = x * _parameters[0].stride +
		y * _parameters[1].stride +
		z * _parameters[2].stride;
	return Get( index );
}
//3D end--------------------------------------------

//4D------------------------------------------------
template < typename ElementType >
inline ElementType
ImageDataTemplate< ElementType >::Get( size_t x, size_t y, size_t z, size_t t )const
{
	size_t index;

	index = x * _parameters[0].stride +
		y * _parameters[1].stride +
		z * _parameters[2].stride +
		t * _parameters[3].stride;
	return Get( index );
}

template < typename ElementType >
inline ElementType&
ImageDataTemplate< ElementType >::Get( size_t x, size_t y, size_t z, size_t t )
{
	size_t index;

	index = x * _parameters[0].stride +
		y * _parameters[1].stride +
		z * _parameters[2].stride +
		t * _parameters[3].stride;
	return Get( index );
}
//4D end--------------------------------------------
template < typename ElementType >
typename ImageDataTemplate< ElementType >::Ptr
ImageDataTemplate< ElementType >::CastAbstractPointer( AImageData::APtr aptr )
{
	if( dynamic_cast< ImageDataTemplate< ElementType > * >( aptr.get() ) == NULL ) {
		//TODO _THROW_ exception
	}

	return boost::static_pointer_cast< ImageDataTemplate< ElementType > >( aptr );
}

} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*_IMAGE_DATA_TEMPLATE_H*/

/** @} */

