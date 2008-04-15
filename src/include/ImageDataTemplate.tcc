#ifndef _IMAGE_DATA_TEMPLATE_H
#error File ImageDataTemplate.tcc cannot be included directly!
#else

namespace M4D
{
namespace Images
{

template < typename ElementType >
ImageDataTemplate< ElementType >::ImageDataTemplate( 
			ElementType 		*data, 
			DimensionInfo		*parameters,
			unsigned short		dimension,
			size_t			elementCount
			) 
	: _elementCount( elementCount ), _dimension( dimension ), 
	_parameters( parameters ), _data( data ) 	
{
	if ( _data == NULL ) {
		//TODO handle problem
	}
	if ( _parameters == NULL ) {
		//TODO handle problem
	}
	if ( _dimension == 0 ) {
		//TODO handle problem
	}
	if ( elementCount == 0 ) {
		//TODO handle problem
	}
}

template < typename ElementType >
ImageDataTemplate< ElementType >::~ImageDataTemplate()
{
	if ( _data ) {
		delete[] _data;
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
		throw EIndexOutOfBounds( index );
	}

	return _data[ index ];
}

template < typename ElementType >
ElementType&
ImageDataTemplate< ElementType >::Get( size_t index )
{
	if ( index >= _elementCount ) {
		throw EIndexOutOfBounds( index );
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
template < typename ElementType >
const DimensionInfo&
ImageDataTemplate< ElementType >::GetDimensionInfo( unsigned short dim )const
{
	if( dim >= _dimension ) {
		throw EWrongDimension( dim, _dimension );
	}

	return _parameters[ dim ];
}

} /*namespace Images*/
} /*namespace M4D*/

#endif /*_IMAGE_DATA_TEMPLATE_H*/
