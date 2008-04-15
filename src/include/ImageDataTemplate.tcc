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
