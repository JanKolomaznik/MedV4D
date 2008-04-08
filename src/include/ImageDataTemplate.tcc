#ifndef _IMAGE_DATA_TEMPLATE_H
#error File ImageDataTemplate.tcc cannot be included directly!
#elif

namespace Images
{
template < typename ElementType >
ImageDataTemplate< ElementType >::ImageDataTemplate( 
			ElementType 		*data, 
			DimensionInformations	*parameters,
			unsigned short		dimension,
			size_t			elementCount
			) 
	: _data( data ), _parameters( parameters ), 
	_dimension( dimension ), _elementCount( elementCount )
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
ImageDataTemplate< ElementType >::get( size_t index )const
{
	if ( index >= _elementCount ) {
		throw EIndexOutOfBounds( index );
	}

	return _data[ index ];
}

template < typename ElementType >
ElementType&
ImageDataTemplate< ElementType >::get( size_t index )
{
	if ( index >= _elementCount ) {
		throw EIndexOutOfBounds( index );
	}

	return _data[ index ];
}

} /*namespace Images*/

#endif /*_IMAGE_DATA_TEMPLATE_H*/
