#include "AbstractImage.h"

namespace M4D
{

namespace Images
{

AbstractImage::AbstractImage( 
			DimensionInfo		*parameters,
			unsigned short		dimension,
			size_t			elementCount
			)
	: _elementCount( elementCount ), _dimension( dimension ), _parameters( parameters )
{
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

/**
 * Destructor is pure virtual, but definition is needed to
 * avoid compile time errors.
 **/
AbstractImage::~AbstractImage()
{
}


const DimensionInfo&
AbstractImage::GetDimensionInfo( unsigned short dim )const
{
	if( dim >= _dimension ) {
		throw EWrongDimension( dim, _dimension );
	}

	return _parameters[ dim ];
}

}/*namespace Images*/
} /*namespace M4D*/
