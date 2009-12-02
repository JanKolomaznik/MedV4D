/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AImageData.cpp 
 * @{ 
 **/

#include "Imaging/AImageData.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 *
 *  @author Jan Kolomaznik
 */

namespace M4D
{

namespace Imaging
{

AImageData::AImageData( 
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
AImageData::~AImageData()
{
}


const DimensionInfo&
AImageData::GetDimensionInfo( unsigned short dim )const
{
	if( dim >= _dimension ) {
		_THROW_ EBadDimension( dim, _dimension );
	}

	return _parameters[ dim ];
}

}/*namespace Imaging*/
} /*namespace M4D*/

/** @} */
/** @} */

