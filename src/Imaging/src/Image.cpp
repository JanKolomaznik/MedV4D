/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file Image.cpp 
 * @{ 
 **/

#include "Imaging/Image.h"

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

AbstractImage::AbstractImage( uint16 dim, DimensionExtents *dimExtents )
: AbstractDataSet( DATASET_IMAGE ), _dimCount( dim ), _dimensionExtents( dimExtents )
{

}

AbstractImage::~AbstractImage()
{

}

const DimensionExtents &
AbstractImage::GetDimensionExtents( unsigned dimension )const
{
	if( dimension >= _dimCount ) {
		throw EBadDimension();
	}
	return _dimensionExtents[ dimension ];
}
	

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */
/** @} */

