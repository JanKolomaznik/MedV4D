#include "Imaging/Image.h"


namespace M4D
{
namespace Imaging
{

AbstractImage::AbstractImage( unsigned dim, DimensionExtents *dimExtents )
: _dimCount( dim ), _dimensionExtents( dimExtents )
{

}

const DimensionExtents &
AbstractImage::GetDimensionExtents( unsigned dimension )const
{
	if( dimension >= _dimCount ) {
		//TODO throw exception
	}
	return _dimensionExtents[ dimension ];
}
	

}/*namespace Imaging*/
}/*namespace M4D*/
