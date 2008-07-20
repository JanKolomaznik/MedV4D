#include "Imaging/Image.h"


namespace M4D
{
namespace Imaging
{

AbstractImage::AbstractImage( unsigned dim, DimensionExtents *dimExtents )
: _dimCount( dim ), _dimensionExtents( dimExtents )
{

}

AbstractImage::~AbstractImage()
{

}

const DimensionExtents &
AbstractImage::GetDimensionExtents( unsigned dimension )const
{
	if( dimension >= _dimCount ) {
		throw EWrongDimension();
	}
	return _dimensionExtents[ dimension ];
}
	

}/*namespace Imaging*/
}/*namespace M4D*/
