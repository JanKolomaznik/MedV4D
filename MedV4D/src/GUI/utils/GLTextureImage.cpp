#include "MedV4D/GUI/utils/GLTextureImage.h"

namespace M4D
{

	
GLTextureImage::Ptr
CreateTextureFromImage( const M4D::Imaging::AImageRegion &image, bool aLinearInterpolation )
{
	switch ( image.GetDimension() ) {
	case 2:
		return CreateTextureFromImageTyped< 2 >( static_cast< const M4D::Imaging::AImageRegionDim<2> & >( image ), aLinearInterpolation );
	case 3:
		return CreateTextureFromImageTyped< 3 >( static_cast< const M4D::Imaging::AImageRegionDim<3> & >( image ), aLinearInterpolation );
	default:
		_THROW_ M4D::ErrorHandling::EBadParameter( "Image with wrong dimension - supported only 2 and 3" );
	}
}

} /*namespace M4D*/

