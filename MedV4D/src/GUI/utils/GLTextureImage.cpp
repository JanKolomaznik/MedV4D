#include "MedV4D/GUI/utils/GLTextureImage.h"
#include <boost/cast.hpp>

namespace M4D
{

/*void
updateTextureSubImage( GLTextureImage &aTexImage, const M4D::Imaging::AImageRegion &aSubImage )
{
	if ( aTexImage.GetDimension() != aSubImage.GetDimension() ) {
		_THROW_ M4D::ErrorHandling::EBadParameter( "Texture and subimage have different dimension" );
	}
	switch ( aTexImage.GetDimension() ) {
	case 2:
		return updateTextureSubImageTyped< 2 >( aTexImage.GetDimensionedInterface<2>(), static_cast< const M4D::Imaging::AImageRegionDim<2> & >( aSubImage ) );
	case 3:
		return updateTextureSubImageTyped< 3 >( aTexImage.GetDimensionedInterface<3>(), static_cast< const M4D::Imaging::AImageRegionDim<3> & >( aSubImage ) );
	default:
		_THROW_ M4D::ErrorHandling::EBadParameter( "Image with wrong dimension - supported only 2 and 3" );
	}
}*/
	
GLTextureImage::Ptr
createTextureFromImage( const M4D::Imaging::AImageRegion &image, bool aLinearInterpolation )
{
	switch ( image.GetDimension() ) {
	case 2:
		return createTextureFromImageTyped< 2 >( static_cast< const M4D::Imaging::AImageRegionDim<2> & >( image ), aLinearInterpolation );
	case 3:
		return createTextureFromImageTyped< 3 >( static_cast< const M4D::Imaging::AImageRegionDim<3> & >( image ), aLinearInterpolation );
	default:
		_THROW_ M4D::ErrorHandling::EBadParameter( "Image with wrong dimension - supported only 2 and 3" );
	}
}

void
recreateTextureFromImage( GLTextureImage &aTexImage, const M4D::Imaging::AImageRegion &image )
{
	if ( aTexImage.GetDimension() != image.GetDimension() ) {
		_THROW_ M4D::ErrorHandling::EBadParameter( "Texture and subimage have different dimension" );
	}
	switch ( image.GetDimension() ) {
	case 2:
		recreateTextureFromImageTyped< 2 >( aTexImage.GetDimensionedInterface<2>(), static_cast< const M4D::Imaging::AImageRegionDim<2> & >( image ) );
		break;
	case 3:
		recreateTextureFromImageTyped< 3 >( aTexImage.GetDimensionedInterface<3>(), static_cast< const M4D::Imaging::AImageRegionDim<3> & >( image ) );
		break;
	default:
		_THROW_ M4D::ErrorHandling::EBadParameter( "Image with wrong dimension - supported only 2 and 3" );
	}
}

} /*namespace M4D*/

