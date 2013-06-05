#include <MedV4D/GUI/utils/TextureUtils.h>
#include <boost/cast.hpp>

namespace M4D
{

GLuint
GLPrepareTextureFromImageData( const M4D::Imaging::AImageRegion &image, bool linearInterpolation )
{
	switch ( image.GetDimension() )
	{
	case 2:	
		return GLPrepareTextureFromImageData( M4D::Imaging::AImageRegionDim< 2 >::Cast( image ), linearInterpolation );
	case 3:
		return GLPrepareTextureFromImageData( M4D::Imaging::AImageRegionDim< 3 >::Cast( image ), linearInterpolation );
	default:
		ASSERT( false );
		return 0;
	}
}

GLuint
GLPrepareTextureFromImageData( const M4D::Imaging::AImageRegionDim< 2 > &image, bool linearInterpolation )
{
	TYPE_TEMPLATE_SWITCH_MACRO( image.GetElementTypeID(), return GLPrepareTextureFromImageData2D( static_cast< const M4D::Imaging::ImageRegion< TTYPE, 2 > &>( image ), linearInterpolation ); );
	ASSERT( false );
	return 0;
}

GLuint
GLPrepareTextureFromImageData( const M4D::Imaging::AImageRegionDim< 3 > &image, bool linearInterpolation )
{
	TYPE_TEMPLATE_SWITCH_MACRO( image.GetElementTypeID(), return GLPrepareTextureFromImageData3D( static_cast< const M4D::Imaging::ImageRegion< TTYPE, 3 > &>( image ), linearInterpolation ); );
	ASSERT( false );
	return 0;
}

void
GLUpdateTextureFromSubImageData( GLuint aTexture, const M4D::Imaging::AImageRegionDim< 2 > &image, Vector2i aMinimum, Vector2i aMaximum )
{
	TYPE_TEMPLATE_SWITCH_MACRO( image.GetElementTypeID(), GLUpdateTextureFromSubImageData2D( aTexture, static_cast< const M4D::Imaging::ImageRegion< TTYPE, 2 > &>( image ), aMinimum, aMaximum ); return; );
}

void
GLUpdateTextureFromSubImageData( GLuint aTexture, const M4D::Imaging::AImageRegionDim< 3 > &image, Vector3i aMinimum, Vector3i aMaximum )
{
	TYPE_TEMPLATE_SWITCH_MACRO( image.GetElementTypeID(), GLUpdateTextureFromSubImageData3D( aTexture, static_cast< const M4D::Imaging::ImageRegion< TTYPE, 3 > &>( image ), aMinimum, aMaximum ); return; );
}

void
GLUpdateTextureFromSubImageData( GLuint aTexture, const M4D::Imaging::AImageRegionDim< 3 > &image, Vector3i aMinimum, Vector3i aMaximum );

//-----------------------------------------------------------------------------------------------------------------------------------------
	
	
/*	
void
updateTextureSubImage( soglu::GLTextureImage &aTexImage, const M4D::Imaging::AImageRegion &aSubImage )
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
	
soglu::GLTextureImage::Ptr
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
recreateTextureFromImage( soglu::GLTextureImage &aTexImage, const M4D::Imaging::AImageRegion &image )
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