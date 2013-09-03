#pragma once
#include <GL/glew.h>
#include <MedV4D/Imaging/Imaging.h>
#include <soglu/GLTextureImage.hpp>
#include <soglu/GLMUtils.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/cast.hpp>
#include <MedV4D/Common/Common.h>

namespace M4D {

template< typename ImageRegionType >
GLuint
GLPrepareTextureFromImageData2D( const ImageRegionType &image, bool linearInterpolation );

template< typename ImageRegionType >
GLuint
GLPrepareTextureFromImageData3D( const ImageRegionType &image, bool linearInterpolation );

GLuint
GLPrepareTextureFromImageData( const M4D::Imaging::AImageRegion &image, bool linearInterpolation = true );

GLuint
GLPrepareTextureFromImageData( const M4D::Imaging::AImageRegionDim< 2 > &image, bool linearInterpolation = true );

GLuint
GLPrepareTextureFromImageData( const M4D::Imaging::AImageRegionDim< 3 > &image, bool linearInterpolation = true );
//**************************************************************************************
template< typename ImageRegionType >
void
GLUpdateTextureFromSubImageData2D( GLuint aTexture, const ImageRegionType &image, Vector2i aMinimum, Vector2i aMaximum );

template< typename ImageRegionType >
void
GLUpdateTextureFromSubImageData3D( GLuint aTexture, const ImageRegionType &image, Vector3i aMinimum, Vector3i aMaximum );

void
GLUpdateTextureFromSubImageData( GLuint aTexture, const M4D::Imaging::AImageRegionDim< 2 > &image, Vector2i aMinimum, Vector2i aMaximum );

void
GLUpdateTextureFromSubImageData( GLuint aTexture, const M4D::Imaging::AImageRegionDim< 3 > &image, Vector3i aMinimum, Vector3i aMaximum );







template < size_t Dim >
void
updateTextureSubImageTyped(soglu::GLTextureImageTyped< Dim > &aTexImage, const M4D::Imaging::AImageRegionDim<Dim> &aImage, Vector< int, Dim > aMinimum, Vector< int, Dim > aMaximum)
{
	//TODO some checks
	M4D::GLUpdateTextureFromSubImageData( aTexImage.GetTextureGLID(), aImage, aMinimum, aMaximum );
}

template < size_t Dim >
void
updateTextureSubImage(soglu::GLTextureImage &aTexImage, const M4D::Imaging::AImageRegion &aSubImage, Vector< int, Dim > aMinimum, Vector< int, Dim > aMaximum )
{
	if ( aTexImage.GetDimension() != aSubImage.GetDimension() || Dim != aTexImage.GetDimension() ) {
		_THROW_ M4D::ErrorHandling::EBadParameter( "Texture and subimage have different dimension" );
	}

	updateTextureSubImageTyped< Dim >( aTexImage.GetDimensionedInterface<Dim>(), static_cast< const M4D::Imaging::AImageRegionDim<Dim> & >( aSubImage ), aMinimum, aMaximum);
}


soglu::GLTextureImage::Ptr
createTextureFromImage(const M4D::Imaging::AImageRegion &image, bool aLinearInterpolation = true );

template < size_t Dim >
soglu::GLTextureImage::Ptr
createTextureFromImageTyped(const M4D::Imaging::AImageRegionDim<Dim> &image, bool aLinearInterpolation = true )
{
	typedef soglu::GLTextureImageTyped< Dim > TextureImage;
	M4D::Imaging::ImageExtentsRecord< Dim > extents = image.GetImageExtentsRecord();
	GLuint textureID = GLPrepareTextureFromImageData( image, aLinearInterpolation ); //TODO prevent loosing texture during exception
	D_PRINT("Obtained texture : " << textureID);

	soglu::ExtentsRecord< Dim > ext;
	soglu::set(ext.realMinimum, extents.realMinimum.GetData());
	soglu::set(ext.realMaximum, extents.realMaximum.GetData());
	soglu::set(ext.minimum, extents.minimum.GetData());
	soglu::set(ext.maximum, extents.maximum.GetData());
	soglu::set(ext.elementExtents, extents.elementExtents.GetData());

	typename soglu::GLMDimension<Dim>::fvec elementExtents;

	return typename TextureImage::Ptr( new TextureImage( textureID, aLinearInterpolation, ext ) );
}

void
recreateTextureFromImage(soglu::GLTextureImage &aTexImage, const M4D::Imaging::AImageRegion &image );

template < size_t Dim >
void
recreateTextureFromImageTyped(soglu::GLTextureImageTyped< Dim > &aTexImage, const M4D::Imaging::AImageRegionDim<Dim> &image )
{
	//TODO test image properties if same
	aTexImage.DeleteTexture();

	M4D::Imaging::ImageExtentsRecord< Dim > extents = image.GetImageExtentsRecord();
	soglu::ExtentsRecord< Dim > ext;
	soglu::set(ext.realMinimum, extents.realMinimum.GetData());
	soglu::set(ext.realMaximum, extents.realMaximum.GetData());
	soglu::set(ext.minimum, extents.minimum.GetData());
	soglu::set(ext.maximum, extents.maximum.GetData());
	soglu::set(ext.elementExtents, extents.elementExtents.GetData());

	GLuint textureID = GLPrepareTextureFromImageData( image, aTexImage.linearInterpolation() ); //TODO prevent loosing texture during exception
	aTexImage.updateTexture(textureID, ext);
	/*aTexImage.SetImage( image );
	aTexImage.PrepareTexture();*/
}
//------------------------------------------------------------------------------------------
/*inline void
GLColorFromQColor( const QColor &color )
{
	glColor4f( color.redF(), color.greenF(), color.blueF(), color.alphaF() );
}*/

template< typename T >
struct M4DToGLType
{
	static const GLenum GLTypeID = 0;
};

template<>
struct M4DToGLType< uint8 >
{
	static const GLenum GLTypeID = GL_UNSIGNED_BYTE;
};

template<>
struct M4DToGLType< int8 >
{
	static const GLenum GLTypeID = GL_BYTE;
};

template<>
struct M4DToGLType< uint16 >
{
	static const GLenum GLTypeID = GL_UNSIGNED_SHORT;
};

template<>
struct M4DToGLType< int16 >
{
	static const GLenum GLTypeID = GL_SHORT;
};

template<>
struct M4DToGLType< uint32 >
{
	static const GLenum GLTypeID = GL_UNSIGNED_INT;
};

template<>
struct M4DToGLType< int32 >
{
	static const GLenum GLTypeID = GL_INT;
};

template<>
struct M4DToGLType< float32 >
{
	static const GLenum GLTypeID = GL_FLOAT;
};
//-**************************************************************************
template< typename T >
struct M4DToGLTextureInternal
{
	static const GLint GLInternal = 0;
};

template<>
struct M4DToGLTextureInternal< uint8 >
{
	static const GLint GLInternal = GL_LUMINANCE;
};

template<>
struct M4DToGLTextureInternal< int8 >
{
	static const GLint GLInternal = GL_LUMINANCE;
};

template<>
struct M4DToGLTextureInternal< uint16 >
{
	static const GLint GLInternal = GL_R16F;
};

template<>
struct M4DToGLTextureInternal< int16 >
{
	static const GLint GLInternal = GL_R16F;
};

template<>
struct M4DToGLTextureInternal< uint32 >
{
	static const GLint GLInternal = GL_R32F;
};

template<>
struct M4DToGLTextureInternal< int32 >
{
	static const GLint GLInternal = GL_R32F;
};

template<>
struct M4DToGLTextureInternal< int64 >
{
	static const GLint GLInternal = GL_R32F;
};

template<>
struct M4DToGLTextureInternal< uint64 >
{
	static const GLint GLInternal = GL_R32F;
};

template<>
struct M4DToGLTextureInternal< float32 >
{
	static const GLint GLInternal = GL_R32F;
};

template<>
struct M4DToGLTextureInternal< float64 >
{
	static const GLint GLInternal = GL_R32F;
};


template< typename ImageRegionType >
GLuint
GLPrepareTextureFromImageData2D( const ImageRegionType &image, bool linearInterpolation )
{
	BOOST_STATIC_ASSERT(ImageRegionType::Dimension == 2);

	//Test whether image data have properties needed by OGL - continuous piece of memory
	ASSERT( image.GetStride( 0 ) == 1 );


	Vector< unsigned, 2 > size = image.GetSize();

	GLuint texName;

	// opengl texture setup functions
	glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
	glPixelStorei( GL_UNPACK_ROW_LENGTH, image.GetStride( 1 ) );
	glGenTextures( 1, &texName );

	glBindTexture ( GL_TEXTURE_2D, texName );
	glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );

	//glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, linearInterpolation ? GL_LINEAR : GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, linearInterpolation ? GL_LINEAR : GL_NEAREST );

	/*float scale = 2.0f;
	float bias = 0.0f;
	glPixelTransferf( GL_RED_SCALE, scale );
	glPixelTransferf( GL_GREEN_SCALE, scale );
	glPixelTransferf( GL_BLUE_SCALE, scale );

	glPixelTransferf( GL_RED_BIAS, bias );
	glPixelTransferf( GL_GREEN_BIAS, bias );
	glPixelTransferf( GL_BLUE_BIAS, bias );*/


	glEnable( GL_TEXTURE_2D );

	glBindTexture( GL_TEXTURE_2D, texName );

	glTexImage2D(	GL_TEXTURE_2D,
			0,
			GL_LUMINANCE16,
			size[0],
			size[1],
			0,
			GL_LUMINANCE,
			M4DToGLType< typename ImageRegionType::ElementType >::GLTypeID,
			image.GetPointer()
			);

	soglu::checkForGLError( "OGL building texture : " );
	D_PRINT( "2D texture created id = " << texName );
	return texName;
}

template< typename ImageRegionType >
GLuint
GLPrepareTextureFromImageData3D( const ImageRegionType &image, bool linearInterpolation )
{
	BOOST_STATIC_ASSERT(ImageRegionType::Dimension == 3);

	//Test whether image data have properties needed by OGL - continuous piece of memory
	ASSERT( image.GetStride( 0 ) == 1 );


	Vector< unsigned, 3 > size = image.GetSize();

	GLuint texName;

	// opengl texture setup functions
	GL_CHECKED_CALL( glPixelStorei( GL_UNPACK_ALIGNMENT, 1 ) );
	//glPixelStorei( GL_UNPACK_ROW_LENGTH, image.GetStride( 1 ) );
	GL_CHECKED_CALL( glPixelStorei(GL_PACK_ALIGNMENT, 1) );
	GL_CHECKED_CALL( glGenTextures( 1, &texName ) );

	GL_CHECKED_CALL( glBindTexture ( GL_TEXTURE_3D, texName ) );
	GL_ERROR_CLEAR_AFTER_CALL( glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE ) );

	//glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
	GL_CHECKED_CALL( glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE ) );
	GL_CHECKED_CALL( glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE ) );
	GL_CHECKED_CALL( glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE ) );
	GL_CHECKED_CALL( glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, linearInterpolation ? GL_LINEAR : GL_NEAREST ) );
	GL_CHECKED_CALL( glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, linearInterpolation ? GL_LINEAR : GL_NEAREST ) );


	GL_ERROR_CLEAR_AFTER_CALL( glEnable( GL_TEXTURE_3D ) ); //Opengl 3.3 throws error

	GL_CHECKED_CALL( glBindTexture( GL_TEXTURE_3D, texName ) );

	GL_CHECKED_CALL( glTexImage3D(	GL_TEXTURE_3D,
			0,
			M4DToGLTextureInternal< typename ImageRegionType::ElementType >::GLInternal,
			size[0],
			size[1],
			size[2],
			0,
			GL_RED,
			M4DToGLType< typename ImageRegionType::ElementType >::GLTypeID,
			image.GetPointer()
			) );

	soglu::checkForGLError( "OGL building texture : " );
	D_PRINT( "3D texture created id = " << texName );
	return texName;
}

template< typename ImageRegionType >
void
GLUpdateTextureFromSubImageData2D( GLuint aTexture, const ImageRegionType &image, Vector2i aMinimum, Vector2i aMaximum )
{
	BOOST_STATIC_ASSERT(ImageRegionType::Dimension == 2);

	//Test whether image data have properties needed by OGL - continuous piece of memory
	ASSERT( image.GetStride( 0 ) == 1 );


	Vector2u size = aMaximum - aMinimum;
	Vector2i offset = aMinimum - image.GetMinimum();

	// opengl texture setup functions
	GL_CHECKED_CALL( glPixelStorei( GL_UNPACK_ALIGNMENT, 1 ) );
	//glPixelStorei( GL_UNPACK_ROW_LENGTH, image.GetStride( 1 ) );
	GL_CHECKED_CALL( glPixelStorei(GL_PACK_ALIGNMENT, 1) );

	GL_CHECKED_CALL( glBindTexture ( GL_TEXTURE_2D, aTexture ) );
	GL_ERROR_CLEAR_AFTER_CALL( glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE ) );

	GL_ERROR_CLEAR_AFTER_CALL( glEnable( GL_TEXTURE_2D ) ); //Opengl 3.3 throws error

	GL_CHECKED_CALL( glBindTexture( GL_TEXTURE_2D, aTexture ) );

	GL_CHECKED_CALL( glTexSubImage2D(
			GL_TEXTURE_2D,
			0,
			offset[0],
			offset[1],
			size[0],
			size[1],
			GL_RED,
			M4DToGLType< typename ImageRegionType::ElementType >::GLTypeID,
			image.GetPointer()
		       ) );

	soglu::checkForGLError( "OGL updating texture : " );
	D_PRINT( "2D texture updated id = " << aTexture );
}

template< typename ImageRegionType >
void
GLUpdateTextureFromSubImageData3D( GLuint aTexture, const ImageRegionType &image, Vector3i aMinimum, Vector3i aMaximum )
{
	BOOST_STATIC_ASSERT(ImageRegionType::Dimension == 3);

	//Test whether image data have properties needed by OGL - continuous piece of memory
	ASSERT( image.GetStride( 0 ) == 1 );
	ASSERT( glIsTexture( aTexture ) );

	Vector3u size = aMaximum - aMinimum;
	Vector3i offset = aMinimum - image.GetMinimum();

	// opengl texture setup functions
	GL_CHECKED_CALL( glPixelStorei( GL_UNPACK_ALIGNMENT, 1 ) );
	GL_CHECKED_CALL( glPixelStorei( GL_UNPACK_ROW_LENGTH, image.GetStride( 1 ) ) );
	GL_CHECKED_CALL( glPixelStorei( GL_UNPACK_IMAGE_HEIGHT, image.GetStride( 2 ) / image.GetStride( 1 ) ) );
	//glPixelStorei( GL_UNPACK_ROW_LENGTH, image.GetStride( 1 ) );
	GL_CHECKED_CALL( glPixelStorei(GL_PACK_ALIGNMENT, 1) );

	GL_CHECKED_CALL( glBindTexture ( GL_TEXTURE_3D, aTexture ) );
	GL_ERROR_CLEAR_AFTER_CALL( glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE ) );

	GL_ERROR_CLEAR_AFTER_CALL( glEnable( GL_TEXTURE_3D ) ); //Opengl 3.3 throws error

	GL_CHECKED_CALL( glBindTexture( GL_TEXTURE_3D, aTexture ) );

	GL_CHECKED_CALL( glTexSubImage3D(
			GL_TEXTURE_3D,
			0,
			offset[0],
			offset[1],
			offset[2],
			size[0],
			size[1],
			size[2],
			GL_RED,
			M4DToGLType< typename ImageRegionType::ElementType >::GLTypeID,
			image.GetPointer() + offset * image.GetStride()
		       ) );

	soglu::checkForGLError( "OGL updating texture : " );
	D_PRINT( "3D texture updated id = " << aTexture );
}



} //namespace M4D
