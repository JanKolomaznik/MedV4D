/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AImage.tcc 
 * @{ 
 **/

#ifndef OGL_DRAWING_H
#error File OGLDrawing.tcc cannot be included directly!
#else

#include <algorithm>
#include <boost/static_assert.hpp>
namespace M4D
{

template< typename ImageRegionType >
void
GLDrawImageData( const ImageRegionType &image, bool linearInterpolation )
{
	BOOST_STATIC_ASSERT(ImageRegionType::Dimension == 2);
	
	//Test whether image data have properties needed by OGL - continuous piece of memory
	ASSERT( image.GetStride( 0 ) == 1 );
	ASSERT( image.GetStride( 1 ) == (int32)image.GetSize( 0 ) );
	
	
	Vector< uint32, 2 > size = image.GetSize();
	Vector< float32, 2 > extents = image.GetElementExtents();

	GLuint texName;

	// opengl texture setup functions
	glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
	glGenTextures( 1, &texName );

	glBindTexture ( GL_TEXTURE_2D, texName );
	glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );

	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, linearInterpolation ? GL_LINEAR : GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, linearInterpolation ? GL_LINEAR : GL_NEAREST );


	glEnable( GL_TEXTURE_2D );
	glBindTexture( GL_TEXTURE_2D, texName );

	glTexImage2D(	GL_TEXTURE_2D, 
			0, 
			GL_LUMINANCE, 
			size[0], 
			size[1], 
			0, 
			GL_LUMINANCE, 
			M4DToGLType< typename ImageRegionType::ElementType >::GLTypeID, 
			image.GetPointer() 
			);


	// draw surface and map texture on it
	glBegin( GL_QUADS );
		glTexCoord2d( 0.0, 0.0 ); 
		glVertex2d( 0.0, 0.0 );

		glTexCoord2d( 1.0, 0.0 ); 
		glVertex2d( size[ 0 ] * extents[ 0 ], 0.0);

		glTexCoord2d( 1.0, 1.0 ); 
		glVertex2d( size[ 0 ] * extents[ 0 ], size[ 1 ] * extents[ 1 ] );

		glTexCoord2d( 0.0, 1.0 );
		glVertex2d( 0.0, size[ 1 ] * extents[ 1 ] );
	glEnd();

	glDeleteTextures( 1, &texName );
	glFlush();
}

template< typename VectorType >
void
GLDrawPointSetPoints( const M4D::Imaging::Geometry::PointSet< VectorType > &pointset )
{
	glBegin( GL_POINTS );
		std::for_each( pointset.Begin(), pointset.End(), GLVertexVector );
	glEnd();
}

template< typename VectorType >
void
GLDrawPointSetLines( const M4D::Imaging::Geometry::PointSet< VectorType > &pointset, bool closed = false )
{
	glBegin( closed ? GL_LINE_LOOP : GL_LINE_STRIP );
		std::for_each( pointset.Begin(), pointset.End(), GLVertexVector );
	glEnd();
}

} /*namespace M4D*/


#endif /*A_IMAGE_H*/

/** @} */
