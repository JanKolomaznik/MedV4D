#ifndef OGL_DRAWING_H
#define OGL_DRAWING_H


#include "common/OGLTools.h"
#include "Imaging/ImageRegion.h"

namespace M4D {

	//*********************************************

	/*glLoadIdentity();

	uint32 height, width;
	glTranslatef( offset.x(), offset.y(), 0 );
	glScalef( _flipH * zoomRate, _flipV * zoomRate, 0. );*/

	//*********************************************

template< typename ElementType >
void
GLDrawImageData( const M4D::Imaging::ImageRegion< ElementType, 2 > &image, bool linearInterpolation );
	
	
void
GLDrawImageData( const M4D::Imaging::AImageRegion &image, bool linearInterpolation )
{
	TYPE_TEMPLATE_SWITCH_MACRO( image.GetElementTypeID(), GLDrawImageData( static_cast< const M4D::Imaging::ImageRegion< TTYPE, 2 > &>( image ), linearInterpolation ); );
}


template< typename ElementType >
void
GLDrawImageData( const M4D::Imaging::ImageRegion< ElementType, 2 > &image, bool linearInterpolation )
{
	typedef M4D::Imaging::ImageRegion< ElementType, 2 > RegionType;

	
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
			M4DToGLType< ElementType >::GLTypeID, 
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


} /*namespace M4D*/

#endif /*OGL_DRAWING_H*/
