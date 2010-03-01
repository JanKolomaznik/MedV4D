#ifndef OGL_DRAWING_H
#define OGL_DRAWING_H


#include "common/OGLTools.h"
#include "Imaging/ImageRegion.h"
#include "Imaging/PointSet.h"

namespace M4D {

	//*********************************************

	/*glLoadIdentity();

	uint32 height, width;
	glTranslatef( offset.x(), offset.y(), 0 );
	glScalef( _flipH * zoomRate, _flipV * zoomRate, 0. );*/

	//*********************************************

template< typename ImageRegionType >
void
GLDrawImageData( const ImageRegionType &image, bool linearInterpolation );
	
	
void
GLDrawImageData( const M4D::Imaging::AImageRegion &image, bool linearInterpolation )
{
	TYPE_TEMPLATE_SWITCH_MACRO( image.GetElementTypeID(), GLDrawImageData( static_cast< const M4D::Imaging::ImageRegion< TTYPE, 2 > &>( image ), linearInterpolation ); );
}

template< typename VectorType >
void
GLDrawPointSetPoints( const M4D::Imaging::Geometry::PointSet< VectorType > &pointset );

template< typename VectorType >
void
GLDrawPointSetLines( const M4D::Imaging::Geometry::PointSet< VectorType > &pointset, bool closed = false );


} /*namespace M4D*/

//include implementation
#include "GUI/utils/OGLDrawing.tcc"

#endif /*OGL_DRAWING_H*/
