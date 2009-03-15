#ifndef OGL_DRAWING_H
#define OGL_DRAWING_H

#include "Common.h"
#include "Imaging.h"

inline void
GLPoint( const Vector< float32, 2 > &point )
{
	glVertex2f( point[0], point[1] );
}

inline void
GLDrawPointSize( const Vector< float32, 2 > &point, float32 size )
{
	float32 tmp;
	glGetFloatv( GL_POINT_SIZE, &tmp );
	glPointSize( size );

	glBegin( GL_POINTS );
		glVertex2f( point[0], point[1] );
	glEnd();

	glPointSize( tmp );
}

inline void
GLDrawPolyline( const M4D::Imaging::Geometry::Polyline< float32, 2 > &polyline )
{
	int style = polyline.Cyclic() ? GL_LINE_LOOP : GL_LINE_STRIP;
		
	glBegin( style );
		std::for_each( polyline.Begin(), polyline.End(), GLPoint );
	glEnd();
}

inline void
GLDrawBSpline( const M4D::Imaging::Geometry::BSpline< float32, 2 > &spline )
{
	GLDrawPolyline( spline.GetSamplePoints() );
}

inline void
GLDrawBSplineCP( const M4D::Imaging::Geometry::BSpline< float32, 2 > &spline )
{
	GLDrawPolyline( spline.GetSamplePoints() );
	
	float32 pointSize = 3.0f;
	float32 tmp;
	glGetFloatv( GL_POINT_SIZE, &tmp );
	glPointSize( pointSize );

	glBegin( GL_POINTS );
		std::for_each( spline.Begin(), spline.End(), GLPoint );
	glEnd();

	glPointSize( tmp );
}

inline void
GLDrawCrossMark( Vector< float32, 2 > center, float32 radius )
{
	static const float32 cosA = cos( PI / 6.0 );
	static const float32 sinA = sin( PI / 6.0 );
	static const float32 sqrt2inv = 1.0 / sqrt( 2.0 );
	float32 pomR = radius * sqrt2inv;

	glPushMatrix();

	glTranslatef( center[0], center[1], 0.0f );
	glBegin( GL_LINES );
		glVertex2f( pomR, pomR );		
		glVertex2f( -pomR, -pomR );		
		glVertex2f( -pomR, pomR );		
		glVertex2f( pomR, -pomR );		
	glEnd();

	glBegin( GL_LINE_LOOP );
		float32 px = pomR;
		float32 py = pomR;
		for( int i = 0; i < 12; ++i ) {
			glVertex2f( px, py );
			float32 oldpx = px;
			px = px * cosA - py * sinA;
			py = oldpx * sinA + py * cosA;
		}
	glEnd();

	glPopMatrix();
}

#endif /*OGL_DRAWING_H*/
