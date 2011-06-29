#include "GUI/utils/OGLDrawing.h"

#ifdef __APPLE__
#include <OpenGL/glu.h>
#else
#include <GL/glu.h>
#endif


namespace M4D
{



void
SetToViewConfiguration2D( const ViewConfiguration2D &aConfig )
{
	GLint	viewportParams[4];
	glGetIntegerv( GL_VIEWPORT, viewportParams );

	Vector< float32, 2 > hsize = ( 0.5f / aConfig.zoom ) *  Vector< float32, 2 >( viewportParams[2], viewportParams[3] );
	Vector< float32, 2 > min =  aConfig.centerPoint - hsize;
	Vector< float32, 2 > max =  aConfig.centerPoint + hsize;
	if ( aConfig.hFlip ) {
		std::swap( min[0], max[0] );
	}
	if ( aConfig.vFlip ) {
		std::swap( min[1], max[1] );
	}


	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho( 
		(double)min[0], 
		(double)max[0], 
		(double)max[1], 
		(double)min[1], 
		-1.0, 
		1.0
		);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void
SetViewAccordingToCamera( const Camera &camera )
{	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(
		camera.GetFieldOfView(), 
 		camera.GetAspectRatio(), 
 		camera.GetZNear(), 
 		camera.GetZFar()
		);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(	
		camera.GetEyePosition()[0], 
		camera.GetEyePosition()[1], 
		camera.GetEyePosition()[2], 
		camera.GetTargetPosition()[0], 
		camera.GetTargetPosition()[1], 
		camera.GetTargetPosition()[2], 
		camera.GetUpDirection()[0], 
		camera.GetUpDirection()[1], 
		camera.GetUpDirection()[2]
		);
}

void
SetVolumeTextureCoordinateGeneration( const Vector< float, 3 > &minCoord, const Vector< float, 3 > &size )
{
	GLfloat params1[]={1.0f/size[0], 0.0f, 0.0f, -minCoord[0]/size[0]};
	GLfloat params2[]={0.0f, 1.0f/size[1], 0.0f, -minCoord[1]/size[1]};
	GLfloat params3[]={0.0f, 0.0f, 1.0f/size[2], -minCoord[2]/size[2]};


	glTexGeni( GL_S, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
	glTexGenfv(GL_S, GL_OBJECT_PLANE, params1);
	glEnable(GL_TEXTURE_GEN_S);

	glTexGeni( GL_T, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
	glTexGenfv(GL_T, GL_OBJECT_PLANE, params2);
	glEnable(GL_TEXTURE_GEN_T);
	
	glTexGeni( GL_R, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
	glTexGenfv(GL_R, GL_OBJECT_PLANE, params3);
	glEnable(GL_TEXTURE_GEN_R);
}

void
DisableVolumeTextureCoordinateGeneration()
{
	glDisable(GL_TEXTURE_GEN_S);
	glDisable(GL_TEXTURE_GEN_T);
	glDisable(GL_TEXTURE_GEN_R);
}

void
GLDrawVolumeSlices(
		BoundingBox3D	bbox,
		Camera		camera,
		unsigned 	numberOfSteps,
		float		cutPlane
		)
{
	Vector< float, 3> vertices[6];

	float 				min = 0; 
	float 				max = 0;
	unsigned			minId = 0;	
	unsigned			maxId = 0;	
	GetBBoxMinMaxDistance( 
		bbox, 
		camera.GetEyePosition(), 
		camera.GetTargetDirection(), 
		min, 
		max,
		minId,	
		maxId	
		);
	
	float stepSize = cutPlane * (max - min) / numberOfSteps;
	Vector< float, 3> planePoint = camera.GetEyePosition() + camera.GetTargetDirection() * max;
	for( unsigned i = 0; i < numberOfSteps; ++i ) {
		//Obtain intersection of the optical axis and the currently rendered plane
		planePoint -= stepSize * camera.GetTargetDirection();
		//Get n-gon as intersection of the current plane and bounding box
		unsigned count = M4D::GetPlaneVerticesInBoundingBox( 
				bbox, planePoint, camera.GetTargetDirection(), minId, vertices
				);

		//Render n-gon
		glBegin( GL_POLYGON );
			for( unsigned j = 0; j < count; ++j ) {
				GLVertexVector( vertices[ j ] );
			}
		glEnd();
	}

}

void
GLDrawVolumeSliceCenterSamples(
		BoundingBox3D	bbox,
		Camera		camera,
		unsigned 	numberOfSteps,
		float		cutPlane
		)
{
	Vector< float, 3> vertices[6];

	float 				min = 0; 
	float 				max = 0;
	unsigned			minId = 0;	
	unsigned			maxId = 0;	
	GetBBoxMinMaxDistance( 
		bbox, 
		camera.GetEyePosition(), 
		camera.GetTargetDirection(), 
		min, 
		max,
		minId,	
		maxId	
		);
	
	numberOfSteps = 4; //**************
	float stepSize = cutPlane * (max - min) / numberOfSteps;
	Vector< float, 3> planePoint = camera.GetEyePosition() + camera.GetTargetDirection() * max;
	for( unsigned i = 0; i < numberOfSteps; ++i ) {
		//Obtain intersection of the optical axis and the currently rendered plane
		planePoint -= stepSize * camera.GetTargetDirection();
		//Get n-gon as intersection of the current plane and bounding box
		/*unsigned count = M4D::GetPlaneVerticesInBoundingBox( 
				bbox, planePoint, camera.GetTargetDirection(), minId, vertices
				);*/


		glBegin( GL_POINTS );
				GLVertexVector( planePoint );
				//GLVertexVector( planePoint- Vector3f( 22.0f, 0.0f, 0.0f ) );
				//GLVertexVector( planePoint- Vector3f( 0.0f, 0.0f, 22.0f ) );
		glEnd();
		/*glBegin( GL_TRIANGLE_FAN );
			for( unsigned j = 0; j < count; ++j ) {
				GLVertexVector( vertices[ j ] );
			}
		glEnd();*/
	}

}




void
GLDrawVolumeSlice(
		const Vector< float32, 3 > 	&min, 
		const Vector< float32, 3 > 	&max,
		float32				sliceCoord,
		CartesianPlanes			plane
		)
{
	float32 sliceTexCoord = (sliceCoord - min[plane]) / (max[plane] - min[plane]);
	Vector< float32, 2 > point1 = VectorPurgeDimension( min, plane );
	Vector< float32, 2 > point3 = VectorPurgeDimension( max, plane );

	Vector< float32, 2 > point2( point3[0], point1[1] );
	Vector< float32, 2 > point4( point1[0], point3[1] );

	Vector< float32, 3 > tex1 = VectorInsertDimension( Vector< float32, 2 >( 0.0f, 0.0f ), sliceTexCoord, plane );
	Vector< float32, 3 > tex2 = VectorInsertDimension( Vector< float32, 2 >( 1.0f, 0.0f ), sliceTexCoord, plane );
	Vector< float32, 3 > tex3 = VectorInsertDimension( Vector< float32, 2 >( 1.0f, 1.0f ), sliceTexCoord, plane );
	Vector< float32, 3 > tex4 = VectorInsertDimension( Vector< float32, 2 >( 0.0f, 1.0f ), sliceTexCoord, plane );

	//std::cout << sliceCoord << "  " << sliceTexCoord << " tex\n";
	glBegin( GL_QUADS );
		GLTextureVector( tex1 ); 
		GLVertexVector( point1 );

		GLTextureVector( tex2 ); 
		GLVertexVector( point2 );

		GLTextureVector( tex3 ); 
		GLVertexVector( point3 );

		GLTextureVector( tex4 ); 
		GLVertexVector( point4 );
	glEnd();
}

void
GLDraw2DImage(
		const Vector< float32, 2 > 	&min, 
		const Vector< float32, 2 > 	&max
		)
{
	Vector< float32, 2 > point1 = min;
	Vector< float32, 2 > point3 = max;

	Vector< float32, 2 > point2( point3[0], point1[1] );
	Vector< float32, 2 > point4( point1[0], point3[1] );

	Vector< float32, 2 > tex1 = Vector< float32, 2 >( 0.0f, 0.0f );
	Vector< float32, 2 > tex2 = Vector< float32, 2 >( 1.0f, 0.0f );
	Vector< float32, 2 > tex3 = Vector< float32, 2 >( 1.0f, 1.0f );
	Vector< float32, 2 > tex4 = Vector< float32, 2 >( 0.0f, 1.0f );

	glBegin( GL_QUADS );
		GLTextureVector( tex1 ); 
		GLVertexVector( point1 );

		GLTextureVector( tex2 ); 
		GLVertexVector( point2 );

		GLTextureVector( tex3 ); 
		GLVertexVector( point3 );

		GLTextureVector( tex4 ); 
		GLVertexVector( point4 );
	glEnd();
}


void
GLDrawImageData( const M4D::Imaging::AImageRegionDim< 2 > &image, bool linearInterpolation )
{
	TYPE_TEMPLATE_SWITCH_MACRO( image.GetElementTypeID(), GLDrawImageData( static_cast< const M4D::Imaging::ImageRegion< TTYPE, 2 > &>( image ), linearInterpolation ); );
}

void
GLDrawImageDataContrastBrightness( const M4D::Imaging::AImageRegionDim< 2 > &image, float brightness, float contrast, bool linearInterpolation )
{
	TYPE_TEMPLATE_SWITCH_MACRO( image.GetElementTypeID(), GLDrawImageDataContrastBrightness( static_cast< const M4D::Imaging::ImageRegion< TTYPE, 2 > &>( image ), brightness, contrast, linearInterpolation ); );
}

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


GLuint
GLPrepareTextureFromMaskData( const M4D::Imaging::AImageRegionDim< 2 > &image, bool linearInterpolation )
{
	TYPE_TEMPLATE_SWITCH_MACRO( image.GetElementTypeID(), return GLPrepareTextureFromImageData( static_cast< const M4D::Imaging::ImageRegion< TTYPE, 2 > &>( image ), linearInterpolation ); );
	ASSERT( false );
	return 0;
}

void
GLDrawTexturedQuad( const Vector< float, 2 > &point1, const Vector< float, 2 > &point3 )
{
	Vector< float, 2 > point2( point3[0], point1[1] );
	Vector< float, 2 > point4( point1[0], point3[1] );

	// draw surface and map texture on it
	glBegin( GL_QUADS );
		glTexCoord2d( 0.0, 0.0 ); 
		GLVertexVector( point1 );

		glTexCoord2d( 1.0, 0.0 ); 
		GLVertexVector( point2 );

		glTexCoord2d( 1.0, 1.0 ); 
		GLVertexVector( point3 );

		glTexCoord2d( 0.0, 1.0 );
		GLVertexVector( point4 );
	glEnd();
}

void
GLDrawBoundingBox( const Vector< float, 3 > &corner1, const Vector< float, 3 > &corner2 )
{
	Vector< float, 3 > v1( corner1 );
	Vector< float, 3 > v2( corner2[0], corner1[1], corner1[2] );
	Vector< float, 3 > v3( corner2[0], corner2[1], corner1[2] );
	Vector< float, 3 > v4( corner1[0], corner2[1], corner1[2] );

	Vector< float, 3 > v5( corner1[0], corner1[1], corner2[2] );
	Vector< float, 3 > v6( corner2[0], corner1[1], corner2[2] );
	Vector< float, 3 > v7( corner2 );
	Vector< float, 3 > v8( corner1[0], corner2[1], corner2[2] );

	glBegin( GL_LINE_LOOP );
		GLVertexVector( v1 );
		GLVertexVector( v2 );
		GLVertexVector( v3 );
		GLVertexVector( v4 );
		GLVertexVector( v8 );
		GLVertexVector( v7 );
		GLVertexVector( v6 );
		GLVertexVector( v5 );
	glEnd();
	glBegin( GL_LINES );
		GLVertexVector( v1 );
		GLVertexVector( v4 );

		GLVertexVector( v2 );
		GLVertexVector( v6 );

		GLVertexVector( v3 );
		GLVertexVector( v7 );

		GLVertexVector( v5 );
		GLVertexVector( v8 );
	glEnd();
}

void
GLDrawBox( const Vector< float, 3 > &corner1, const Vector< float, 3 > &corner2 )
{
	Vector< float, 3 > v1( corner1 );
	Vector< float, 3 > v2( corner2[0], corner1[1], corner1[2] );
	Vector< float, 3 > v3( corner2[0], corner2[1], corner1[2] );
	Vector< float, 3 > v4( corner1[0], corner2[1], corner1[2] );

	Vector< float, 3 > v5( corner1[0], corner1[1], corner2[2] );
	Vector< float, 3 > v6( corner2[0], corner1[1], corner2[2] );
	Vector< float, 3 > v7( corner2 );
	Vector< float, 3 > v8( corner1[0], corner2[1], corner2[2] );

	ASSERT( false && "NOT FINISHED" );
	glBegin( GL_QUADS );
		GLVertexVector( v1 );
		GLVertexVector( v2 );
		GLVertexVector( v3 );
		GLVertexVector( v4 );

		GLVertexVector( v5 );
		GLVertexVector( v6 );
		GLVertexVector( v7 );
		GLVertexVector( v8 );

		GLVertexVector( v5 );
		GLVertexVector( v6 );
		GLVertexVector( v2 );
		GLVertexVector( v1 );
	glEnd();
}

void
DrawCircle( float32 radius, size_t segCount )
{
	float sAlpha = sin( 2*PI / segCount );
	float cAlpha = cos( 2*PI / segCount );

	Vector< float, 2 > v( radius, 0.0f );
	glBegin( GL_TRIANGLE_FAN );
	GLVertexVector( Vector2f() );
	for( size_t i = 0; i < segCount; ++i ) {
		GLVertexVector( v );
		v = Vector< float, 2 >( v[0] * cAlpha - v[1] * sAlpha, v[0] * sAlpha + v[1] * cAlpha );
	}
	GLVertexVector( Vector2f(radius,0.0f) );
	glEnd();
}

void
DrawCircle( Vector2f center, float32 radius, size_t segCount )
{
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glTranslatef( center[0], center[1], 0.0f );
	DrawCircle( radius, segCount );
	glPopMatrix();
}

void
DrawCircle( const Circlef &circle, size_t segCount )
{
	DrawCircle( circle.center(), circle.radius(), segCount );
}

void
DrawCircleContour( float32 radius, size_t segCount )
{
	float sAlpha = sin( 2*PI / segCount );
	float cAlpha = cos( 2*PI / segCount );

	Vector< float, 2 > v( radius, 0.0f );
	glBegin( GL_LINE_LOOP );
	GLVertexVector( Vector2f() );
	for( size_t i = 0; i < segCount; ++i ) {
		GLVertexVector( v );
		v = Vector< float, 2 >( v[0] * cAlpha - v[1] * sAlpha, v[0] * sAlpha + v[1] * cAlpha );
	}
	glEnd();
}

void
DrawCircleContour( Vector2f center, float32 radius, size_t segCount )
{
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glTranslatef( center[0], center[1], 0.0f );
	DrawCircle( radius, segCount );
	glPopMatrix();
}

void
DrawCircleContour( const Circlef &circle, size_t segCount )
{
	DrawCircle( circle.center(), circle.radius(), segCount );
}



void
DrawSphere( float32 radius )
{
	GLUquadric* quadratic=gluNewQuadric();			
	gluQuadricNormals(quadratic, GLU_SMOOTH);
	gluQuadricTexture(quadratic, GL_TRUE);

	gluSphere(quadratic,radius,32,32);

	gluDeleteQuadric(quadratic);
}

void
DrawSphere( Vector3f center, float32 radius )
{
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glTranslatef( center[0], center[1], center[2] );
	DrawSphere( radius );
	glPopMatrix();
}

void
DrawSphere( const Sphere3Df &sphere )
{
	DrawSphere( sphere.center(), sphere.radius() );
}

void
DrawArrow( float arrowHeight, float bitHeight, float bitRadius, float bodyRadius1, float bodyRadius2 )
{
	assert( arrowHeight > bitHeight );
	assert( bitRadius > bodyRadius1 );

	size_t segCount = 32;
	GLUquadric* quadratic=gluNewQuadric();			
	gluQuadricNormals(quadratic, GLU_SMOOTH);
	gluQuadricTexture(quadratic, GL_TRUE);

	gluCylinder( quadratic, 0.0, bitRadius, bitHeight, segCount, 2 );
	glPushMatrix();
	glTranslatef( 0.0f, 0.0f, bitHeight );

	gluCylinder( quadratic, bodyRadius1, bodyRadius2, arrowHeight - bitHeight, segCount, 2 );

	glPopMatrix();
	gluDeleteQuadric(quadratic);
}


} /*namespace M4D*/

