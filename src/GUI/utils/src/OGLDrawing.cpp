#include "GUI/utils/OGLDrawing.h"
#include <GL/glu.h>

PFNGLTEXIMAGE3DPROC glTexImage3D;

namespace M4D
{

void 
CheckForGLError( const std::string &situation  )
{
	GLenum errorCode = glGetError();
	if (errorCode != GL_NO_ERROR) {
		const char *string = (const char *)gluErrorString(errorCode);
		_THROW_ GLException( TO_STRING( situation << string ) );
	}
}


void
SetToViewConfiguration2D( const ViewConfiguration2D &config )
{

	glOrtho(0.0, (double)config.width, 0.0, (double)config.height, -1.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glTranslatef( config.offset[0], config.offset[1], 0 );
	glScalef( config.zoom * config.hFlip, config.zoom * config.vFlip, 0.0f );
}

void
SetViewAccordingToCamera( const Camera &camera )
{

	gluPerspective(
		camera.GetFieldOfView(), 
 		camera.GetAspectRatio(), 
 		camera.GetZNear(), 
 		camera.GetZFar()
		);

	gluLookAt(	
		camera.GetEyePosition()[0], 
		camera.GetEyePosition()[1], 
		camera.GetEyePosition()[2], 
		camera.GetCenterPosition()[0], 
		camera.GetCenterPosition()[1], 
		camera.GetCenterPosition()[2], 
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
		camera.GetCenterDirection(), 
		min, 
		max,
		minId,	
		maxId	
		);
	
	float stepSize = cutPlane * (max - min) / numberOfSteps;
	Vector< float, 3> planePoint = camera.GetEyePosition() + camera.GetCenterDirection() * max;
	for( unsigned i = 0; i < numberOfSteps; ++i ) {
		//Obtain intersection of the optical axis and the currently rendered plane
		planePoint -= stepSize * camera.GetCenterDirection();
		//Get n-gon as intersection of the current plane and bounding box
		unsigned count = M4D::GetPlaneVerticesInBoundingBox( 
				bbox, planePoint, camera.GetCenterDirection(), minId, vertices
				);

		//Render n-gon
		glBegin( GL_POLYGON );
			for( unsigned i = 0; i < count; ++i ) {
				GLVertexVector( vertices[ i ] );
			}
		glEnd();
	}

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


} /*namespace M4D*/

