#include "MedV4D/GUI/utils/OGLDrawing.h"
#include <glm/glm.hpp>
#include <glm/ext.hpp>

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

/*void
aaaSetToViewConfiguration2D( const ViewConfiguration2D &aConfig )
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
}*/


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

GLViewSetup
getViewSetupFromCamera( const Camera &camera )
{
	GLViewSetup setup;
	
	getProjectionAndViewMatricesFromCamera( camera, setup.projection, setup.view );
	setup.modelView = setup.view;
	setup.modelViewProj = setup.projection * setup.modelView;
	return setup;
}

GLViewSetup
getViewSetupFromOrthoCamera( const OrthoCamera &camera )
{
	GLViewSetup setup;
	
	getProjectionAndViewMatricesFromOrthoCamera( camera, setup.projection, setup.view );
	setup.modelView = setup.view;
	setup.modelViewProj = setup.projection * setup.modelView;
	return setup;
}

void
getProjectionAndViewMatricesFromCamera( const Camera &camera, glm::dmat4x4 &aProjection, glm::dmat4x4 &aView )
{	

	aProjection = glm::perspective<double>(
		camera.GetFieldOfView(), 
 		camera.GetAspectRatio(), 
 		camera.GetZNear(), 
 		camera.GetZFar()
	);
	
	aView = glm::lookAt<double>( 
		glm::dvec3( camera.GetEyePosition()[0], camera.GetEyePosition()[1], camera.GetEyePosition()[2] ),
		glm::dvec3( camera.GetTargetPosition()[0], camera.GetTargetPosition()[1], camera.GetTargetPosition()[2] ),
		glm::dvec3( camera.GetUpDirection()[0], camera.GetUpDirection()[1], camera.GetUpDirection()[2] )
	);
}

void
getProjectionAndViewMatricesFromOrthoCamera( const OrthoCamera &camera, glm::dmat4x4 &aProjection, glm::dmat4x4 &aView )
{
	aProjection = glm::ortho(
				camera.GetLeft(),
				camera.GetRight(),
				camera.GetBottom(),
				camera.GetTop(),
				camera.GetNear(),
				camera.GetFar()
			   );
	
	aView = glm::lookAt<double>( 
		glm::dvec3( camera.GetEyePosition()[0], camera.GetEyePosition()[1], camera.GetEyePosition()[2] ),
		glm::dvec3( camera.GetTargetPosition()[0], camera.GetTargetPosition()[1], camera.GetTargetPosition()[2] ),
		glm::dvec3( camera.GetUpDirection()[0], camera.GetUpDirection()[1], camera.GetUpDirection()[2] )
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
		const BoundingBox3D	&bbox,
		const Camera		&camera,
		unsigned 		numberOfSteps,
		float			cutPlane
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
GLDrawVolumeSlices_Buffered(
		const BoundingBox3D	&bbox,
		const Camera		&camera,
		unsigned 		numberOfSteps,
		Vector3f		*vertices,
		unsigned		*indices,
		float			cutPlane
		)
{
	//Vector3f *vertices = new Vector3f[ (numberOfSteps+1) * 6 ];
	//unsigned *indices = new unsigned[ (numberOfSteps+1) * 7 ];
	unsigned primitiveRestart = numberOfSteps * 20;

	ASSERT( GL_VERSION_3_1 );
	ASSERT( glPrimitiveRestartIndex != NULL );
	GL_CHECKED_CALL( glEnable(GL_PRIMITIVE_RESTART) );
	GL_CHECKED_CALL( glPrimitiveRestartIndex(primitiveRestart) );
	
#ifdef OPTIMIZED_VERSION_FFF //not working well yet
	size_t indicesSize = fillPlaneBBoxIntersectionBufferFill( bbox, camera, numberOfSteps, vertices, indices, cutPlane, primitiveRestart	);
#else
	ASSERT( vertices );
	ASSERT( indices );

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

	Vector3f *currentVertexPtr = vertices;
	unsigned *currentIndexPtr = indices;
	size_t primitiveStartIndex = 0;
	size_t indicesSize = 0;
	for( unsigned i = 0; i < numberOfSteps; ++i ) {
		//Obtain intersection of the optical axis and the currently rendered plane
		planePoint -= stepSize * camera.GetTargetDirection();
		//Get n-gon as intersection of the current plane and bounding box
		unsigned count = M4D::GetPlaneVerticesInBoundingBox( 
				bbox, planePoint, camera.GetTargetDirection(), minId, currentVertexPtr
				);

		currentVertexPtr += count;
		primitiveStartIndex += count;
		//currentVertexPtr += 6;
		//primitiveStartIndex += 6;
		for( unsigned j = 0; j < count; ++j ) {
			*(currentIndexPtr++) = primitiveStartIndex + j;
		}
		*(currentIndexPtr++) = primitiveRestart;
		indicesSize += count+1;
		/*for( unsigned j = count; j <= 6; ++j ) {
			*(currentIndexPtr++) = primitiveRestart;
		}
		indicesSize += 7;*/
	}
#endif
	GL_CHECKED_CALL( glEnableClientState(GL_VERTEX_ARRAY) );
	GL_CHECKED_CALL( glVertexPointer( 3, GL_FLOAT, 0, vertices ) );
	GL_CHECKED_CALL( glDrawElements(GL_TRIANGLE_FAN, indicesSize, GL_UNSIGNED_INT, indices) );
	GL_CHECKED_CALL( glDisableClientState(GL_VERTEX_ARRAY) );
	GL_CHECKED_CALL( glDisable(GL_PRIMITIVE_RESTART) );

	//delete [] vertices;
	//delete [] indices;
}

void
GLDrawVolumeSliceCenterSamples(
		const BoundingBox3D	&bbox,
		const Camera		&camera,
		unsigned 	numberOfSteps,
		float		cutPlane
		)
{
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
	
	//numberOfSteps = 1; //**************
	float stepSize = cutPlane * (max - min) / numberOfSteps;
	Vector< float, 3> planePoint = camera.GetEyePosition() + camera.GetTargetDirection() * max;
	for( unsigned i = 0; i < numberOfSteps; ++i ) {
		//Obtain intersection of the optical axis and the currently rendered plane
		planePoint -= stepSize * camera.GetTargetDirection();

		glBegin( GL_POINTS );
				GLVertexVector( planePoint );
		glEnd();
	}
}

void
GLDrawVolumeSlice(
		const Vector< float32, 3 > 	&aMin, 
		const Vector< float32, 3 > 	&aMax,
		float32				sliceCoord,
		CartesianPlanes			plane
		)
{
	//float32 sliceTexCoord = (sliceCoord - aMin[plane]) / (aMax[plane] - aMin[plane]);
	Vector< float32, 2 > point1 = VectorPurgeDimension( aMin, plane );
	Vector< float32, 2 > point3 = VectorPurgeDimension( aMax, plane );

	Vector< float32, 2 > point2( point3[0], point1[1] );
	Vector< float32, 2 > point4( point1[0], point3[1] );

	/*Vector< float32, 3 > tex1 = VectorInsertDimension( Vector< float32, 2 >( 0.0f, 0.0f ), sliceTexCoord, plane );
	Vector< float32, 3 > tex2 = VectorInsertDimension( Vector< float32, 2 >( 1.0f, 0.0f ), sliceTexCoord, plane );
	Vector< float32, 3 > tex3 = VectorInsertDimension( Vector< float32, 2 >( 1.0f, 1.0f ), sliceTexCoord, plane );
	Vector< float32, 3 > tex4 = VectorInsertDimension( Vector< float32, 2 >( 0.0f, 1.0f ), sliceTexCoord, plane );*/

	Vector< float32, 3 > tex1 = VectorInsertDimension( point1, sliceCoord, plane );
	Vector< float32, 3 > tex2 = VectorInsertDimension( point2, sliceCoord, plane );
	Vector< float32, 3 > tex3 = VectorInsertDimension( point3, sliceCoord, plane );
	Vector< float32, 3 > tex4 = VectorInsertDimension( point4, sliceCoord, plane );

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
GLDrawVolumeSlice3D(
		const Vector< float32, 3 > 	&aMin, 
		const Vector< float32, 3 > 	&aMax,
		float32				sliceCoord,
		CartesianPlanes			plane
		)
{
	//float32 sliceTexCoord = (sliceCoord - aMin[plane]) / (aMax[plane] - aMin[plane]);
	Vector< float32, 2 > point1 = VectorPurgeDimension( aMin, plane );
	Vector< float32, 2 > point3 = VectorPurgeDimension( aMax, plane );

	Vector< float32, 2 > point2( point3[0], point1[1] );
	Vector< float32, 2 > point4( point1[0], point3[1] );

	/*Vector< float32, 3 > tex1 = VectorInsertDimension( Vector< float32, 2 >( 0.0f, 0.0f ), sliceTexCoord, plane );
	Vector< float32, 3 > tex2 = VectorInsertDimension( Vector< float32, 2 >( 1.0f, 0.0f ), sliceTexCoord, plane );
	Vector< float32, 3 > tex3 = VectorInsertDimension( Vector< float32, 2 >( 1.0f, 1.0f ), sliceTexCoord, plane );
	Vector< float32, 3 > tex4 = VectorInsertDimension( Vector< float32, 2 >( 0.0f, 1.0f ), sliceTexCoord, plane );*/

	Vector< float32, 3 > tex1 = VectorInsertDimension( point1, sliceCoord, plane );
	Vector< float32, 3 > tex2 = VectorInsertDimension( point2, sliceCoord, plane );
	Vector< float32, 3 > tex3 = VectorInsertDimension( point3, sliceCoord, plane );
	Vector< float32, 3 > tex4 = VectorInsertDimension( point4, sliceCoord, plane );

	//std::cout << sliceCoord << "  " << sliceTexCoord << " tex\n";
	glBegin( GL_QUADS );
		GLTextureVector( tex1 ); 
		GLVertexVector( tex1 );

		GLTextureVector( tex2 ); 
		GLVertexVector( tex2 );

		GLTextureVector( tex3 ); 
		GLVertexVector( tex3 );

		GLTextureVector( tex4 ); 
		GLVertexVector( tex4 );
	glEnd();
}

void
GLDraw2DImage(
		const Vector< float32, 2 > 	&aMin, 
		const Vector< float32, 2 > 	&aMax
		)
{
	Vector< float32, 2 > point1 = aMin;
	Vector< float32, 2 > point3 = aMax;

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
DrawRectangleOverViewPort( const Vector2f &aFirst, const Vector2f &aSecond )
{
	M4D::GLPushAtribs pushAttribs;
	GLPushMatrices pushMatrices;
	Vector4i viewport;
	GL_CHECKED_CALL( glDisable( GL_LIGHTING ) );
	GL_CHECKED_CALL( glGetIntegerv( GL_VIEWPORT, (GLint *)&viewport ) );
	
	GL_CHECKED_CALL( glMatrixMode(GL_PROJECTION) );
	GL_CHECKED_CALL( glLoadIdentity() );
	GL_CHECKED_CALL( glOrtho( 
		(double)viewport[0], 
		(double)viewport[0]+viewport[2],
		(double)viewport[1],
		(double)viewport[1]+viewport[3], 
		-1.0, 
		1.0
		) );

	GL_CHECKED_CALL( glMatrixMode(GL_MODELVIEW) );
	GL_CHECKED_CALL( glLoadIdentity() );
	
	Vector2f point1 = aFirst;
	Vector2f point3 = aSecond;

	Vector2f point2( point3[0], point1[1] );
	Vector2f point4( point1[0], point3[1] );
	
	glBegin( GL_QUADS );
		GLVertexVector( point1 );
		D_PRINT( point1 );

		GLVertexVector( point2 );

		GLVertexVector( point3 );

		GLVertexVector( point4 );
	glEnd();
	CheckForGLError( "DrawRectangleOverViewPort" );
}
void
DrawRectangleOverViewPort( float aFirstX, float aFirstY, float aSecondX, float aSecondY )
{
	DrawRectangleOverViewPort( Vector2f( aFirstX, aFirstY ), Vector2f( aSecondX, aSecondY ) );
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
GLDrawBoundingBox( const BoundingBox3D &aBBox )
{
	glBegin( GL_LINE_LOOP );
		GLVertexVector( aBBox.vertices[0] );
		GLVertexVector( aBBox.vertices[1] );
		GLVertexVector( aBBox.vertices[2] );
		GLVertexVector( aBBox.vertices[3] );
		GLVertexVector( aBBox.vertices[7] );
		GLVertexVector( aBBox.vertices[6] );
		GLVertexVector( aBBox.vertices[5] );
		GLVertexVector( aBBox.vertices[4] );
	glEnd();
	glBegin( GL_LINES );
		GLVertexVector( aBBox.vertices[0] );
		GLVertexVector( aBBox.vertices[3] );

		GLVertexVector( aBBox.vertices[1] );
		GLVertexVector( aBBox.vertices[5] );

		GLVertexVector( aBBox.vertices[2] );
		GLVertexVector( aBBox.vertices[6] );

		GLVertexVector( aBBox.vertices[4] );
		GLVertexVector( aBBox.vertices[7] );
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
drawCircle( float32 radius, size_t segCount )
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
drawCircle( Vector2f center, float32 radius, size_t segCount )
{
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glTranslatef( center[0], center[1], 0.0f );
	drawCircle( radius, segCount );
	glPopMatrix();
}

void
drawCircle( const Circlef &circle, size_t segCount )
{
	drawCircle( circle.center(), circle.radius(), segCount );
}

void
drawCircleContour( float32 radius, size_t segCount )
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
drawCircleContour( Vector2f center, float32 radius, size_t segCount )
{
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glTranslatef( center[0], center[1], 0.0f );
	drawCircle( radius, segCount );
	glPopMatrix();
}

void
drawCircleContour( const Circlef &circle, size_t segCount )
{
	drawCircle( circle.center(), circle.radius(), segCount );
}



void
drawSphere( float32 radius )
{
	GLUquadric* quadratic=gluNewQuadric();			
	gluQuadricNormals(quadratic, GLU_SMOOTH);
	gluQuadricTexture(quadratic, GL_TRUE);

	gluSphere(quadratic,radius,32,32);

	gluDeleteQuadric(quadratic);
}

void
drawSphere( Vector3f center, float32 radius )
{
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glTranslatef( center[0], center[1], center[2] );
	drawSphere( radius );
	glPopMatrix();
}

void
drawSphere( const Sphere3Df &sphere )
{
	drawSphere( sphere.center(), sphere.radius() );
}

void
drawCylinder( float radius, float height )
{
	GLUquadric* quadratic=gluNewQuadric();			
	gluQuadricNormals(quadratic, GLU_SMOOTH);
	gluQuadricTexture(quadratic, GL_TRUE);

	gluCylinder(quadratic,radius,radius,height,32,2);

	gluDeleteQuadric(quadratic);
}

void
drawCylinder( Vector3f aBaseCenter, Vector3f aBaseNormal, float radius, float height )
{
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glTranslatef( aBaseCenter[0], aBaseCenter[1], aBaseCenter[2] );
	
	Vector3f axis;
	float angle = angleAndRotationAxisFromVectors( Vector3f( 0.0f, 0.0f, 1.0f ), aBaseNormal, axis ) * 180.f / PI;
	glRotatef( angle, axis[0], axis[1], axis[2] );

	drawCylinder( radius, height );
	glPopMatrix();
}

void
drawSphericalCap( float aBaseRadius, float aHeight )
{
	ASSERT( aBaseRadius > 0.0f && aHeight > 0.0f );

	float radius = ( M4D::sqr( aBaseRadius ) + M4D::sqr( aHeight ) ) / ( 2 * aHeight );
	float iRadius = 1.0f / radius;
	const int latitudeSteps = 16;
	const int longitudeSteps = 32;

	//float minSin = (radius - aHeight)/radius;
	
	float lonSin = sin( 2*PI / longitudeSteps );
	float lonCos = cos( 2*PI / longitudeSteps );

	float latStep = aHeight / latitudeSteps;

	float r = sqrt( 2* radius * latStep - M4D::sqr( latStep ) );
	Vector3f tmp = Vector3f( r, 0.0f, aHeight - latStep );
	glBegin( GL_TRIANGLE_FAN );
		GLNormalVector( Vector3f( 0.0f, 0.0f, 1.0f ) );
		GLVertexVector( Vector3f( 0.0f, 0.0f, aHeight ) );
		for( int i = 0; i < longitudeSteps; ++i ) {
			GLNormalVector( iRadius * (tmp + Vector3f( 0.0f, 0.0f, radius - aHeight )) );
			GLVertexVector( tmp );
			tmp = Vector3f( tmp[0] * lonCos - tmp[1] * lonSin, tmp[0] * lonSin + tmp[1] * lonCos, aHeight - latStep );
		}
		GLNormalVector( iRadius * (Vector3f( r, 0.0f, aHeight - latStep ) + Vector3f( 0.0f, 0.0f, radius - aHeight )) );
		GLVertexVector( Vector3f( r, 0.0f, aHeight - latStep ) );
	glEnd();

	for( int j = 1; j < latitudeSteps; ++j ) {
		float h1 = aHeight - j * latStep;
		float h2 = aHeight - (j+1) * latStep;
		float r1 = sqrt( 2* radius * j * latStep - M4D::sqr( j * latStep ) ); //TODO - optimize
		float r2 = sqrt( 2* radius * (j+1) * latStep - M4D::sqr( (j+1) * latStep ) );
		Vector3f tmp1 = Vector3f( r1, 0.0f, h1 );
		Vector3f tmp2 = Vector3f( r2, 0.0f, h2 );
		glBegin( GL_TRIANGLE_STRIP );
			GLVertexVector( tmp1 );
			GLVertexVector( tmp2 );
			for( int i = 0; i < longitudeSteps; ++i ) {
				GLNormalVector( iRadius * (tmp1 + Vector3f( 0.0f, 0.0f, radius - aHeight )) );
				GLVertexVector( tmp1 );
				GLNormalVector( iRadius * (tmp2 + Vector3f( 0.0f, 0.0f, radius - aHeight )) );
				GLVertexVector( tmp2 );
				tmp1 = Vector3f( tmp1[0] * lonCos - tmp1[1] * lonSin, tmp1[0] * lonSin + tmp1[1] * lonCos, h1 );
				tmp2 = Vector3f( tmp2[0] * lonCos - tmp2[1] * lonSin, tmp2[0] * lonSin + tmp2[1] * lonCos, h2 );
			}
			GLNormalVector( iRadius * (Vector3f( r1, 0.0f, h1 ) + Vector3f( 0.0f, 0.0f, radius - aHeight )) );
			GLVertexVector( Vector3f( r1, 0.0f, h1 ) );
			GLNormalVector( iRadius * (Vector3f( r2, 0.0f, h2 ) + Vector3f( 0.0f, 0.0f, radius - aHeight )) );
			GLVertexVector( Vector3f( r2, 0.0f, h2 ) );
		glEnd();
	}
}

void
drawSphericalCap( Vector3f aBaseCenter, Vector3f aBaseNormal, float aBaseRadius, float aHeight )
{
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glTranslatef( aBaseCenter[0], aBaseCenter[1], aBaseCenter[2] );
	
	Vector3f axis;
	float angle = angleAndRotationAxisFromVectors( Vector3f( 0.0f, 0.0f, 1.0f ), aBaseNormal, axis ) * 180.f / PI;
	glRotatef( angle, axis[0], axis[1], axis[2] );

	drawSphericalCap( aBaseRadius, aHeight );
	glPopMatrix();
}


void
drawArrow( float arrowHeight, float bitHeight, float bitRadius, float bodyRadius1, float bodyRadius2 )
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

void
drawStippledLine( const Vector3f &aStart, const Vector3f &aEnd )
{
	glEnable( GL_LINE_STIPPLE );
	glLineStipple( 3,  0x7777 );
	glBegin( GL_LINES );
		GLVertexVector( aStart );
		GLVertexVector( aEnd );
	glEnd();
	glLineStipple( 1,  0xFFFF );
}

void
drawPlane( float aWidth, float aHeight )
{
	Vector< float32, 2 > point1( -0.5f*aWidth, -0.5f*aHeight );
	Vector< float32, 2 > point3 = -point1;

	Vector< float32, 2 > point2( point3[0], point1[1] );
	Vector< float32, 2 > point4( point1[0], point3[1] );

	glBegin( GL_QUADS );
		GLVertexVector( point1 );

		GLVertexVector( point2 );

		GLVertexVector( point3 );

		GLVertexVector( point4 );
	glEnd();
}

void
drawPlane( const Vector3f &aCenter, const Vector3f &aVDirection, const Vector3f &aWDirection, float aWidth, float aHeight )
{

}

void
drawGrid( const Vector3f &aCenter, const Vector3f &aVDirection, const Vector3f &aWDirection, float aWidth, float aHeight, float aStep )
{
	Vector3f vSize = 0.5f*aWidth*aVDirection;
	Vector3f wSize = 0.5f*aHeight*aWDirection;
	glBegin( GL_LINE_LOOP );	
		GLVertexVector( aCenter + vSize + wSize );

		GLVertexVector( aCenter + vSize - wSize );

		GLVertexVector( aCenter - vSize - wSize );

		GLVertexVector( aCenter - vSize + wSize );
	glEnd();

	int vCount = static_cast<int>( 0.5f*aWidth / aStep );
	int wCount = static_cast<int>( 0.5f*aHeight / aStep );
	glBegin( GL_LINES );
		for ( int i = -vCount; i <= vCount; ++i ) {
			GLVertexVector( aCenter + i*aStep*aVDirection + wSize );
			GLVertexVector( aCenter + i*aStep*aVDirection - wSize );
		}
		for ( int i = -wCount; i <= wCount; ++i ) {
			GLVertexVector( aCenter + i*aStep*aWDirection + vSize );
			GLVertexVector( aCenter + i*aStep*aWDirection - vSize );
		}
	glEnd();

	GL_CHECKED_CALL( glLineWidth( 3.5f ) );
	glBegin( GL_LINES );
			GLVertexVector( aCenter + wSize );
			GLVertexVector( aCenter - wSize );
			GLVertexVector( aCenter + vSize );
			GLVertexVector( aCenter - vSize );
	glEnd();
	GL_CHECKED_CALL( glLineWidth( 1.0f ) );
}


} /*namespace M4D*/

