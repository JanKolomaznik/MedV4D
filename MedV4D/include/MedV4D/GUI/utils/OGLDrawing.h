#ifndef OGL_DRAWING_H
#define OGL_DRAWING_H


#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/Sphere.h"
#include "MedV4D/GUI/utils/OGLTools.h"
#include "MedV4D/Imaging/AImage.h"
#include "MedV4D/Imaging/ImageRegion.h"
#include "MedV4D/Imaging/PointSet.h"
#include "MedV4D/Imaging/Mesh.h"
#include "MedV4D/Imaging/VertexInfo.h"
#include "MedV4D/Imaging/FaceInfo.h"
#include "MedV4D/GUI/utils/ViewConfiguration.h"
#include "MedV4D/GUI/utils/Camera.h"
#include "MedV4D/GUI/utils/OrthoCamera.h"
#include "MedV4D/GUI/utils/DrawingTools.h"

namespace M4D {


void
SetToViewConfiguration2D( const ViewConfiguration2D &config );


void
SetViewAccordingToCamera( const Camera &camera );

GLViewSetup
getViewSetupFromCamera( const Camera &camera );

GLViewSetup
getViewSetupFromOrthoCamera( const OrthoCamera &camera );

void
getProjectionAndViewMatricesFromCamera( const Camera &camera, glm::dmat4x4 &aProjection, glm::dmat4x4 &aView );

void
getProjectionAndViewMatricesFromOrthoCamera( const OrthoCamera &camera, glm::dmat4x4 &aProjection, glm::dmat4x4 &aView );

void
SetVolumeTextureCoordinateGeneration( const Vector< float, 3 > &minCoord, const Vector< float, 3 > &size );

void
DisableVolumeTextureCoordinateGeneration();

void
GLDrawVolumeSlices(
		const BoundingBox3D	&bbox,
		const Camera		&camera,
		unsigned 	numberOfSteps,
		float		cutPlane = 1.0f
		);

void
GLDrawVolumeSlices_Buffered(
		const BoundingBox3D	&bbox,
		const Camera		&camera,
		unsigned 	numberOfSteps,
		Vector3f	*vertices,
		unsigned	*indices,
		float		cutPlane = 1.0f
		);

void
GLDrawVolumeSliceCenterSamples(
		const BoundingBox3D	&bbox,
		const Camera		&camera,
		unsigned 	numberOfSteps,
		float		cutPlane
		);

void
GLDrawVolumeSlice(
		const Vector< float32, 3 > 	&min, 
		const Vector< float32, 3 > 	&max,
		float32				sliceCoord,
		CartesianPlanes			plane
		);

void
GLDrawVolumeSlice3D(
		const Vector< float32, 3 > 	&min, 
		const Vector< float32, 3 > 	&max,
		float32				sliceCoord,
		CartesianPlanes			plane
		);

void
GLDraw2DImage(
		const Vector< float32, 2 > 	&min, 
		const Vector< float32, 2 > 	&max
		);

template< typename ImageRegionType >
void
GLDrawImageData( const ImageRegionType &image, bool linearInterpolation );
	
	
void
GLDrawImageData( const M4D::Imaging::AImageRegionDim< 2 > &image, bool linearInterpolation );

void
DrawRectangleOverViewPort( const Vector2f &aFirst, const Vector2f &aSecond );

void
DrawRectangleOverViewPort( float aFirstX, float aFirstY, float aSecondX, float aSecondY );

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
//**************************************************************************************
template< typename ImageRegionType >
GLuint
GLPrepareTextureFromMaskData( const ImageRegionType &image, bool linearInterpolation );

GLuint
GLPrepareTextureFromMaskData( const M4D::Imaging::AImageRegionDim< 2 > &image, bool linearInterpolation );


void
GLDrawTexturedQuad( const Vector< float, 2 > &corner1, const Vector< float, 2 > &corner2 );


template< typename VectorType >
void
GLDrawPointSetPoints( const M4D::Imaging::Geometry::PointSet< VectorType > &pointset );

template< typename VectorType >
void
GLDrawPointSetLines( const M4D::Imaging::Geometry::PointSet< VectorType > &pointset, bool closed = false );

void
GLDrawBoundingBox( const BoundingBox3D &aBBox );

void
GLDrawBoundingBox( const Vector< float, 3 > &corner1, const Vector< float, 3 > &corner2 );

void
GLDrawBox( const Vector< float, 3 > &corner1, const Vector< float, 3 > &corner2 );

template< 
	typename CoordType, 
	template <typename CType> class NormalPartType, 
	typename ColorPartType
	>
inline void
GLVertexInfoDraw( const M4D::Imaging::Geometry::VertexInfo< CoordType, NormalPartType, ColorPartType > &vinfo );


template<
	typename VertexList,
	typename FaceInfoPartType,
	typename NormalPartType, 
	typename ColorPartType 
	>
inline void
GLFaceInfoDraw( const VertexList & vertices, const M4D::Imaging::Geometry::FaceInfo< FaceInfoPartType, NormalPartType, ColorPartType > &finfo );

template<
	typename VertexType,
	typename FaceType
	>
void
GLDrawMesh( const M4D::Imaging::Geometry::Mesh< VertexType, FaceType > &mesh );


void
drawCircle( float32 radius, size_t segCount = 32 );

void
drawCircle( Vector2f center, float32 radius, size_t segCount = 32 );

void
drawCircle( const Circlef &circle, size_t segCount = 32 );

void
drawCircleContour( float32 radius, size_t segCount = 32 );

void
drawCircleContour( Vector2f center, float32 radius, size_t segCount = 32 );

void
drawCircleContour( const Circlef &circle, size_t segCount = 32 );

void
drawSphere( float32 radius );

void
drawSphere( Vector3f center, float32 radius );

void
drawSphere( const Sphere3Df &sphere );

void
drawCylinder( float radius, float height );

void
drawCylinder( Vector3f aBaseCenter, Vector3f aBaseNormal, float radius, float height );

void
drawSphericalCap( float aBaseRadius, float aHeight );

void
drawSphericalCap( Vector3f aBaseCenter, Vector3f aBaseNormal, float aBaseRadius, float aHeight );

void
drawArrow( float32 arrowHeight, float32 bitHeight, float bitRadius, float bodyRadius1, float bodyRadius2 );

void
drawPlane( float aWidth, float aHeight );

void
drawPlane( const Vector3f &aCenter, const Vector3f &aVDirection, const Vector3f &aWDirection, float aWidth, float aHeight );

void
drawGrid( const Vector3f &aCenter, const Vector3f &aVDirection, const Vector3f &aWDirection, float aWidth, float aHeight, float aStep = 10.0f );

void
drawStippledLine( const Vector3f &aStart, const Vector3f &aEnd );

template< typename TIterator >
void
drawPointSet2D( TIterator aBegin, TIterator aEnd, Vector2f aInterval, CartesianPlanes aPlane )
{
	glBegin( GL_POINTS );
		for( TIterator it = aBegin; it != aEnd; ++it ) {
			if ( intervalTest( aInterval[0], aInterval[1], (*it)[aPlane] ) ) { 
				M4D::GLVertexVector( VectorPurgeDimension( *it, aPlane ) );
			}
		}
	glEnd();
}

template< typename TIterator >
void
drawLineSet2D( TIterator aBegin, TIterator aEnd, Vector2f aInterval, CartesianPlanes aPlane )
{
	glBegin( GL_LINES );
		for( TIterator it = aBegin; it != aEnd; ++it ) {
			if ( (it->firstPoint()[aPlane] < aInterval[0] && it->secondPoint()[aPlane] < aInterval[0])
				|| (it->firstPoint()[aPlane] > aInterval[1] && it->secondPoint()[aPlane] > aInterval[1]) )
			{ 
				continue;
			}
			M4D::GLVertexVector( VectorPurgeDimension( it->firstPoint(), aPlane ) );
			M4D::GLVertexVector( VectorPurgeDimension( it->secondPoint(), aPlane ) );
		}
	glEnd();
	glBegin( GL_POINTS );
		for( TIterator it = aBegin; it != aEnd; ++it ) {
			if ( intervalTest( aInterval[0], aInterval[1], it->firstPoint()[aPlane] ) ) { 
				M4D::GLVertexVector( VectorPurgeDimension( it->firstPoint(), aPlane ) );
			}
			if ( intervalTest( aInterval[0], aInterval[1], it->secondPoint()[aPlane] ) ) { 
				M4D::GLVertexVector( VectorPurgeDimension( it->secondPoint(), aPlane ) );
			}
		}
	glEnd();
}

template< typename TIterator >
void
drawPointSet( TIterator aBegin, TIterator aEnd )
{
	glBegin( GL_POINTS );
		for( TIterator it = aBegin; it != aEnd; ++it ) {
			M4D::GLVertexVector( *it );
		}
	glEnd();
}

template< typename TIterator >
void
drawLineSet( TIterator aBegin, TIterator aEnd )
{
	glBegin( GL_LINES );
		for( TIterator it = aBegin; it != aEnd; ++it ) {
			M4D::GLVertexVector( it->firstPoint() );
			M4D::GLVertexVector( it->secondPoint() );
		}
	glEnd();
	glBegin( GL_POINTS );
		for( TIterator it = aBegin; it != aEnd; ++it ) {
			M4D::GLVertexVector( it->firstPoint() );
			M4D::GLVertexVector( it->secondPoint() );
		}
	glEnd();
}


} /*namespace M4D*/

//include implementation
#include "MedV4D/GUI/utils/OGLDrawing.tcc"

#endif /*OGL_DRAWING_H*/
