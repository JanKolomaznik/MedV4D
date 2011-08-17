#ifndef OGL_DRAWING_H
#define OGL_DRAWING_H


#include "common/Common.h"
#include "common/Sphere.h"
#include "GUI/utils/OGLTools.h"
#include "Imaging/AImage.h"
#include "Imaging/ImageRegion.h"
#include "Imaging/PointSet.h"
#include "Imaging/Mesh.h"
#include "Imaging/VertexInfo.h"
#include "Imaging/FaceInfo.h"
#include "GUI/utils/ViewConfiguration.h"
#include "GUI/utils/Camera.h"
#include "GUI/utils/DrawingTools.h"

namespace M4D {


void
SetToViewConfiguration2D( const ViewConfiguration2D &config );


void
SetViewAccordingToCamera( const Camera &camera );

void
SetVolumeTextureCoordinateGeneration( const Vector< float, 3 > &minCoord, const Vector< float, 3 > &size );

void
DisableVolumeTextureCoordinateGeneration();

void
GLDrawVolumeSlices(
		BoundingBox3D	bbox,
		Camera		camera,
		unsigned 	numberOfSteps,
		float		cutPlane = 1.0f
		);

void
GLDrawVolumeSliceCenterSamples(
		BoundingBox3D	bbox,
		Camera		camera,
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
GLDraw2DImage(
		const Vector< float32, 2 > 	&min, 
		const Vector< float32, 2 > 	&max
		);

template< typename ImageRegionType >
void
GLDrawImageData( const ImageRegionType &image, bool linearInterpolation );
	
	
void
GLDrawImageData( const M4D::Imaging::AImageRegionDim< 2 > &image, bool linearInterpolation );




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
DrawCircle( float32 radius, size_t segCount = 32 );

void
DrawCircle( Vector2f center, float32 radius, size_t segCount = 32 );

void
DrawCircle( const Circlef &circle, size_t segCount = 32 );

void
DrawCircleContour( float32 radius, size_t segCount = 32 );

void
DrawCircleContour( Vector2f center, float32 radius, size_t segCount = 32 );

void
DrawCircleContour( const Circlef &circle, size_t segCount = 32 );


void
DrawSphere( float32 radius );

void
DrawSphere( Vector3f center, float32 radius );

void
DrawSphere( const Sphere3Df &sphere );

void
DrawArrow( float32 arrowHeight, float32 bitHeight, float bitRadius, float bodyRadius1, float bodyRadius2 );

template< typename TIterator >
void
DrawPointSet2D( TIterator aBegin, TIterator aEnd, Vector2f aInterval, CartesianPlanes aPlane )
{
	glBegin( GL_POINTS );
		for( TIterator it = aBegin; it != aEnd; ++it ) {
			if ( IntervalTest( aInterval[0], aInterval[1], (*it)[aPlane] ) ) { 
				M4D::GLVertexVector( VectorPurgeDimension( *it, aPlane ) );
			}
		}
	glEnd();
}

template< typename TIterator >
void
DrawLineSet2D( TIterator aBegin, TIterator aEnd, Vector2f aInterval, CartesianPlanes aPlane )
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
			if ( IntervalTest( aInterval[0], aInterval[1], it->firstPoint()[aPlane] ) ) { 
				M4D::GLVertexVector( VectorPurgeDimension( it->firstPoint(), aPlane ) );
			}
			if ( IntervalTest( aInterval[0], aInterval[1], it->secondPoint()[aPlane] ) ) { 
				M4D::GLVertexVector( VectorPurgeDimension( it->secondPoint(), aPlane ) );
			}
		}
	glEnd();
}

template< typename TIterator >
void
DrawPointSet( TIterator aBegin, TIterator aEnd )
{
	glBegin( GL_POINTS );
		for( TIterator it = aBegin; it != aEnd; ++it ) {
			M4D::GLVertexVector( *it );
		}
	glEnd();
}

template< typename TIterator >
void
DrawLineSet( TIterator aBegin, TIterator aEnd )
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
#include "GUI/utils/OGLDrawing.tcc"

#endif /*OGL_DRAWING_H*/
