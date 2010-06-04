#ifndef OGL_DRAWING_H
#define OGL_DRAWING_H


#include "common/Common.h"
#include "common/OGLTools.h"
#include "Imaging/ImageRegion.h"
#include "Imaging/PointSet.h"
#include "Imaging/Mesh.h"
#include "Imaging/VertexInfo.h"
#include "Imaging/FaceInfo.h"
#include "GUI/utils/ViewConfiguration.h"
#include "GUI/utils/Camera.h"
#include "GUI/utils/DrawingTools.h"

namespace M4D {

class GLException: public M4D::ErrorHandling::ExceptionBase
{
public:
	GLException( std::string name ) throw() : ExceptionBase( name ) {}
	~GLException() throw(){}
};

void 
CheckForGLError( const std::string &situation  );

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
GLDrawVolumeSlice(
		const Vector< float32, 3 > 	&min, 
		const Vector< float32, 3 > 	&max,
		float32				sliceCoord,
		CartesianPlanes			plane
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
GLPrepareTextureFromImageData( const M4D::Imaging::AImageRegionDim< 2 > &image, bool linearInterpolation );

GLuint
GLPrepareTextureFromImageData( const M4D::Imaging::AImageRegionDim< 3 > &image, bool linearInterpolation );



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

} /*namespace M4D*/

//include implementation
#include "GUI/utils/OGLDrawing.tcc"

#endif /*OGL_DRAWING_H*/
