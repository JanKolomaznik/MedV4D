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
	GLuint texName = GLPrepareTextureFromImageData( image, linearInterpolation );

	glBindTexture( GL_TEXTURE_2D, texName );

	GLDrawTexturedQuad( image.GetRealMinimum(), image.GetRealMaximum() );
	
	glDeleteTextures( 1, &texName );
	glFlush();//TODO decide removal
}

template< typename ImageRegionType >
GLuint
GLPrepareTextureFromImageData2D( const ImageRegionType &image, bool linearInterpolation )
{
	BOOST_STATIC_ASSERT(ImageRegionType::Dimension == 2);
	
	//Test whether image data have properties needed by OGL - continuous piece of memory
	ASSERT( image.GetStride( 0 ) == 1 );
	
	
	Vector< unsigned, 2 > size = image.GetSize();

	GLuint texName;

	// opengl texture setup functions
	glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
	glPixelStorei( GL_UNPACK_ROW_LENGTH, image.GetStride( 1 ) );
	glGenTextures( 1, &texName );

	glBindTexture ( GL_TEXTURE_2D, texName );
	glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );

	//glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, linearInterpolation ? GL_LINEAR : GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, linearInterpolation ? GL_LINEAR : GL_NEAREST );

	/*float scale = 2.0f;
	float bias = 0.0f;
	glPixelTransferf( GL_RED_SCALE, scale );
	glPixelTransferf( GL_GREEN_SCALE, scale );
	glPixelTransferf( GL_BLUE_SCALE, scale );

	glPixelTransferf( GL_RED_BIAS, bias );
	glPixelTransferf( GL_GREEN_BIAS, bias );
	glPixelTransferf( GL_BLUE_BIAS, bias );*/


	glEnable( GL_TEXTURE_2D );
	
	glBindTexture( GL_TEXTURE_2D, texName );

	glTexImage2D(	GL_TEXTURE_2D, 
			0, 
			GL_LUMINANCE16, 
			size[0], 
			size[1], 
			0, 
			GL_LUMINANCE, 
			M4DToGLType< typename ImageRegionType::ElementType >::GLTypeID, 
			image.GetPointer() 
			);

	M4D::CheckForGLError( "OGL building texture : " );
	D_PRINT( "2D texture created id = " << texName );
	return texName;
}

template< typename ImageRegionType >
GLuint
GLPrepareTextureFromImageData3D( const ImageRegionType &image, bool linearInterpolation )
{
	BOOST_STATIC_ASSERT(ImageRegionType::Dimension == 3);
	
	//Test whether image data have properties needed by OGL - continuous piece of memory
	ASSERT( image.GetStride( 0 ) == 1 );
	
	
	Vector< unsigned, 3 > size = image.GetSize();

	GLuint texName;

	// opengl texture setup functions
	glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
	//glPixelStorei( GL_UNPACK_ROW_LENGTH, image.GetStride( 1 ) );
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glGenTextures( 1, &texName );

	glBindTexture ( GL_TEXTURE_3D, texName );
	glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );

	//glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
	glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, linearInterpolation ? GL_LINEAR : GL_NEAREST );
	glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, linearInterpolation ? GL_LINEAR : GL_NEAREST );


	glEnable( GL_TEXTURE_3D );
	
	glBindTexture( GL_TEXTURE_3D, texName );

	glTexImage3D(	GL_TEXTURE_3D, 
			0, 
			M4DToGLTextureInternal< typename ImageRegionType::ElementType >::GLInternal, 
			size[0], 
			size[1], 
			size[2], 
			0, 
			GL_RED, 
			M4DToGLType< typename ImageRegionType::ElementType >::GLTypeID, 
			image.GetPointer() 
			);

	M4D::CheckForGLError( "OGL building texture : " );
	D_PRINT( "3D texture created id = " << texName );
	return texName;
}


template< typename ImageRegionType >
GLuint
GLPrepareTextureFromMaskData( const ImageRegionType &image, bool linearInterpolation )
{
	BOOST_STATIC_ASSERT(ImageRegionType::Dimension == 2);
	
	//Test whether image data have properties needed by OGL - continuous piece of memory
	ASSERT( image.GetStride( 0 ) == 1 );
	
	
	Vector< unsigned, 2 > size = image.GetSize();

	GLuint texName;

	// opengl texture setup functions
	glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
	glPixelStorei( GL_UNPACK_ROW_LENGTH, image.GetStride( 1 ) );
	glGenTextures( 1, &texName );

	glBindTexture ( GL_TEXTURE_2D, texName );
	//glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );

	glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, linearInterpolation ? GL_LINEAR : GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, linearInterpolation ? GL_LINEAR : GL_NEAREST );

	/*float scale = 2.0f;
	float bias = 0.0f;
	glPixelTransferf( GL_RED_SCALE, scale );
	glPixelTransferf( GL_GREEN_SCALE, scale );
	glPixelTransferf( GL_BLUE_SCALE, scale );

	glPixelTransferf( GL_RED_BIAS, bias );
	glPixelTransferf( GL_GREEN_BIAS, bias );
	glPixelTransferf( GL_BLUE_BIAS, bias );*/


	glEnable( GL_TEXTURE_2D );
	
	glBindTexture( GL_TEXTURE_2D, texName );

	glTexImage2D(	GL_TEXTURE_2D, 
			0, 
			GL_ALPHA, 
			size[0], 
			size[1], 
			0, 
			GL_ALPHA, 
			M4DToGLType< typename ImageRegionType::ElementType >::GLTypeID, 
			image.GetPointer() 
			);

	return texName;
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
GLDrawPointSetLines( const M4D::Imaging::Geometry::PointSet< VectorType > &pointset, bool closed )
{
	glBegin( closed ? GL_LINE_LOOP : GL_LINE_STRIP );
		std::for_each( pointset.Begin(), pointset.End(), GLVertexVector );
	glEnd();
}


//*************************************************************************************************

template< typename CoordType >
inline void
GLSetPositionPart( const M4D::Imaging::Geometry::PositionPart< CoordType > & pPart ) 
{
	GLVertexVector( pPart.position );
}

template< typename CoordType >
inline void
GLSetNormalPart( const M4D::Imaging::Geometry::NormalPart< CoordType > & nPart ) 
{
	GLNormalVector( nPart.normal );
}

template< typename CoordType >
inline void
GLSetNormalPart( const M4D::Imaging::Geometry::NormalDummy< CoordType > & nPart ) {}

inline void
GLSetColorPart( const M4D::Imaging::Geometry::ColorPart &cPart) 
{
	GLColorVector( cPart.color );
}

inline void
GLSetColorPart( const M4D::Imaging::Geometry::ColorDummy &) {}

template< typename VertexList >
inline void
GLSetVertices( const VertexList & vertices, const M4D::Imaging::Geometry::TriangleInfoPart &triangle ) 
{
	GLVertexInfoDraw( vertices[ triangle.indices[0] ] );
	GLVertexInfoDraw( vertices[ triangle.indices[1] ] );
	GLVertexInfoDraw( vertices[ triangle.indices[2] ] );
}

template< 
	typename CoordType, 
	template <typename CType> class NormalPartType, 
	typename ColorPartType
	>
inline void
GLVertexInfoDraw( const M4D::Imaging::Geometry::VertexInfo< CoordType, NormalPartType, ColorPartType > &vinfo )
{
	GLSetNormalPart( static_cast< const NormalPartType< CoordType > & >( vinfo ) );
	GLSetColorPart( static_cast< const ColorPartType & >( vinfo ) );
	GLSetPositionPart( static_cast< const M4D::Imaging::Geometry::PositionPart< CoordType > & >( vinfo ) );
}

template<
	typename VertexList,
	typename FaceInfoPartType,
	typename NormalPartType, 
	typename ColorPartType 
	>
inline void
GLFaceInfoDraw( const VertexList & vertices, const M4D::Imaging::Geometry::FaceInfo< FaceInfoPartType, NormalPartType, ColorPartType > &finfo )
{
	GLSetNormalPart( static_cast< const NormalPartType & >( finfo ) );
	GLSetColorPart( static_cast< const ColorPartType & >( finfo ) );

	GLSetVertices( vertices, static_cast< const FaceInfoPartType & >( finfo ) );	
}

template<
	typename VertexType,
	typename FaceType
	>
void
GLDrawMesh( const M4D::Imaging::Geometry::Mesh< VertexType, FaceType > &mesh )
{
	const typename M4D::Imaging::Geometry::Mesh< VertexType, FaceType >::FaceList &faces = mesh.GetFaces();
	const typename M4D::Imaging::Geometry::Mesh< VertexType, FaceType >::VertexList &vertices = mesh.GetVertices();

	for( unsigned i = 0; i < faces.size(); ++i ) {

		GLFaceInfoDraw( vertices, faces[i] );
	}
}


//*************************************************************************************************




} /*namespace M4D*/


#endif /*OGL_DRAWING_H*/

/** @} */
