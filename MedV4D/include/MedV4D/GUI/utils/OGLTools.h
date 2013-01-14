#ifndef OGL_TOOLS_H
#define OGL_TOOLS_H

//Temporary workaround
#ifndef Q_MOC_RUN 
#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/GeometricPrimitives.h"
#include <boost/shared_array.hpp>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#endif

#ifdef _MSC_VER
# define WIN32_LEAN_AND_MEAN 1
# include <windows.h>

# ifdef GetMessage
#   undef GetMessage
# endif

# ifdef SendMessage
#   undef SendMessage
# endif

# ifdef RGB
#   undef RGB
# endif

#endif

#include <GL/glew.h>
/*#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>
*/
//#include <QtOpenGL>

#include <QGLContext>
#include <QColor>


#ifdef DEBUG_LEVEL
#define GL_CHECKED_CALL( ... ) { __VA_ARGS__ ; M4D::CheckForGLError( TO_STRING( __FILE__ " on " << __LINE__) ); }
#else
#define GL_CHECKED_CALL( ... ) { __VA_ARGS__ ; }
#endif /*DEBUG_LEVEL*/


#define GL_ERROR_CLEAR_AFTER_CALL( ... ) { __VA_ARGS__ ; glGetError(); }

namespace M4D
{

	
class GLException: public M4D::ErrorHandling::ExceptionBase
{
public:
	GLException( std::string name ) throw() : ExceptionBase( name ) {}
	~GLException() throw(){}
};

struct GLViewSetup
{
	GLViewSetup(): model(1.0), view(1.0), modelView(1.0), modelViewProj(1.0), projection(1.0)
	{}
	
	glm::dmat4x4 model;
	glm::dmat4x4 view;
	glm::dmat4x4 modelView;
	glm::dmat4x4 modelViewProj;
	glm::dmat4x4 projection;
	glm::ivec4  viewport;
	//GLdouble model[16];
	//GLdouble proj[16];
	//GLint view[4];
};


template< typename TType >
std::ostream &
operator<<( std::ostream &stream, const glm::detail::tmat4x4< TType > &matrix )
{
	for (size_t j = 0 ; j < 4; ++j) {
		for (size_t i = 0 ; i < 4; ++i) {
			stream << matrix[i][j] << "\t";
		}
		stream << std::endl;
	}
	return stream;
}

std::ostream &
operator<<( std::ostream & stream, const GLViewSetup &setup );

void 
getCurrentGLSetup( GLViewSetup &aSetup );

Vector3d
getPointFromScreenCoordinates( Vector2f aScreenCoords, const GLViewSetup &aViewSetup, double aZValue = 0.0 );

Vector3f
getDirectionFromScreenCoordinatesAndCameraPosition( Vector2f aScreenCoords, const GLViewSetup &aViewSetup, const Vector3f aCameraPos );


void 
CheckForGLError( const std::string &situation );


inline bool
isGLContextActive()
{
	return QGLContext::currentContext() != NULL;
}

void
getImageBufferFromTexture( uint32 &aWidth, uint32 &aHeight, boost::shared_array< uint8 > &aBuffer, GLuint aTexture );

#ifdef USE_DEVIL

class DevILException: public M4D::ErrorHandling::ExceptionBase
{
public:
	DevILException( std::string name ) throw() : ExceptionBase( name ) {}
	~DevILException() throw(){}
};

void 
CheckForDevILError( const std::string &situation );

#define DEVIL_CHECKED_CALL( ... ) { __VA_ARGS__ ; CheckForDevILError( TO_STRING( __FILE__ " on " << __LINE__) ); }

void
SaveTextureToImageFile( uint32 aWidth, uint32 aHeight, GLuint aTexture, std::string aPath, bool aOverwrite = false );
#endif /*USE_DEVIL*/


void
InitOpenGL();

struct GLPushAtribs
{
	GLPushAtribs(GLbitfield attribs = GL_ALL_ATTRIB_BITS )
	{
		GL_CHECKED_CALL( glPushAttrib( attribs ) );
	}
	~GLPushAtribs()
	{
		GL_CHECKED_CALL( glPopAttrib() );
	}
};

struct GLPushMatrices
{
	GLPushMatrices()
	{
		GL_CHECKED_CALL( glMatrixMode(GL_PROJECTION) );
		GL_CHECKED_CALL( glPushMatrix() );
		GL_CHECKED_CALL( glMatrixMode(GL_MODELVIEW) );
		GL_CHECKED_CALL( glPushMatrix() );
	}
	~GLPushMatrices()
	{
		GL_CHECKED_CALL( glMatrixMode(GL_MODELVIEW) );
		GL_CHECKED_CALL( glPopMatrix() );
		GL_CHECKED_CALL( glMatrixMode(GL_PROJECTION) );
		GL_CHECKED_CALL( glPopMatrix() );
	}
};


//*******************************************************************************************************************
//*******************************************************************************************************************
//GLenum

#define TYPE_FROM_GL_b	int8
#define TYPE_FROM_GL_ub	uint8
#define TYPE_FROM_GL_s	int16
#define TYPE_FROM_GL_us	uint16
#define TYPE_FROM_GL_i	int32
#define TYPE_FROM_GL_ui	uint32
#define TYPE_FROM_GL_f	float32
#define TYPE_FROM_GL_d	float64

#define GL_FUNCTION_VECTOR_DEFINITION_MACRO( FUNC_NAME, GL_FUNC, TYPE_SUFFIX, DIM ) \
inline void FUNC_NAME( const Vector< TYPE_FROM_GL_##TYPE_SUFFIX, DIM > &coord )\
{ GL_FUNC##DIM## TYPE_SUFFIX ##v ( coord.GetData() ); }

#define GL_VERTEX_VECTOR_DEFINITION_MACRO( TYPE_SUFFIX, DIM ) \
	GL_FUNCTION_VECTOR_DEFINITION_MACRO( GLVertexVector, glVertex, TYPE_SUFFIX, DIM );

#define GL_NORMAL_VECTOR_DEFINITION_MACRO( TYPE_SUFFIX, DIM ) \
	GL_FUNCTION_VECTOR_DEFINITION_MACRO( GLNormalVector, glNormal, TYPE_SUFFIX, DIM );

#define GL_COLOR_VECTOR_DEFINITION_MACRO( TYPE_SUFFIX, DIM ) \
	GL_FUNCTION_VECTOR_DEFINITION_MACRO( GLColorVector, glColor, TYPE_SUFFIX, DIM );

#define GL_TEXTURE_VECTOR_DEFINITION_MACRO( TYPE_SUFFIX, DIM ) \
	GL_FUNCTION_VECTOR_DEFINITION_MACRO( GLTextureVector, glTexCoord, TYPE_SUFFIX, DIM );

GL_VERTEX_VECTOR_DEFINITION_MACRO( s, 2 )
GL_VERTEX_VECTOR_DEFINITION_MACRO( i, 2 )
GL_VERTEX_VECTOR_DEFINITION_MACRO( f, 2 )
GL_VERTEX_VECTOR_DEFINITION_MACRO( d, 2 )

GL_VERTEX_VECTOR_DEFINITION_MACRO( s, 3 )
GL_VERTEX_VECTOR_DEFINITION_MACRO( i, 3 )
GL_VERTEX_VECTOR_DEFINITION_MACRO( f, 3 )
GL_VERTEX_VECTOR_DEFINITION_MACRO( d, 3 )

GL_VERTEX_VECTOR_DEFINITION_MACRO( s, 4 )
GL_VERTEX_VECTOR_DEFINITION_MACRO( i, 4 )
GL_VERTEX_VECTOR_DEFINITION_MACRO( f, 4 )
GL_VERTEX_VECTOR_DEFINITION_MACRO( d, 4 )

//***************************************
GL_NORMAL_VECTOR_DEFINITION_MACRO( b, 3 )
GL_NORMAL_VECTOR_DEFINITION_MACRO( s, 3 )
GL_NORMAL_VECTOR_DEFINITION_MACRO( i, 3 )
GL_NORMAL_VECTOR_DEFINITION_MACRO( f, 3 )
GL_NORMAL_VECTOR_DEFINITION_MACRO( d, 3 )
//***************************************
GL_COLOR_VECTOR_DEFINITION_MACRO(  b, 3 )
GL_COLOR_VECTOR_DEFINITION_MACRO( ub, 3 )
GL_COLOR_VECTOR_DEFINITION_MACRO(  s, 3 )
GL_COLOR_VECTOR_DEFINITION_MACRO( us, 3 )
GL_COLOR_VECTOR_DEFINITION_MACRO(  i, 3 )
GL_COLOR_VECTOR_DEFINITION_MACRO( ui, 3 )
GL_COLOR_VECTOR_DEFINITION_MACRO(  f, 3 )
GL_COLOR_VECTOR_DEFINITION_MACRO(  d, 3 )

GL_COLOR_VECTOR_DEFINITION_MACRO(  b, 4 )
GL_COLOR_VECTOR_DEFINITION_MACRO( ub, 4 )
GL_COLOR_VECTOR_DEFINITION_MACRO(  s, 4 )
GL_COLOR_VECTOR_DEFINITION_MACRO( us, 4 )
GL_COLOR_VECTOR_DEFINITION_MACRO(  i, 4 )
GL_COLOR_VECTOR_DEFINITION_MACRO( ui, 4 )
GL_COLOR_VECTOR_DEFINITION_MACRO(  f, 4 )
GL_COLOR_VECTOR_DEFINITION_MACRO(  d, 4 )
//***************************************
GL_TEXTURE_VECTOR_DEFINITION_MACRO( s, 1 )
GL_TEXTURE_VECTOR_DEFINITION_MACRO( i, 1 )
GL_TEXTURE_VECTOR_DEFINITION_MACRO( f, 1 )
GL_TEXTURE_VECTOR_DEFINITION_MACRO( d, 1 )

GL_TEXTURE_VECTOR_DEFINITION_MACRO( s, 2 )
GL_TEXTURE_VECTOR_DEFINITION_MACRO( i, 2 )
GL_TEXTURE_VECTOR_DEFINITION_MACRO( f, 2 )
GL_TEXTURE_VECTOR_DEFINITION_MACRO( d, 2 )

GL_TEXTURE_VECTOR_DEFINITION_MACRO( s, 3 )
GL_TEXTURE_VECTOR_DEFINITION_MACRO( i, 3 )
GL_TEXTURE_VECTOR_DEFINITION_MACRO( f, 3 )
GL_TEXTURE_VECTOR_DEFINITION_MACRO( d, 3 )

GL_TEXTURE_VECTOR_DEFINITION_MACRO( s, 4 )
GL_TEXTURE_VECTOR_DEFINITION_MACRO( i, 4 )
GL_TEXTURE_VECTOR_DEFINITION_MACRO( f, 4 )
GL_TEXTURE_VECTOR_DEFINITION_MACRO( d, 4 )
//***************************************


inline void
GLColorFromQColor( const QColor &color )
{
	glColor4f( color.redF(), color.greenF(), color.blueF(), color.alphaF() );
}

template< typename T >
struct M4DToGLType
{
	static const GLenum GLTypeID = 0;
};

template<>
struct M4DToGLType< uint8 >
{
	static const GLenum GLTypeID = GL_UNSIGNED_BYTE;
};

template<>
struct M4DToGLType< int8 >
{
	static const GLenum GLTypeID = GL_BYTE;
};

template<>
struct M4DToGLType< uint16 >
{
	static const GLenum GLTypeID = GL_UNSIGNED_SHORT;
};

template<>
struct M4DToGLType< int16 >
{
	static const GLenum GLTypeID = GL_SHORT;
};

template<>
struct M4DToGLType< uint32 >
{
	static const GLenum GLTypeID = GL_UNSIGNED_INT;
};

template<>
struct M4DToGLType< int32 >
{
	static const GLenum GLTypeID = GL_INT;
};

template<>
struct M4DToGLType< float32 >
{
	static const GLenum GLTypeID = GL_FLOAT;
};
//**************************************************************************
template< typename T >
struct M4DToGLTextureInternal
{
	static const GLint GLInternal = 0;
};

template<>
struct M4DToGLTextureInternal< uint8 >
{
	static const GLint GLInternal = GL_LUMINANCE;
};

template<>
struct M4DToGLTextureInternal< int8 >
{
	static const GLint GLInternal = GL_LUMINANCE;
};

template<>
struct M4DToGLTextureInternal< uint16 >
{
	static const GLint GLInternal = GL_R16F;
};

template<>
struct M4DToGLTextureInternal< int16 >
{
	static const GLint GLInternal = GL_R16F;
};

template<>
struct M4DToGLTextureInternal< uint32 >
{
	static const GLint GLInternal = GL_R32F;
};

template<>
struct M4DToGLTextureInternal< int32 >
{
	static const GLint GLInternal = GL_R32F;
};

template<>
struct M4DToGLTextureInternal< int64 >
{
	static const GLint GLInternal = GL_R32F;
};

template<>
struct M4DToGLTextureInternal< uint64 >
{
	static const GLint GLInternal = GL_R32F;
};

template<>
struct M4DToGLTextureInternal< float32 >
{
	static const GLint GLInternal = GL_R32F;
};

template<>
struct M4DToGLTextureInternal< float64 >
{
	static const GLint GLInternal = GL_R32F;
};


} /*namespace M4D*/

#endif /*OGL_TOOLS_H*/


