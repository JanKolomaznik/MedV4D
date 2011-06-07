#ifndef OGL_TOOLS_H
#define OGL_TOOLS_H

#include "common/Common.h"

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
#include <QtOpenGL>

namespace M4D
{

class GLException: public M4D::ErrorHandling::ExceptionBase
{
public:
	GLException( std::string name ) throw() : ExceptionBase( name ) {}
	~GLException() throw(){}
};



void 
CheckForGLError( const std::string &situation );

#define GL_CHECKED_CALL( ... ) { __VA_ARGS__ ; M4D::CheckForGLError( TO_STRING( __FILE__ " on " << __LINE__) ); }

inline bool
IsGLContextActive()
{
	return QGLContext::currentContext() != NULL;
}


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


