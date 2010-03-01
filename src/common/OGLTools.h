#ifndef OGL_TOOLS_H
#define OGL_TOOLS_H

#ifdef _MSC_VER
# define WIN32_LEAN_AND_MEAN 1
# include <windows.h>
#endif

#include <GL/gl.h>
#include <GL/glu.h>

#include "common/Types.h"
#include "common/Vector.h"
//GLenum

#define TYPE_FROM_GL_s	int16
#define TYPE_FROM_GL_i	int32
#define TYPE_FROM_GL_f	float32
#define TYPE_FROM_GL_d	float64

#define GL_VERTEX_VECTOR_MACRO( TYPE_SUFFIX, DIM ) \
void GLVertexVector( const Vector< TYPE_FROM_GL_##TYPE_SUFFIX, DIM > &coord )\
{ glVertex##DIM## TYPE_SUFFIX ##v ( coord.GetData() ); }

GL_VERTEX_VECTOR_MACRO( s, 2 )
GL_VERTEX_VECTOR_MACRO( i, 2 )
GL_VERTEX_VECTOR_MACRO( f, 2 )
GL_VERTEX_VECTOR_MACRO( d, 2 )

GL_VERTEX_VECTOR_MACRO( s, 3 )
GL_VERTEX_VECTOR_MACRO( i, 3 )
GL_VERTEX_VECTOR_MACRO( f, 3 )
GL_VERTEX_VECTOR_MACRO( d, 3 )

GL_VERTEX_VECTOR_MACRO( s, 4 )
GL_VERTEX_VECTOR_MACRO( i, 4 )
GL_VERTEX_VECTOR_MACRO( f, 4 )
GL_VERTEX_VECTOR_MACRO( d, 4 )



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

#endif /*OGL_TOOLS_H*/
