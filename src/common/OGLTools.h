#ifndef OGL_TOOLS_H
#define OGL_TOOLS_H

#include <GL/gl.h>
#include <GL/glu.h>

#include "common/Types.h"
//GLenum

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
