#ifndef ID_MAPPING_BUFFER_H
#define ID_MAPPING_BUFFER_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/DefinitionMacros.h"
//#include "MedV4D/GUI/utils/OGLTools.h"
#include <soglu/OGLTools.hpp>

#include <vector>
#include "MedV4D/GUI/managers/OpenGLManager.h"

namespace M4D
{
namespace GUI
{
	
class IDMappingBuffer
{
public:
	SMART_POINTER_TYPEDEFS( IDMappingBuffer );

	typedef RGBAf 				ValueType;
	typedef std::vector< ValueType >		InternalBuffer;
	typedef InternalBuffer::iterator		Iterator;
	typedef InternalBuffer::const_iterator	ConstIterator;

	typedef ValueType 		value_type;
	typedef Iterator 		iterator;
	typedef ConstIterator 		const_iterator;

	IDMappingBuffer( size_t aSize = 0 ): mBuffer( aSize )
	{ /*empty*/ }

	
	Iterator
	begin()
	{ return mBuffer.begin(); }

	Iterator
	end()
	{ return mBuffer.end(); }

	ConstIterator
	begin()const
	{ return mBuffer.begin(); }

	ConstIterator
	end()const
	{ return mBuffer.end(); }

	size_t
	size()const
	{ return mBuffer.size(); }

	void
	resize( size_t aSize )
	{ mBuffer.resize( aSize ); }

	ValueType &
	operator[]( size_t aIdx )
	{
		ASSERT( aIdx < mSize );
		return mBuffer[ aIdx ];
	}

	ValueType
	operator[]( size_t aIdx ) const
	{
		ASSERT( aIdx < mSize );
		return mBuffer[ aIdx ];
	}
	const ValueType *
	data()const
	{
		return &mBuffer.front();
	}
protected:
	std::vector< RGBAf > mBuffer;
	size_t	mSize;
private:

};

struct GLIDMappingBuffer
{
	SMART_POINTER_TYPEDEFS( GLIDMappingBuffer );
	
	~GLIDMappingBuffer()
	{
		OpenGLManager::getInstance()->deleteTextures( mGLTextureID );
	}
	friend GLIDMappingBuffer::Ptr createGLMappingBuffer( const IDMappingBuffer &aBuffer );

	GLuint
	getTextureID()const
	{ return mGLTextureID; }
	
	size_t
	size()const
	{ return mSize; }
private:
	GLIDMappingBuffer( GLuint aTexture, size_t aSize ): mGLTextureID( aTexture ), mSize( aSize )
	{}

	GLuint	mGLTextureID;
	size_t mSize;
};

inline GLIDMappingBuffer::Ptr
createGLMappingBuffer( const IDMappingBuffer &aBuffer )
{
	if ( aBuffer.size() == 0 ) {
		_THROW_ ErrorHandling::EBadParameter( "IDMappingBuffer of 0 size" );
	}

	GLuint texName;

	try {
		GL_CHECKED_CALL( glPixelStorei( GL_UNPACK_ALIGNMENT, 1 ) );
		GL_CHECKED_CALL( glPixelStorei(GL_PACK_ALIGNMENT, 1) );
		GL_CHECKED_CALL( glGenTextures( 1, &texName ) );
		GL_CHECKED_CALL( glBindTexture ( GL_TEXTURE_1D, texName ) );
		GL_CHECKED_CALL( glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE ) );

		GL_CHECKED_CALL( glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE ) );
		GL_CHECKED_CALL( glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ) );
		GL_CHECKED_CALL( glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ) );


		GL_CHECKED_CALL( glEnable( GL_TEXTURE_1D ) );
		
		GL_CHECKED_CALL( glBindTexture( GL_TEXTURE_1D, texName ) );

		GL_CHECKED_CALL( 
			glTexImage1D(
				GL_TEXTURE_1D, 
				0, 
				GL_RGBA32F, 
				static_cast<GLsizei>(aBuffer.size()), 
				0, 
				GL_RGBA, 
				GL_FLOAT, 
				aBuffer.data()
				)
			);

		
		soglu::checkForGLError( "OGL building texture for transfer function: " );
	} 
	catch( ... ) {
		if( texName != 0 ) {
			glDeleteTextures( 1, &texName );
		}
		throw;
	}

	return GLIDMappingBuffer::Ptr( new GLIDMappingBuffer( texName, aBuffer.size() ) );
}

struct IDMappingBufferInfo
{
	IDMappingBufferInfo():id(0)
	{ }

	M4D::Common::IDNumber id;
	M4D::GUI::GLIDMappingBuffer::Ptr glBuffer;
	M4D::GUI::IDMappingBuffer::Ptr buffer;
};


} /*namespace GUI*/
} /*namespace M4D*/


#endif //ID_MAPPING_BUFFER_H
