#ifndef FRAME_BUFFER_OBJECT_H
#define FRAME_BUFFER_OBJECT_H

#include "MedV4D/Common/Common.h"
#include "GUI/utils/OGLDrawing.h"

namespace M4D {

class FrameBufferObject
{
public:
	FrameBufferObject(): mInitialized( false ), mBinded( false )
	{}

	~FrameBufferObject()
	{
		if ( mInitialized ) {
			GL_CHECKED_CALL( glDeleteFramebuffersEXT( 1, &mFrameBufferObject ) );
			GL_CHECKED_CALL( glDeleteTextures( 1, &mColorTexture ) );
			GL_CHECKED_CALL( glDeleteRenderbuffersEXT( 1, &mDepthBuffer ) );
		}
	}

	GLuint
	GetColorBuffer()
	{
		ASSERT( mInitialized );
		return mColorTexture;
	}

	void
	Initialize( unsigned aWidth, unsigned aHeight )
	{
		GL_CHECKED_CALL( glGenFramebuffersEXT( 1, &mFrameBufferObject ) );
		GL_CHECKED_CALL( glGenRenderbuffersEXT( 1, &mDepthBuffer ) );
		GL_CHECKED_CALL( glGenTextures( 1, &mColorTexture ) );

		mInitialized = true;
		Resize( aWidth, aHeight );
	}

	void
	Finalize()
	{
		if ( mInitialized ) {
			GL_CHECKED_CALL( glDeleteFramebuffersEXT( 1, &mFrameBufferObject ) );
			GL_CHECKED_CALL( glDeleteTextures( 1, &mColorTexture ) );
			GL_CHECKED_CALL( glDeleteRenderbuffersEXT( 1, &mDepthBuffer ) );
		}
		mInitialized = false;
	}

	void
	Render()
	{
		GL_CHECKED_CALL( glMatrixMode( GL_PROJECTION ) );
		GL_CHECKED_CALL( glLoadIdentity() );
		GL_CHECKED_CALL( glMatrixMode( GL_MODELVIEW ) );
		GL_CHECKED_CALL( glLoadIdentity() );
		GL_CHECKED_CALL( glBindTexture( GL_TEXTURE_2D, mColorTexture ) );
		GL_CHECKED_CALL( glDisable( GL_TEXTURE_1D ) );
		GL_CHECKED_CALL( glDisable( GL_TEXTURE_3D ) );
		GL_CHECKED_CALL( glEnable( GL_TEXTURE_2D ) );

		GL_CHECKED_CALL( gluOrtho2D( 0, (float32)mSize[0], 0, (float32)mSize[1] ) );

		GLDrawTexturedQuad(
			Vector< float32, 2 >( 0.0f, 0.0f ), 
			Vector< float32, 2 >( (float32)mSize[0], (float32)mSize[1] )
			);
		GL_CHECKED_CALL( glBindTexture( GL_TEXTURE_2D, 0 ) );
	}

	void
	Bind()
	{
		ASSERT( mInitialized );
		GL_CHECKED_CALL( glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, mFrameBufferObject ) );
		mBinded = true;
	}

	void
	Unbind()
	{
		ASSERT( mInitialized );
		GL_CHECKED_CALL( glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 ) );
		mBinded = false;
	}

	void
	Resize( unsigned aWidth, unsigned aHeight )
	{
		ASSERT ( mInitialized );
		GL_CHECKED_CALL( glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, mFrameBufferObject ) );

		GL_CHECKED_CALL( glBindTexture ( GL_TEXTURE_2D, mColorTexture ) );
		GL_CHECKED_CALL( glBindRenderbufferEXT( GL_RENDERBUFFER_EXT, mDepthBuffer ) );

		GL_CHECKED_CALL( glRenderbufferStorageEXT( GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT, aWidth, aHeight ) );
		GL_CHECKED_CALL( glBindRenderbufferEXT( GL_RENDERBUFFER_EXT, 0 ) );

		glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
		GL_CHECKED_CALL( glTexImage2D(
					GL_TEXTURE_2D, 
					0, 
					GL_RGBA32F, 
					aWidth, 
					aHeight, 
					0, 
					GL_RGBA, 
					GL_FLOAT, 
					NULL
					) );
		GL_CHECKED_CALL( glBindTexture ( GL_TEXTURE_2D, 0 ) );
		
		GL_CHECKED_CALL( glFramebufferRenderbufferEXT( 
					GL_FRAMEBUFFER_EXT,
					GL_DEPTH_ATTACHMENT_EXT,
					GL_RENDERBUFFER_EXT,
					mDepthBuffer
					) );

		GL_CHECKED_CALL( glFramebufferTexture2DEXT( 
					GL_FRAMEBUFFER_EXT,
					GL_COLOR_ATTACHMENT0_EXT,
					GL_TEXTURE_2D,
					mColorTexture,
					0 
					) );

		GL_CHECKED_CALL( glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 ) );

		mSize[ 0 ] = aWidth;
		mSize[ 1 ] = aHeight;
	}

	Vector2u
	GetSize()const
	{ return mSize; }

protected:
	GLuint	mFrameBufferObject, 
		mDepthBuffer, 
		mColorTexture;
	bool mInitialized;
	bool mBinded;
	Vector2u mSize;
};

} /*namespace M4D*/

#endif /*FRAME_BUFFER_OBJECT_H*/
