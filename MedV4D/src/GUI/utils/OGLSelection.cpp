#include "MedV4D/GUI/utils/OGLSelection.h"

namespace M4D
{

const GLint SELECTION_BUFFER_COLOR_DEPTH = GL_RGBA16;
	
boost::filesystem::path gPickingShaderPath = "./data/shaders/PickingShader.cgfx";
	
void
PickManager::initialize( unsigned aPickingRadius )
{
	ASSERT(IsGLContextActive());
	mPickingRadius = aPickingRadius;
	mCgEffect.Initialize( gPickingShaderPath );
	mFrameBuffer.Initialize( 2*mPickingRadius, 2*mPickingRadius, SELECTION_BUFFER_COLOR_DEPTH );
	
	mBuffer = BufferArray( new uint16[4*2*mPickingRadius*2*mPickingRadius] );
}

void
PickManager::finalize()
{
	ASSERT(IsGLContextActive());
	mCgEffect.Finalize();
}
	
void
PickManager::getIDs( SelectedIDsSet &aIDs )
{
	for( size_t i = 0; i < 2*mPickingRadius*2*mPickingRadius; ++i ) {
		uint16 r, g, b, a;
		r = mBuffer[3*i];
		/*g = mBuffer[3*i+1];
		b = mBuffer[3*i+2];
		a = mBuffer[3*i+3];*/

		if( r > 0 ) {
			aIDs.insert( r );
		}
	}
}
	
void
PickManager::getBufferFromGPU()
{
	ASSERT(IsGLContextActive());
	ASSERT( mBuffer );
	GL_CHECKED_CALL( glBindTexture( GL_TEXTURE_2D, mFrameBuffer.GetColorBuffer() ) );
	GL_CHECKED_CALL( glGetTexImage(	
				GL_TEXTURE_2D, 
				0, 
				SELECTION_BUFFER_COLOR_DEPTH, 
				GL_UNSIGNED_SHORT, 
				(void*)mBuffer.get()
				) );
	GL_CHECKED_CALL( glBindTexture( GL_TEXTURE_2D, 0 ) );
}
	
} //namespace M4D
