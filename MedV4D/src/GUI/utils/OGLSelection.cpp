#include "MedV4D/GUI/utils/OGLSelection.h"

namespace M4D
{

const GLint SELECTION_BUFFER_COLOR_DEPTH = GL_RGBA16;
	
boost::filesystem::path gPickingShaderPath = "./data/shaders/PickingShader.cgfx";
	
void
PickManager::initialize( unsigned aPickingRadius )
{
	mPickingRadius = aPickingRadius;
	mCgEffect.Initialize( gPickingShaderPath );
	mFrameBuffer.Initialize( 2*mPickingRadius, 2*mPickingRadius, SELECTION_BUFFER_COLOR_DEPTH );
}

void
PickManager::finalize()
{
	mCgEffect.Finalize();
}
	
} //namespace M4D