#include "MedV4D/GUI/utils/OGLSelection.h"

namespace M4D
{

boost::filesystem::path gPickingShaderPath = "./data/shaders/PickingShader.cgfx";
	
void
PickManager::initialize( unsigned aPickingRadius )
{
	mPickingRadius = aPickingRadius;
	mCgEffect.Initialize( gPickingShaderPath );
	mFrameBuffer.Initialize( 2*mPickingRadius, 2*mPickingRadius );
}

void
PickManager::finalize()
{
	mCgEffect.Finalize();
}
	
} //namespace M4D