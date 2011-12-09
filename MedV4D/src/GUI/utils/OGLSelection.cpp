#include "MedV4D/GUI/utils/OGLSelection.h"

namespace M4D
{

boost::filesystem::path gPickingShaderPath = "./data/shaders/PickingShader.cgfx";
	
void
PickManager::initialize( unsigned aPickingRadius )
{
	mCgEffect.Initialize( gPickingShaderPath );
}

void
PickManager::finalize()
{
	mCgEffect.Finalize();
}
	
} //namespace M4D