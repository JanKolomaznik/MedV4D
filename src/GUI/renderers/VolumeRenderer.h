#ifndef VOLUME_RENDERER_H
#define VOLUME_RENDERER_H

#include "common/Common.h"
#include "GUI/utils/CgShaderTools.h"
#include "GUI/utils/GLTextureImage.h"
#include <boost/bind.hpp>
#include "GUI/utils/TransferFunctionBuffer.h"

namespace M4D
{
namespace GUI
{
namespace Renderer
{


class VolumeRenderer
{
public:
	struct RenderingConfiguration;

	void
	Initialize();

	void
	Finalize();
	
	virtual void
	Render( RenderingConfiguration & aConfig, bool aSetupView = true );
protected:
	enum ColorTransform
	{
		ctLUTWindow,
		ctTransferFunction1D,
		ctMaxIntensityProjection,
		ctSimpleColorMap
	};

	CGcontext   				mCgContext;
	CgEffect				mCgEffect;
	GLuint					mNoiseMap;
};

struct VolumeRenderer::RenderingConfiguration
{
	const GLTextureImage3D			*imageData;
	
	int					colorTransform;
	const GLTransferFunctionBuffer1D	*transferFunction;
	Vector2f				lutWindow;
	Camera					camera;
	bool					jitterEnabled;
	bool					shadingEnabled;
	size_t					sampleCount;				
};


}//Renderer
}//GUI
}//M4D

#endif /*VOLUME_RENDERER_H*/
