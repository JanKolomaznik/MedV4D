#ifndef SLICE_RENDERER_H
#define SLICE_RENDERER_H

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


class SliceRenderer
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
};

struct SliceRenderer::RenderingConfiguration
{
	const GLTextureImage3D			*imageData;
	CartesianPlanes				plane;
	Vector3i				currentSlice;

	int					colorTransform;
	const GLTransferFunctionBuffer1D	*transferFunction;
	Vector2f				lutWindow;
	ViewConfiguration2D			viewConfig;
	bool					enableInterpolation;
};


}//Renderer
}//GUI
}//M4D

#endif /*SLICE_RENDERER_H*/
