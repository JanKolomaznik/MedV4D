#ifndef SLICE_RENDERER_H
#define SLICE_RENDERER_H

#include "common/Common.h"
#include "GUI/utils/CgShaderTools.h"
#include "GUI/utils/GLTextureImage.h"
#include <boost/bind.hpp>
#include "GUI/utils/TransferFunctionBuffer.h"
#include "GUI/renderers/RendererTools.h"

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

	const ColorTransformNameIDList&
	GetAvailableColorTransforms()const
	{
		return mAvailableColorTransforms;
	}
protected:
	CGcontext   				mCgContext;
	CgEffect				mCgEffect;

	ColorTransformNameIDList		mAvailableColorTransforms;
};

struct SliceRenderer::RenderingConfiguration
{
	RenderingConfiguration(): imageData( false ), plane( XY_PLANE ), currentSlice( 0 ), colorTransform( ctLUTWindow ), transferFunction( NULL ), enableInterpolation( true )
	{}

	const GLTextureImage3D			*imageData;
	CartesianPlanes				plane;
	Vector3i				currentSlice;

	Vector3f
	getCurrentRealSlice()
	{
		if ( imageData ) {
			return imageData->GetMinimum() + VectorMemberProduct( currentSlice, imageData->GetElementExtents() );
		} else {
			return Vector3f();
		}
	}

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
