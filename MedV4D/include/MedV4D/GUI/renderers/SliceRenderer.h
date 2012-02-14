#ifndef SLICE_RENDERER_H
#define SLICE_RENDERER_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/GUI/utils/CgShaderTools.h"
#include "MedV4D/GUI/utils/GLTextureImage.h"
#include <boost/bind.hpp>
#include "MedV4D/GUI/utils/TransferFunctionBuffer.h"
#include "MedV4D/GUI/utils/OrthoCamera.h"
#include "MedV4D/GUI/renderers/RendererTools.h"

namespace M4D
{
namespace GUI
{
namespace Renderer
{

extern boost::filesystem::path gSliceRendererShaderPath;

class SliceRenderer
{
public:
	struct RenderingConfiguration;
	
	void
	Initialize();

	void
	Finalize();

	virtual void
	Render( RenderingConfiguration & aConfig, const GLViewSetup &aViewSetup );

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
	RenderingConfiguration(): primaryImageData( NULL ), secondaryImageData( NULL ), plane( XY_PLANE ), currentSlice( 0 ), colorTransform( ctLUTWindow ), transferFunction( NULL ), enableInterpolation( true )
	{}

	const GLTextureImage3D			*primaryImageData;
	const GLTextureImage3D			*secondaryImageData;
	CartesianPlanes				plane;
	Vector3i				currentSlice;
	
	OrthoCamera				camera;
	Vector3f sliceCenter;
	Vector3f sliceNormal;

	Vector3f
	getCurrentRealSlice()const
	{
		if ( primaryImageData ) {
			return primaryImageData->GetMinimum() + VectorMemberProduct( currentSlice, primaryImageData->GetElementExtents() );
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
