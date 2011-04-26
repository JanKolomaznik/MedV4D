#ifndef VOLUME_RENDERER_H
#define VOLUME_RENDERER_H

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


class VolumeRenderer
{
public:
	//typedef std::map< std::wstring, int > ColorTransformNameToIDMap;
	//typedef std::map< int, std::wstring > ColorTransformIDToNameMap;

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
	GLuint					mNoiseMap;

	ColorTransformNameIDList		mAvailableColorTransforms;
};

struct VolumeRenderer::RenderingConfiguration
{
	RenderingConfiguration(): imageData( NULL ), colorTransform( ctMaxIntensityProjection ), transferFunction( NULL ), jitterEnabled( true ), shadingEnabled( true ), sampleCount( 150 )
	{ }
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
