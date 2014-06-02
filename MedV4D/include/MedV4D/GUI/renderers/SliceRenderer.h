#pragma once

#include "MedV4D/Common/Common.h"

#include <soglu/Camera.hpp>
#include <soglu/GLTextureImage.hpp>
#include <vorgl/TransferFunctionBuffer.hpp>

//#include "MedV4D/GUI/utils/CgShaderTools.h"
//#include "MedV4D/GUI/utils/GLTextureImage.h"
#include <boost/bind.hpp>
//#include "MedV4D/GUI/utils/TransferFunctionBuffer.h"
//#include "MedV4D/GUI/utils/OrthoCamera.h"

#include "MedV4D/GUI/renderers/RendererTools.h"

#include <soglu/GLTextureImage.hpp>
#include <soglu/ViewConfiguration.hpp>
#include <vorgl/TransferFunctionBuffer.hpp>
#include <vorgl/SliceRenderer.hpp>

namespace M4D
{
namespace GUI
{
namespace Renderer
{

extern boost::filesystem::path gSliceRendererShaderPath;

class SliceRenderer: public vorgl::SliceRenderer
{
public:
	struct RenderingConfiguration;

	void
	initialize();

	void
	reloadShaders()
	{
		vorgl::SliceRenderer::loadShaders(gSliceRendererShaderPath);
	}

	void
	finalize();

	virtual void
	render( RenderingConfiguration & aConfig, const soglu::GLViewSetup &aViewSetup );

	const ColorTransformNameIDList&
	GetAvailableColorTransforms()const
	{
		return mAvailableColorTransforms;
	}
protected:


	ColorTransformNameIDList		mAvailableColorTransforms;
};

struct SliceRenderer::RenderingConfiguration
{
	RenderingConfiguration():
		//primaryImageData( NULL ),
		//secondaryImageData( NULL ),
		plane( XY_PLANE ),
		currentSlice( 0 ),
		colorTransform( ctLUTWindow ),
		enableInterpolation( true ),
		multiDatasetRenderingStyle( mdrsOnlyPrimary )
	{}

	soglu::GLTextureImage3D::WPtr			primaryImageData;
	soglu::GLTextureImage3D::WPtr			secondaryImageData;
	CartesianPlanes				plane;
	glm::ivec3				currentSlice;

	soglu::OrthoCamera				camera;
	glm::fvec3 sliceCenter;
	glm::fvec3 sliceNormal;

	glm::fvec3
	getCurrentRealSlice()const
	{
		soglu::GLTextureImageTyped<3>::Ptr primaryData = primaryImageData.lock();

		if ( primaryData ) {
			return primaryData->getExtents().realMinimum + /*VectorMemberProduct*/( glm::fvec3(currentSlice) * primaryData->getExtents().elementExtents );
		} else {
			return glm::fvec3();
		}
	}

	int					colorTransform;
	vorgl::GLTransferFunctionBuffer1D::ConstWPtr	transferFunction;
	glm::fvec2					lutWindow;
	soglu::ViewConfiguration2D			viewConfig;
	bool					enableInterpolation;

	MultiDatasetRenderingStyle		multiDatasetRenderingStyle;
};


}//Renderer
}//GUI
}//M4D
