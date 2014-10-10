#include "MedV4D/GUI/renderers/SliceRenderer.h"

namespace M4D
{
namespace GUI
{
namespace Renderer
{

boost::filesystem::path gSliceRendererShaderPath;

void
SliceRenderer::initialize()
{
	vorgl::SliceRenderer::initialize(gSliceRendererShaderPath);

	mAvailableColorTransforms.clear();
	//mAvailableColorTransforms.push_back( WideNameIdPair( L"LUT window", ctLUTWindow ) );
	//mAvailableColorTransforms.push_back( WideNameIdPair( L"Transfer function", ctTransferFunction1D ) );
	//mAvailableColorTransforms.push_back( WideNameIdPair( L"Region colormap", ctSimpleColorMap ) );
	mAvailableColorTransforms.push_back( ColorTransformNameIDList::value_type( "LUT window", ctLUTWindow ) );
	mAvailableColorTransforms.push_back( ColorTransformNameIDList::value_type( "Transfer function", ctTransferFunction1D ) );
	mAvailableColorTransforms.push_back( ColorTransformNameIDList::value_type( "Region colormap", ctSimpleColorMap ) );
}

void
SliceRenderer::finalize()
{
	vorgl::SliceRenderer::finalize();
	//TODO
}


void
SliceRenderer::render( SliceRenderer::RenderingConfiguration & aConfig, const soglu::GLViewSetup &aViewSetup )
{
	soglu::GLTextureImageTyped<3>::Ptr primaryData = aConfig.primaryImageData.lock();
	if ( !primaryData ) {
		_THROW_ ErrorHandling::EObjectUnavailable( "Primary texture not available" );
	}

	std::string techniqueName;
	switch ( aConfig.colorTransform ) {
	case ctLUTWindow:
		{
			vorgl::SliceConfiguration slice = {
					float32(aConfig.currentSlice[ aConfig.plane ]+0.5f) * primaryData->getExtents().elementExtents[aConfig.plane],
					(soglu::CartesianPlanes)aConfig.plane };
			vorgl::SliceRenderingQuality renderingQuality = { aConfig.enableInterpolation };
			vorgl::BrightnessContrastRenderingOptions bcOptions = { aConfig.lutWindow };

			brightnessContrastRendering(
				aViewSetup,
				*primaryData,
				slice,
				renderingQuality,
				bcOptions
				);

			/*lutWindowRendering(
				*primaryData,
				float32(aConfig.currentSlice[ aConfig.plane ]+0.5f) * primaryData->getExtents().elementExtents[aConfig.plane],
				(soglu::CartesianPlanes)aConfig.plane,
				aConfig.lutWindow,
				aConfig.enableInterpolation,
				aViewSetup
				);*/
		}
		break;
	case ctTransferFunction1D:
		{
			vorgl::GLTransferFunctionBuffer1D::ConstPtr transferFunction = aConfig.transferFunction.lock();
			if ( !transferFunction ) {
				_THROW_ M4D::ErrorHandling::EObjectUnavailable( "Transfer function no available" );
			}
			transferFunctionRendering(
				*primaryData,
				float32(aConfig.currentSlice[ aConfig.plane ]+0.5f) * primaryData->getExtents().elementExtents[aConfig.plane],
				(soglu::CartesianPlanes)aConfig.plane,
				*transferFunction,
				aConfig.enableInterpolation,
				aViewSetup
				);
		}
		break;
	case ctSimpleColorMap:
		//techniqueName = "SimpleRegionColorMap_3D";
		ASSERT( false );
		break;
	case ctMaxIntensityProjection:
		ASSERT( false );
		break;
	default:
		ASSERT( false );
	}

	soglu::GLTextureImageTyped<3>::Ptr secondaryData = aConfig.secondaryImageData.lock();

	if( secondaryData ) {
		overlayMaskRendering(
				*primaryData,
				float32(aConfig.currentSlice[ aConfig.plane ]+0.5f) * primaryData->getExtents().elementExtents[aConfig.plane],
				(soglu::CartesianPlanes)aConfig.plane,
				0.5f, //TODO - transparency
				aConfig.enableInterpolation,
				aViewSetup
				);
		/*mCgEffect.executeTechniquePass(
			"OverlayMask_3D",
			boost::bind( &vorgl::GLDrawVolumeSlice3D,
				secondaryData->getExtents().realMinimum,
				secondaryData->getExtents().realMaximum,
				float32(aConfig.currentSlice[ aConfig.plane ]+0.5f) * secondaryData->getExtents().elementExtents[aConfig.plane],
				aConfig.plane
				)
			);*/
	}
}



}//Renderer
}//GUI
}//M4D
