#include "MedV4D/GUI/renderers/VolumeRenderer.h"

#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/verbose_operator.hpp>

namespace M4D
{
namespace GUI
{
namespace Renderer
{

boost::filesystem::path gVolumeRendererShaderPath;

void
applyVolumeRestrictionsOnBoundingBox( soglu::BoundingBox3D &aBBox, const VolumeRestrictions &aVolumeRestrictions )
{
	glm::fvec3 corner1 = aBBox.getMin();
	glm::fvec3 corner2 = aBBox.getMax();
	glm::fvec3 size = corner2 - corner1;

	Vector3f i1Tmp, i2Tmp;
	aVolumeRestrictions.get3D(i1Tmp, i2Tmp);

	glm::fvec3 i1(i1Tmp[0], i1Tmp[1], i1Tmp[2]);
	glm::fvec3 i2(i2Tmp[0], i2Tmp[1], i2Tmp[2]);
	//LOG( "Restrictions : " << i1 << " - " << i2 );

	corner2 = corner1 + (i2 * size);
	corner1 += (i1 * size);
	aBBox = soglu::BoundingBox3D( corner1, corner2 );
}

namespace detail {

static const uint64 FLAGS_TO_NAME_SUFFIXES_MASK = rf_SHADING | rf_JITTERING | rf_PREINTEGRATED;
static const std::string gFlagsToNameSuffixes[] =
	{
		std::string("Simple"),
		std::string("Shading"),
		std::string("Jittering"),
		std::string("JitteringShading"),
		std::string("PreintegratedSimple"),
		std::string("PreintegratedShading"),
		std::string("PreintegratedJittering"),
		std::string("PreintegratedJitteringShading"),

		std::string("UNDEFINED_COMBINATION")
	};

}

void
VolumeRenderer::Render( VolumeRenderer::RenderingConfiguration & aConfig, const soglu::GLViewSetup &aViewSetup )
{
	soglu::GLTextureImageTyped<3>::Ptr primaryData = aConfig.primaryImageData.lock();
	if ( !primaryData ) {
		_THROW_ ErrorHandling::EObjectUnavailable( "Primary texture not available" );
	}

	soglu::GLTextureImageTyped<3>::Ptr secondaryData = aConfig.secondaryImageData.lock();

	size_t sliceCount = aConfig.sampleCount;
	soglu::BoundingBox3D bbox(primaryData->getExtents().realMinimum, primaryData->getExtents().realMaximum);
	if ( aConfig.enableVolumeRestrictions ) {
		applyVolumeRestrictionsOnBoundingBox( bbox, aConfig.volumeRestrictions );
	}

	switch ( aConfig.colorTransform ) {
	case ctTransferFunction1D:
		{
			vorgl::GLTransferFunctionBuffer1D::ConstPtr transferFunction;
			vorgl::GLTransferFunctionBuffer1D::ConstPtr integralTransferFunction;
			transferFunction = aConfig.transferFunction.lock();
			if ( !transferFunction ) {
				_THROW_ M4D::ErrorHandling::EObjectUnavailable( "Transfer function no available" );
			}

			integralTransferFunction = aConfig.integralTransferFunction.lock();

			vorgl::VolumeRenderer::TransferFunctionRenderFlags flags;
			if (aConfig.jitterEnabled) {
				flags.set(vorgl::VolumeRenderer::TFFlags::JITTERING);
			}
			if (aConfig.integralTFEnabled) {
				flags.set(vorgl::VolumeRenderer::TFFlags::PREINTEGRATED_TF);
			}
			if (aConfig.shadingEnabled) {
				flags.set(vorgl::VolumeRenderer::TFFlags::SHADING);
			}

			transferFunctionRendering(
				aConfig.camera,
				aViewSetup,
				*primaryData,
				bbox,
				(aConfig.integralTFEnabled ? *integralTransferFunction : *transferFunction),
				sliceCount,
				aConfig.enableCutPlane,
				aConfig.cutPlane,
				aConfig.enableInterpolation,
				aConfig.lightPosition,
				flags
				);

		}
		break;
	case ctMaxIntensityProjection:
	case ctBasic:
		{
			vorgl::VolumeRenderer::DensityRenderFlags flags;
			if (aConfig.jitterEnabled) {
				flags.set(vorgl::VolumeRenderer::DensityFlags::JITTERING);
			}
			if (aConfig.colorTransform == ctMaxIntensityProjection) {
				flags.set(vorgl::VolumeRenderer::DensityFlags::MIP);
			}
			densityRendering(
				aConfig.camera,
				aViewSetup,
				*primaryData,
				bbox,
				aConfig.lutWindow,
				sliceCount,
				aConfig.enableCutPlane,
				aConfig.cutPlane,
				aConfig.enableInterpolation,
				flags
     			);
		}
		break;
	case ctTestColorTransform:
		{
			vorgl::VolumeRenderer::TransferFunctionRenderFlags flags;
			rayCasting(
				aConfig.camera,
				aViewSetup,
				*primaryData,
				bbox,
				aConfig.lutWindow,
				sliceCount,
				aConfig.enableCutPlane,
				aConfig.cutPlane,
				aConfig.enableInterpolation,
				flags
     				);
		}
		break;
	default:
		ASSERT( false );
	}

	soglu::checkForGLError( "OGL error : " );
}


}//Renderer
}//GUI
}//M4D

