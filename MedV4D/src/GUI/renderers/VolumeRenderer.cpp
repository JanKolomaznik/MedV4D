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
VolumeRenderer::Render(VolumeRenderer::RenderingConfiguration & aConfig, const soglu::GLViewSetup &aViewSetup)
{
	soglu::GLTextureImageTyped<3>::Ptr primaryData = aConfig.primaryImageData.lock();
	if ( !primaryData ) {
		_THROW_ ErrorHandling::EObjectUnavailable( "Primary texture not available" );
	}
	soglu::GLTextureImageTyped<3>::Ptr secondaryData = aConfig.secondaryImageData.lock();
	soglu::GLTextureImageTyped<3>::Ptr maskData = aConfig.maskImageData.lock();

	soglu::BoundingBox3D bbox(primaryData->getExtents().realMinimum, primaryData->getExtents().realMaximum);
	if ( aConfig.enableVolumeRestrictions ) {
		applyVolumeRestrictionsOnBoundingBox( bbox, aConfig.volumeRestrictions );
	}
	aConfig.renderingConfiguration.boundingBox = bbox;
	switch ( aConfig.colorTransform ) {
	case ctTransferFunction1D:
		{
			if (maskData) {
				if (secondaryData) {
					transferFunctionRendering(
						aConfig.renderingConfiguration,
						MaskedImageWithSecondaryImage{*primaryData, *secondaryData, *maskData },
						aConfig.renderingQuality,
						aConfig.clipPlanes,
						aConfig.transferFunctionOptions
						);
				} else {
					transferFunctionRendering(
						aConfig.renderingConfiguration,
						MaskedImage{*primaryData, *maskData },
						aConfig.renderingQuality,
						aConfig.clipPlanes,
						aConfig.transferFunctionOptions
						);
				}
			} else {
				if (secondaryData) {
					transferFunctionRendering(
						aConfig.renderingConfiguration,
						ImageWithSecondaryImage{*primaryData, *secondaryData},
						aConfig.renderingQuality,
						aConfig.clipPlanes,
						aConfig.transferFunctionOptions
						);
				} else {
					transferFunctionRendering(
						aConfig.renderingConfiguration,
						*primaryData,
						aConfig.renderingQuality,
						aConfig.clipPlanes,
						aConfig.transferFunctionOptions
						);
				}
			}
		}
		break;
	case ctMaxIntensityProjection:
	case ctBasic:
		{
			if (maskData) {
				densityRendering(
					aConfig.renderingConfiguration,
					MaskedImage {*primaryData, *maskData },
					aConfig.renderingQuality,
					aConfig.clipPlanes,
					aConfig.densityOptions
					);
			} else {
				densityRendering(
					aConfig.renderingConfiguration,
					*primaryData,
					aConfig.renderingQuality,
					aConfig.clipPlanes,
					aConfig.densityOptions
					);
			}
		}
		break;
	case ctIsoSurfaces:
		{
			if (maskData) {
				isosurfaceRendering(
					aConfig.renderingConfiguration,
					MaskedImage {*primaryData, *maskData },
					aConfig.renderingQuality,
					aConfig.clipPlanes,
					aConfig.isoSurfaceOptions
					);
			} else {
				isosurfaceRendering(
					aConfig.renderingConfiguration,
					*primaryData,
					aConfig.renderingQuality,
					aConfig.clipPlanes,
					aConfig.isoSurfaceOptions
					);
			}
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

