#include "MedV4D/GUI/renderers/SliceRenderer.h"

namespace M4D
{
namespace GUI
{
namespace Renderer
{
	
boost::filesystem::path gSliceRendererShaderPath;

void
SliceRenderer::Initialize()
{
	initializeCg();
	mCgEffect.initialize( gSliceRendererShaderPath/*"ImageRender.cgfx"*/ );


	mAvailableColorTransforms.clear();
	//mAvailableColorTransforms.push_back( WideNameIdPair( L"LUT window", ctLUTWindow ) );
	//mAvailableColorTransforms.push_back( WideNameIdPair( L"Transfer function", ctTransferFunction1D ) );
	//mAvailableColorTransforms.push_back( WideNameIdPair( L"Region colormap", ctSimpleColorMap ) );
	mAvailableColorTransforms.push_back( ColorTransformNameIDList::value_type( "LUT window", ctLUTWindow ) );
	mAvailableColorTransforms.push_back( ColorTransformNameIDList::value_type( "Transfer function", ctTransferFunction1D ) );
	mAvailableColorTransforms.push_back( ColorTransformNameIDList::value_type( "Region colormap", ctSimpleColorMap ) );
}

void
SliceRenderer::Finalize()
{
	//TODO
}


void
SliceRenderer::Render( SliceRenderer::RenderingConfiguration & aConfig, const GLViewSetup &aViewSetup )
{
	GLTextureImageTyped<3>::Ptr primaryData = aConfig.primaryImageData.lock();
	if ( !primaryData ) {
		_THROW_ ErrorHandling::EObjectUnavailable( "Primary texture not available" );
	}
	
	mCgEffect.setParameter( "gPrimaryImageData3D", *primaryData );
	mCgEffect.setParameter( "gMappedIntervalBands", primaryData->GetMappedInterval() );
	
	GLTextureImageTyped<3>::Ptr secondaryData = aConfig.secondaryImageData.lock();
	if( secondaryData ) {
		mCgEffect.setParameter( "gSecondaryImageData3D", *secondaryData );
	}
	
	mCgEffect.setParameter( "gEnableInterpolation", aConfig.enableInterpolation );
	
	mCgEffect.setParameter( "gViewSetup", aViewSetup );

	std::string techniqueName;
	GLTransferFunctionBuffer1D::ConstPtr transferFunction;
	switch ( aConfig.colorTransform ) {
	case ctLUTWindow:
		{
			mCgEffect.setParameter( "gWLWindow", aConfig.lutWindow );
			techniqueName = "WLWindow_3D";
		} 
		break;
	case ctTransferFunction1D:
		{
			transferFunction = aConfig.transferFunction.lock();
			if ( !transferFunction ) {
				_THROW_ M4D::ErrorHandling::EObjectUnavailable( "Transfer function no available" );
			}
			mCgEffect.setParameter( "gTransferFunction1D", *transferFunction );
			techniqueName = "TransferFunction1D_3DNoBlending";
		}
		break;
	case ctSimpleColorMap:
		techniqueName = "SimpleRegionColorMap_3D";
		break;
	case ctMaxIntensityProjection:
		ASSERT( false );
		break;
	default:
		ASSERT( false );
	}

	mCgEffect.executeTechniquePass( 
			techniqueName, 
			boost::bind( &M4D::GLDrawVolumeSlice3D, 
				primaryData->getExtents().realMinimum, 
				primaryData->getExtents().realMaximum, 
				float32(aConfig.currentSlice[ aConfig.plane ]+0.5f) * primaryData->getExtents().elementExtents[aConfig.plane], 
				aConfig.plane 
				) 
			); 
	
	if( secondaryData ) {
		mCgEffect.executeTechniquePass( 
			"OverlayMask_3D", 
			boost::bind( &M4D::GLDrawVolumeSlice3D, 
				secondaryData->getExtents().realMinimum, 
				secondaryData->getExtents().realMaximum, 
				float32(aConfig.currentSlice[ aConfig.plane ]+0.5f) * secondaryData->getExtents().elementExtents[aConfig.plane], 
				aConfig.plane 
				) 
			);
	}
}



}//Renderer
}//GUI
}//M4D
