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
	InitializeCg();
	mCgEffect.Initialize( gSliceRendererShaderPath/*"ImageRender.cgfx"*/ );


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
	ASSERT( aConfig.primaryImageData != NULL );

	mCgEffect.SetParameter( "gPrimaryImageData3D", *aConfig.primaryImageData );
	mCgEffect.SetParameter( "gMappedIntervalBands", aConfig.primaryImageData->GetMappedInterval() );
	if( aConfig.secondaryImageData ) {
		mCgEffect.SetParameter( "gSecondaryImageData3D", *aConfig.secondaryImageData );
	}
	
	mCgEffect.SetParameter( "gEnableInterpolation", aConfig.enableInterpolation );
	
	mCgEffect.SetParameter( "gViewSetup", aViewSetup );

	std::string techniqueName;

	switch ( aConfig.colorTransform ) {
	case ctLUTWindow:
		{
			mCgEffect.SetParameter( "gWLWindow", aConfig.lutWindow );
			techniqueName = "WLWindow_3D";
		} 
		break;
	case ctTransferFunction1D:
		{
			if ( !aConfig.transferFunction ) {
				_THROW_ M4D::ErrorHandling::EObjectUnavailable( "Transfer function no available" );
			}

			//mCgEffect.SetTextureParameter( "gTransferFunction1D", aConfig.transferFunction->GetTextureID() );
			mCgEffect.SetParameter( "gTransferFunction1D", *(aConfig.transferFunction) );
			//mCgEffect.SetParameter( "gTransferFunction1DInterval", aConfig.transferFunction->GetMappedInterval() );
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

	mCgEffect.ExecuteTechniquePass( 
			techniqueName, 
			boost::bind( &M4D::GLDrawVolumeSlice3D, 
				aConfig.primaryImageData->GetMinimum(), 
				aConfig.primaryImageData->GetMaximum(), 
				float32(aConfig.currentSlice[ aConfig.plane ]+0.5f) * aConfig.primaryImageData->GetElementExtents()[aConfig.plane], 
				aConfig.plane 
				) 
			); 
	
	if( aConfig.secondaryImageData ) {
		mCgEffect.ExecuteTechniquePass( 
			"OverlayMask_3D", 
			boost::bind( &M4D::GLDrawVolumeSlice3D, 
				aConfig.secondaryImageData->GetMinimum(), 
				aConfig.secondaryImageData->GetMaximum(), 
				float32(aConfig.currentSlice[ aConfig.plane ]+0.5f) * aConfig.secondaryImageData->GetElementExtents()[aConfig.plane], 
				aConfig.plane 
				) 
			);
	}
}



}//Renderer
}//GUI
}//M4D
