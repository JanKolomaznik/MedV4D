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
SliceRenderer::Render( SliceRenderer::RenderingConfiguration & aConfig, bool aSetupView )
{
	ASSERT( aConfig.imageData != NULL );

	if( aSetupView ) {
		SetToViewConfiguration2D( aConfig.viewConfig );
	}

	mCgEffect.SetParameter( "gImageData3D", *aConfig.imageData );
	mCgEffect.SetParameter( "gMappedIntervalBands", aConfig.imageData->GetMappedInterval() );
	mCgEffect.SetParameter( "gEnableInterpolation", aConfig.enableInterpolation );

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
				aConfig.imageData->GetMinimum(), 
				aConfig.imageData->GetMaximum(), 
				float32(aConfig.currentSlice[ aConfig.plane ]+0.5f) * aConfig.imageData->GetElementExtents()[aConfig.plane], 
				aConfig.plane 
				) 
			); 
}



}//Renderer
}//GUI
}//M4D
