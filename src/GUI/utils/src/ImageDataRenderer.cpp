
#include "GUI/utils/ImageDataRenderer.h"



namespace M4D
{
namespace GUI
{

void
ImageDataRenderer::Initialize()
{
	/*_cgContext = cgCreateContext();
	CheckForCgError("creating context ", _cgContext );

	_shaderConfig.Initialize( _cgContext, "LUT.cg", "SimpleBrightnessContrast3D" );*/

	InitializeCg();
	_cgEffect.Initialize( "ImageRender.cgfx" );
}

void
ImageDataRenderer::Finalize()
{
	/*_shaderConfig.Finalize();
	cgDestroyContext(_cgContext);*/
	_cgEffect.Finalize();
}

void
ImageDataRenderer::SetImageData( GLTextureImage::Ptr aData )
{
	_textureData = aData;
	//_cgEffect.SetParameter( "gImageData3D", *_textureData );
}

void
ImageDataRenderer::SetMaskData( GLTextureImage::Ptr aData )
{

}

void
ImageDataRenderer::SetTransferFunction( GLTransferFunctionBuffer1D::Ptr aTFunction )
{
	mTransferFunctionTexture = aTFunction;
}

void
ImageDataRenderer::SetMaskColorMap( GLTextureImage::Ptr aData )
{

}

void
ImageDataRenderer::SetRendererType( int aRendererType )
{
	//TODO 
	mRendererType = aRendererType;
}

void
ImageDataRenderer::SetColorTransformType( int aColorTransform )
{
	//TODO 
	mColorTransform = aColorTransform;
}

/*void
ImageDataRenderer::SetRenderingMode( RenderingMode aMode )
{

}
*/

void
ImageDataRenderer::Render()
{
	switch ( mRendererType ) {
	case rt2DAlignedSlices:
		RenderAlignedSlices();
		break;
	case rt3DGeneralSlices:
		RenderGeneralSlices();
		break;
	case rt3D:
		RenderVolume();
		break;
	default:
		ASSERT( false );
	};
}

void
ImageDataRenderer::RenderAlignedSlices()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	SetToViewConfiguration2D( GetSliceViewConfig().viewConfiguration );
	_cgEffect.SetParameter( "gImageData3D", *_textureData );
	_cgEffect.SetParameter( "gImageDataResolution3D", _textureData->GetDimensionedInterface<3>().GetSize() );
	_cgEffect.SetParameter( "gMappedIntervalBands", Vector2f( 0, 65535 )/*_textureData->GetMappedInterval()*/ );

	std::string techniqueName;

	switch ( mColorTransform ) {
	case ctLUTWindow:
		{
			_cgEffect.SetParameter( "gWLWindow", _wlWindow );
			techniqueName = "WLWindow_3D";
		} 
		break;
	case ctTransferFunction1D:
		{
			_cgEffect.SetTextureParameter( "gTransferFunction1D", mTransferFunctionTexture->GetTextureID() );
			_cgEffect.SetParameter( "gTransferFunction1DInterval", mTransferFunctionTexture->GetMappedInterval() );
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

	_cgEffect.ExecuteTechniquePass( 
			techniqueName, 
			boost::bind( &M4D::GLDrawVolumeSlice, 
				_textureData->GetDimensionedInterface< 3 >().GetMinimum(), 
				_textureData->GetDimensionedInterface< 3 >().GetMaximum(), 
				(float32)mSliceViewConfig.currentSlice[ mSliceViewConfig.plane ] * _textureData->GetDimensionedInterface< 3 >().GetElementExtents()[mSliceViewConfig.plane], 
				mSliceViewConfig.plane 
				) 
			); 
	//glFlush();
}

void
ImageDataRenderer::RenderGeneralSlices()
{
	//TODO
	ASSERT( false );
}

void
ImageDataRenderer::RenderVolume()
{

	mViewConfig3D.camera.SetTargetPosition( 0.5f * (_textureData->GetDimensionedInterface< 3 >().GetMaximum() + _textureData->GetDimensionedInterface< 3 >().GetMinimum()) );
	mViewConfig3D.camera.SetFieldOfView( 45.0f );


	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//Set viewing parameters
	M4D::SetViewAccordingToCamera( mViewConfig3D.camera );

	glMatrixMode(GL_MODELVIEW);

	unsigned sliceCount = 250;
	float renderingSliceThickness = 1.0f;
	if ( mFineRendering ) {
		sliceCount *=15;
		renderingSliceThickness /= 15;
		mFineRendering = false;
	}

	glColor3f( 1.0f, 0.0f, 0.0f );
	M4D::GLDrawBoundingBox( _textureData->GetDimensionedInterface< 3 >().GetMinimum(), _textureData->GetDimensionedInterface< 3 >().GetMaximum() );

	_cgEffect.SetParameter( "gImageData3D", *_textureData );
	_cgEffect.SetParameter( "gImageDataResolution3D", _textureData->GetDimensionedInterface<3>().GetSize() );
	_cgEffect.SetParameter( "gMappedIntervalBands", Vector2f( 0, 65535 )/*_textureData->GetMappedInterval()*/ );
	_cgEffect.SetParameter( "gLightPosition", Vector3f( 3000.0f, 3000.0f, -3000.0f ) );
	_cgEffect.SetParameter( "gLightColor", Vector3f( 1.0f, 1.0f, 1.0f ) );
	_cgEffect.SetParameter( "gEyePosition", mViewConfig3D.camera.GetEyePosition() );
	_cgEffect.SetParameter( "gRenderingSliceThickness", renderingSliceThickness );


	std::string techniqueName;
	switch ( mColorTransform ) {
	case ctTransferFunction1D:
		{
			_cgEffect.SetTextureParameter( "gTransferFunction1D", mTransferFunctionTexture->GetTextureID() );
			_cgEffect.SetParameter( "gTransferFunction1DInterval", mTransferFunctionTexture->GetMappedInterval() );

			if ( mShadingEnabled ) {
				techniqueName = "TransferFunction1DShading_3D";
			} else {
				techniqueName = "TransferFunction1D_3D";
			}
		}
		break;
	case ctMaxIntensityProjection:
		{
			_cgEffect.SetParameter( "gWLWindow", _wlWindow );
			techniqueName = "WLWindowMIP_3D";
		}
		break;
	default:
		ASSERT( false );
	}

	M4D::SetVolumeTextureCoordinateGeneration( _textureData->GetDimensionedInterface< 3 >().GetMinimum(), _textureData->GetDimensionedInterface< 3 >().GetRealSize() );
	_cgEffect.ExecuteTechniquePass(
			techniqueName, 
			boost::bind( &M4D::GLDrawVolumeSlices, 
				M4D::BoundingBox3D( _textureData->GetDimensionedInterface< 3 >().GetMinimum(), _textureData->GetDimensionedInterface< 3 >().GetMaximum() ),
				mViewConfig3D.camera,
				sliceCount,
				1.0f
				) 
			); 

	M4D::DisableVolumeTextureCoordinateGeneration();
	M4D::CheckForGLError( "OGL error : " );
	//glFlush();
}



} /*namespace M4D*/
} /*namespace GUI*/
