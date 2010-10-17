
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

/*void
ImageDataRenderer::SetRenderingMode( RenderingMode aMode )
{

}
*/

void
ImageDataRenderer::Render()
{
#define _COMMENT
#ifdef _COMMENT
	if( true ) {
		_cgEffect.SetParameter( "gImageData3D", *_textureData );
		_cgEffect.SetParameter( "gImageDataResolution3D", _textureData->GetDimensionedInterface<3>().GetSize() );
		_cgEffect.SetParameter( "gMappedIntervalBands", Vector2f( 0, 65535 )/*_textureData->GetMappedInterval()*/ );
		_cgEffect.SetParameter( "gWLWindow", _wlWindow );

		_cgEffect.ExecuteTechniquePass( 
				"WLWindow_3D", 
				boost::bind( &M4D::GLDrawVolumeSlice, 
					_textureData->GetDimensionedInterface< 3 >().GetMinimum(), 
					_textureData->GetDimensionedInterface< 3 >().GetMaximum(), 
					(float32)mSliceViewConfig.currentSlice[ mSliceViewConfig.plane ] * _textureData->GetDimensionedInterface< 3 >().GetElementExtents()[mSliceViewConfig.plane], 
					mSliceViewConfig.plane 
					) 
				); 
		glFlush();

	} else {

		_cgEffect.SetParameter( "gImageData3D", *_textureData );
		_cgEffect.SetParameter( "gImageDataResolution3D", _textureData->GetDimensionedInterface<3>().GetSize() );
		_cgEffect.SetParameter( "gMappedIntervalBands", Vector2f( 0, 65535 )/*_textureData->GetMappedInterval()*/ );
		_cgEffect.SetTextureParameter( "gTransferFunction1D", mTransferFunctionTexture->GetTextureID() );
		_cgEffect.SetParameter( "gTransferFunction1DInterval", mTransferFunctionTexture->GetMappedInterval() );

		_cgEffect.ExecuteTechniquePass( 
				"TransferFunction1D_3D", 
				boost::bind( &M4D::GLDrawVolumeSlice, 
					_textureData->GetDimensionedInterface< 3 >().GetMinimum(), 
					_textureData->GetDimensionedInterface< 3 >().GetMaximum(), 
					(float32)mSliceViewConfig.currentSlice[ mSliceViewConfig.plane ] * _textureData->GetDimensionedInterface< 3 >().GetElementExtents()[mSliceViewConfig.plane], 
					mSliceViewConfig.plane 
					) 
				); 
		glFlush();
	}
#else
	mViewConfig3D.camera.SetCenterPosition( 0.5f * (_textureData->GetDimensionedInterface< 3 >().GetMaximum() + _textureData->GetDimensionedInterface< 3 >().GetMinimum()) );

	{
		glEnable( GL_BLEND );
		glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
		glDepthFunc(GL_LEQUAL);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		//Set viewing parameters
		M4D::SetViewAccordingToCamera( mViewConfig3D.camera );

		glMatrixMode(GL_MODELVIEW);
		
		//Texture coordinate generation
		M4D::SetVolumeTextureCoordinateGeneration( _textureData->GetDimensionedInterface< 3 >().GetMinimum(), _textureData->GetDimensionedInterface< 3 >().GetRealSize() );
		/*_transferFuncShaderConfig.dataTexture = _texName;
		glBindTexture( GL_TEXTURE_1D, 0 );
		glBindTexture( GL_TEXTURE_2D, 0 );
		glBindTexture( GL_TEXTURE_3D, 0 );
		glDisable(GL_TEXTURE_3D);
		glDisable(GL_TEXTURE_2D);
		glDisable(GL_TEXTURE_1D);*/


		glColor3f( 1.0f, 0.0f, 0.0f );
		M4D::GLDrawBoundingBox( _textureData->GetDimensionedInterface< 3 >().GetMinimum(), _textureData->GetDimensionedInterface< 3 >().GetMaximum() );

		unsigned sliceCount = 250;

		_cgEffect.SetParameter( "gImageData3D", *_textureData );
		_cgEffect.SetParameter( "gImageDataResolution3D", _textureData->GetDimensionedInterface<3>().GetSize() );
		_cgEffect.SetParameter( "gMappedIntervalBands", Vector2f( 0, 65535 )/*_textureData->GetMappedInterval()*/ );
		_cgEffect.SetTextureParameter( "gTransferFunction1D", mTransferFunctionTexture->GetTextureID() );
		_cgEffect.SetParameter( "gTransferFunction1DInterval", mTransferFunctionTexture->GetMappedInterval() );
		_cgEffect.SetParameter( "gAlphaModulation", /*2.0f / (float)sliceCount*/ 1.0f );
		_cgEffect.SetParameter( "gLightPosition", Vector3f( 3000.0f, 3000.0f, -3000.0f ) );
		_cgEffect.SetParameter( "gLightColor", Vector3f( 1.0f, 1.0f, 1.0f ) );
		_cgEffect.SetParameter( "gEyePosition", mViewConfig3D.camera.GetEyePosition() );

		_cgEffect.ExecuteTechniquePass(
				"TransferFunction1DShading_3D", 
				boost::bind( &M4D::GLDrawVolumeSlices, 
					M4D::BoundingBox3D( _textureData->GetDimensionedInterface< 3 >().GetMinimum(), _textureData->GetDimensionedInterface< 3 >().GetMaximum() ),
					mViewConfig3D.camera,
					sliceCount,
					1.0f
					) 
				); 

		M4D::DisableVolumeTextureCoordinateGeneration();
		M4D::CheckForGLError( "OGL error : " );
		glFlush();		

	}
#endif
}


} /*namespace M4D*/
} /*namespace GUI*/
