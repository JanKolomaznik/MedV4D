
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
ImageDataRenderer::SetRenderingMode( RenderingMode aMode )
{

}

void
ImageDataRenderer::Render()
{
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
					(float32)_sliceViewConfig.currentSlice[ _sliceViewConfig.plane ] * _textureData->GetDimensionedInterface< 3 >().GetElementExtents()[_sliceViewConfig.plane], 
					_sliceViewConfig.plane 
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
					(float32)_sliceViewConfig.currentSlice[ _sliceViewConfig.plane ] * _textureData->GetDimensionedInterface< 3 >().GetElementExtents()[_sliceViewConfig.plane], 
					_sliceViewConfig.plane 
					) 
				); 
		glFlush();
	}

}


} /*namespace M4D*/
} /*namespace GUI*/
