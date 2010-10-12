
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

	_wlWindow[0] = 0.008f;
	_wlWindow[1] = 0.007f;
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
ImageDataRenderer::SetTransferFunction( GLTextureImage::Ptr aData )
{

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
	/*
	if( ! _textureData ) return;

	glBindTexture( GL_TEXTURE_1D, 0 );
	glBindTexture( GL_TEXTURE_2D, 0 );
	glBindTexture( GL_TEXTURE_3D, 0 );
	glDisable(GL_TEXTURE_3D);
	glDisable(GL_TEXTURE_2D);
	glDisable(GL_TEXTURE_1D);


	_shaderConfig.textureName = _textureData->GetTextureGLID();
	_shaderConfig.brightnessContrast = _wlWindow;
	_shaderConfig.Enable();
	
	CheckForCgError("Check before drawing ", _cgContext );
	//M4D::GLDrawTexturedQuad( _textureData->GetMinimum3D(), _textureData->GetMaximum3D() );
	SetToViewConfiguration2D( _sliceViewConfig.viewConfiguration );
	M4D::GLDrawVolumeSlice( _textureData->GetMinimum3D(), _textureData->GetMaximum3D(), (float32)_sliceViewConfig.currentSlice[ _sliceViewConfig.plane ] * _textureData->GetElementExtents3D()[_sliceViewConfig.plane], _sliceViewConfig.plane );
	
	_shaderConfig.Disable();
	*/
	
	_cgEffect.SetParameter( "gImageData3D", *_textureData );
	_cgEffect.SetParameter( "gImageDataResolution3D", _textureData->GetDimensionedInterface<3>().GetSize() );
	_cgEffect.SetParameter( "gMappedIntervalBands", Vector3f( 0, 65535 )/*_textureData->GetMappedInterval()*/ );
	_cgEffect.SetParameter( "gWLWindow", _wlWindow );

	_cgEffect.ExecuteTechniquePass( 
			"WLWindow3D", 
			boost::bind( &M4D::GLDrawVolumeSlice, 
				_textureData->GetDimensionedInterface< 3 >().GetMinimum(), 
				_textureData->GetDimensionedInterface< 3 >().GetMaximum(), 
				(float32)_sliceViewConfig.currentSlice[ _sliceViewConfig.plane ] * _textureData->GetDimensionedInterface< 3 >().GetElementExtents()[_sliceViewConfig.plane], 
				_sliceViewConfig.plane 
				) 
			); 
	glFlush();
}


} /*namespace M4D*/
} /*namespace GUI*/
