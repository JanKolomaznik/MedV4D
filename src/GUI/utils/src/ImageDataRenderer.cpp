
#include "GUI/utils/ImageDataRenderer.h"



namespace M4D
{
namespace GUI
{

void
ImageDataRenderer::Initialize()
{
	_cgContext = cgCreateContext();
	CheckForCgError("creating context ", _cgContext );

	_shaderConfig.Initialize( _cgContext, "LUT.cg", "SimpleBrightnessContrast3D" );
}

void
ImageDataRenderer::Finalize()
{
	_shaderConfig.Finalize();
	cgDestroyContext(_cgContext);
}

void
ImageDataRenderer::SetImageData( GLTextureImage::Ptr aData )
{
	_textureData = aData;
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
	if( ! _textureData ) return;

	glBindTexture( GL_TEXTURE_1D, 0 );
	glBindTexture( GL_TEXTURE_2D, 0 );
	glBindTexture( GL_TEXTURE_3D, 0 );
	glDisable(GL_TEXTURE_3D);
	glDisable(GL_TEXTURE_2D);
	glDisable(GL_TEXTURE_1D);


	_shaderConfig.textureName = _textureData->GetTextureGLID();
	_shaderConfig.brightnessContrast = _lutWindow;
	_shaderConfig.Enable();
	
	CheckForCgError("Check before drawing ", _cgContext );
	//M4D::GLDrawTexturedQuad( _textureData->GetMinimum3D(), _textureData->GetMaximum3D() );
	SetToViewConfiguration2D( _sliceViewConfig.viewConfiguration );
	M4D::GLDrawVolumeSlice( _textureData->GetMinimum3D(), _textureData->GetMaximum3D(), (float32)_sliceViewConfig.currentSlice[ _sliceViewConfig.plane ] * _textureData->GetElementExtents3D()[_sliceViewConfig.plane], _sliceViewConfig.plane );
	
	_shaderConfig.Disable();
	
	glFlush();
}


} /*namespace M4D*/
} /*namespace GUI*/
