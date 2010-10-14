#ifndef IMAGE_DATA_RENDERER_H
#define IMAGE_DATA_RENDERER_H

#include "common/Common.h"
#include "GUI/utils/CgShaderTools.h"
#include "GUI/utils/GLTextureImage.h"
#include <boost/bind.hpp>
#include "GUI/utils/TransferFunctionBuffer.h"

namespace M4D
{
namespace GUI
{


typedef uint32 RenderingMode;

struct SliceViewConfig
{
	SliceViewConfig(): plane( XY_PLANE ), currentSlice( 0 )
	{}

	CartesianPlanes		plane;

	Vector< int32, 3 >	currentSlice;

	ViewConfiguration2D	viewConfiguration;
};

class ImageDataRenderer
{
public:
	void
	Initialize();

	void
	Finalize();

	void
	SetImageData( GLTextureImage::Ptr aData );

	void
	SetMaskData( GLTextureImage::Ptr aData );
	
	void
	SetTransferFunction( GLTransferFunctionBuffer1D::Ptr aTFunction );
	
	void
	SetMaskColorMap( GLTextureImage::Ptr aData );

	void
	SetRenderingMode( RenderingMode aMode );

	void
	SetLUTWindow( const Vector< float32, 2 > &aLUTWindow )
	{ 
		_wlWindow = aLUTWindow; 
		LOG( _wlWindow );	
	}

	void
	Render();

	SliceViewConfig &
	GetSliceViewConfig()
	{ return _sliceViewConfig; }

protected:

	SliceViewConfig 			_sliceViewConfig;

	Vector2f				_wlWindow;
	GLTransferFunctionBuffer1D::Ptr 	mTransferFunctionTexture;


	GLTextureImage::Ptr			_textureData;

	CGcontext   				_cgContext;
	CgBrightnessContrastShaderConfig	_shaderConfig;

	CgEffect				_cgEffect;

private:

};

} /*namespace M4D*/
} /*namespace GUI*/

#endif /*IMAGE_DATA_RENDERER_H*/
