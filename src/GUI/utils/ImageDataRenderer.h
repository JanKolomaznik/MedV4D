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
enum RendererType
{
	rt2DAlignedSlices,
	rt3DGeneralSlices,
	rt3D
};

enum ColorTransform
{
	ctLUTWindow,
	ctTransferFunction1D,
	ctMaxIntensityProjection,
	ctSimpleColorMap
};


typedef uint32 RenderingMode;

struct SliceViewConfig
{
	SliceViewConfig(): plane( XY_PLANE ), currentSlice( 0 )
	{}

	CartesianPlanes		plane;

	Vector< int32, 3 >	currentSlice;

	ViewConfiguration2D	viewConfiguration;
};

struct ViewConfig3D
{
	ViewConfig3D(): camera( Vector<float,3>( 0.0f, 0.0f, 750.0f ), Vector<float,3>( 0.0f, 0.0f, 0.0f ) )
	{}


	Camera		camera;
};

class ImageDataRenderer
{
public:
	ImageDataRenderer() : mRendererType( rt2DAlignedSlices ), mColorTransform( ctLUTWindow ), mFineRendering( false ), mShadingEnabled( true )
	{}

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
	SetRendererType( int aRendererType );

	void
	SetColorTransformType( int aColorTransform );

	int
	GetRendererType()
	{
		return mRendererType;
	}

	int
	GetColorTransformType()
	{
		return mColorTransform;
	}


	void
	SetLUTWindow( const Vector< float32, 2 > &aLUTWindow )
	{ 
		_wlWindow = aLUTWindow; 
	}

	void
	Render();

	void
	RenderAlignedSlices();

	void
	RenderGeneralSlices();
	
	void
	RenderVolume();

	SliceViewConfig &
	GetSliceViewConfig()
	{ return mSliceViewConfig; }

	ViewConfig3D &
	GetViewConfig3D()
	{ return mViewConfig3D; }

	void
	FineRender()
	{
		mFineRendering = true;
	}
	
	void
	EnableShading( bool aEnable )
	{
		mShadingEnabled = aEnable;
	}

	bool
	IsShadingEnabled() const
	{
		return mShadingEnabled;
	}
protected:
	

	SliceViewConfig 			mSliceViewConfig;

	ViewConfig3D				mViewConfig3D;

	Vector2f				_wlWindow;
	GLTransferFunctionBuffer1D::Ptr 	mTransferFunctionTexture;


	GLTextureImage::Ptr			_textureData;

	CGcontext   				_cgContext;
	CgBrightnessContrastShaderConfig	_shaderConfig;

	CgEffect				_cgEffect;

	int 					mRendererType;
	int	 				mColorTransform;



	bool	mFineRendering; //TODO delete
	bool	mShadingEnabled;
private:
	GLuint	mNoiseMap;
};

} /*namespace M4D*/
} /*namespace GUI*/

#endif /*IMAGE_DATA_RENDERER_H*/
