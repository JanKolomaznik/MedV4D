#ifndef _SIMPLE_2D_VIEWER_H
#define _SIMPLE_2D_VIEWER_H

#include "GUI/utils/OGLDrawing.h"
#include <QtGui/QWidget>
#include "Imaging/ImageRegion.h"
#include "GUI/utils/ViewConfiguration.h"
#include "common/Vector.h"

#include "GUI/utils/CgShaderTools.h"


template< typename SupportGLWidget >
class Simple2DViewer: public SupportGLWidget
{
public:
	typedef M4D::Imaging::AImageRegionDim< 2 > 	ARegion2D;

	Simple2DViewer( QWidget *parent = NULL ): SupportGLWidget( parent ), _region( NULL ), _mask( NULL ), _viewConfiguration( Vector< float32, 2 >(), 0.005 ), _linearInterpolation( false )
	{}

	~Simple2DViewer()
	{
		_shaderConfig.Finalize();
		cgDestroyContext(_cgContext);
	}

	void
	SetImageRegion( ARegion2D *region )
		{ 
			_region = region; 
			_viewConfiguration = GetOptimalViewConfiguration( 
					_region->GetRealMinimum(),
					_region->GetRealMaximum(),
					Vector< unsigned, 2 >( this->width(), this->height() ) 
					);		
		}
	void
	SetMaskRegion( ARegion2D *region )
		{ 
			_mask = region; 
			
		}

	void
	initializeGL()
	{
		glClearColor(0,0,0,1);
		glShadeModel(GL_FLAT);
		//glDisable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glEnable( GL_BLEND );
		glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
		glDepthFunc(GL_LEQUAL);

		_cgContext = cgCreateContext();
		CheckForCgError("creating context ", _cgContext );

		_shaderConfig.Initialize( _cgContext, "LUT.cg", "LUT_texture" );
		_blendShaderConfig.Initialize( _cgContext, "MaskBlend.cg", "MaskBlend" );
	}
	
	void
	resizeGL( int width, int height )
	{
		glViewport(0, 0, width, height);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
	}

	void	
	paintGL()
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glLoadIdentity();
		_viewConfiguration = GetOptimalViewConfiguration( 
					_region->GetRealMinimum(),
					_region->GetRealMaximum(),
					Vector< unsigned, 2 >( this->width(), this->height() ) ,
					ztFIT
					);
		_shaderConfig.brightnessContrast[0] = 8.0f;

		M4D::SetToViewConfiguration2D( _viewConfiguration );
	
		if( _region ) {
			//M4D::GLDrawImageDataContrastBrightness( *_region, 0.0f, 1.0f, _linearInterpolation );
			GLuint texName = M4D::GLPrepareTextureFromImageData( *_region, false );
			
			_shaderConfig.textureName = texName;
			_shaderConfig.Enable();

			CheckForCgError("Check befor drawing ", _cgContext );
			M4D::GLDrawTexturedQuad( _region->GetRealMinimum(), _region->GetRealMaximum() );
			
			_shaderConfig.Disable();
			
			glDeleteTextures( 1, &texName );
			glFlush();
		}
		if( _mask ) {
			GLuint texName = M4D::GLPrepareTextureFromMaskData( *_mask, false );

			_blendShaderConfig.textureName = texName;
			_blendShaderConfig.blendColor = Vector< float, 4 >( 1.0f, 0.0f, 0.0f, 1.0f );
			_blendShaderConfig.Enable();	

			M4D::GLDrawTexturedQuad( _mask->GetRealMinimum(), _mask->GetRealMaximum() );
			
			_blendShaderConfig.Disable();

			glDeleteTextures( 1, &texName );
			glFlush();

		}
		
	}
protected:
	ARegion2D		*_region;
	ARegion2D		*_mask;
	ViewConfiguration2D	_viewConfiguration;
	bool			_linearInterpolation;
	float			_brightness;
	float			_contrast;

	CGcontext   				_cgContext;
	CgBrightnessContrastShaderConfig	_shaderConfig;
	CgMaskBlendShaderConfig			_blendShaderConfig;
};

#endif /*_SIMPLE_2D_VIEWER_H*/

