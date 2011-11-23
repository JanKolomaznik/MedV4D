#ifndef _SIMPLE_VOLUME_VIEWER_H
#define _SIMPLE_VOLUME_VIEWER_H


#include "GUI/utils/OGLDrawing.h"
#include "GUI/utils/CgShaderTools.h"
#include <QtGui/QWidget>
#include "Imaging/ImageRegion.h"
#include "GUI/utils/ViewConfiguration.h"
#include "MedV4D/Common/Vector.h"
#include "MedV4D/Common/MathTools.h"

#include "GUI/utils/DrawingTools.h"

#include <QtGui/QMouseEvent>
#include <QtGui/QWheelEvent>
#include <QtCore/QPoint>

template< typename SupportGLWidget >
class SimpleVolumeViewer: public SupportGLWidget
{
public:
	typedef M4D::Imaging::AImageRegionDim< 3 > 	ARegion3D;
	typedef M4D::Imaging::AImageRegionDim< 2 > 	ARegion2D;

	SimpleVolumeViewer( QWidget *parent = NULL )
		: SupportGLWidget( parent ), _region( NULL ), 
		_camera( Vector<float,3>( 0.0f, 0.0f, 1500.0f ), Vector<float,3>( 0.0f, 0.0f, 0.0f ) ), _linearInterpolation( true )
	{
		_cutPlane = 1.0f;
		_mouseDown = false;
		_slicePos = 1900;
		_texName = 0;
		_camera.SetFieldOfView( 10.0f );
		_plane = XY_PLANE;
		_sliceCoord = 0.5f;
		_volumeRendering = true;
	}

	~SimpleVolumeViewer()
	{
		_transferFuncShaderConfig.Finalize();
		_brightnessContrastShaderConfig.Finalize();
		cgDestroyContext(_cgContext);
		glDeleteTextures( 1, &_texName );
	}

	void
	SetImageRegion( ARegion3D *region )
		{ 
			if( region == NULL ) {
				_THROW_ M4D::ErrorHandling::ENULLPointer( "NULL pointer to ARegion3D" );
			}

			_region = region; 
			_bbox = M4D::BoundingBox3D( _region->GetRealMinimum(), _region->GetRealMaximum() );

			Vector< float, 3> center( 0.5f * (_region->GetRealMaximum() - _region->GetRealMinimum()) );
			_camera.SetCenterPosition( center );
			/*_viewConfiguration = GetOptimalViewConfiguration( 
					_region->GetRealMinimum(),
					_region->GetRealMaximum(),
					Vector< unsigned, 2 >( this->width(), this->height() ) 
					);*/ 		
		}

	GLuint 
	CreateTransferFunction()
	{
		uint32 * func = new uint32[ 256 ];


		for( unsigned i=0; i < 5; ++i ) {
			func[i] = 0;
		}
		for( unsigned i=5; i < 256; ++i ) {
				func[i] =  0x353545FF;
		}
		/*for( unsigned i=0; i < 256; ++i ) {
			if( i < 8 ) {
				func[i] = 0;
			} else {
				func[i] = Min(i+100, (unsigned) 255)<<24 | (i/2) << 16 | 45; //xFF0000FF;
			}
		}*/

		GLuint texName;

		// opengl texture setup functions
		glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
		glGenTextures( 1, &texName );

		glBindTexture ( GL_TEXTURE_1D, texName );
		glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );

		glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP );
		glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
		glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );

		glEnable( GL_TEXTURE_1D );
		
		glBindTexture( GL_TEXTURE_1D, texName );

		glTexImage1D(	GL_TEXTURE_1D, 
				0, 
				GL_RGBA, 
				256, 
				0, 
				GL_RGBA, 
				GL_UNSIGNED_INT_8_8_8_8, 
				func 
				);

		M4D::CheckForGLError( "OGL building texture : " );

		delete [] func;
		return texName;
	}

	void
	initializeGL()
	{
		InitOpenGL();

		glClearColor(0,0,0,1);
		glShadeModel(GL_FLAT);
		glEnable(GL_DEPTH_TEST);
		//glEnable(GL_CULL_FACE);
		glEnable( GL_BLEND );
		glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
		glDepthFunc(GL_LEQUAL);

		_cgContext = cgCreateContext();
		CheckForCgError("creating context ", _cgContext );

		_brightnessContrastShaderConfig.Initialize( _cgContext, "LUT.cg", "LUT_texture" );
		//_transferFuncShaderConfig.Initialize( _cgContext, "SimpleTransferFunction.cg", "TransferFunctionShadingPreintegration" );
		_transferFuncShaderConfig.Initialize( _cgContext, "SimpleTransferFunction.cg", "TransferFunctionShading" );

		_transferFuncShaderConfig.transferFunctionTexture = CreateTransferFunction();
	}
	
	void
	resizeGL( int width, int height )
	{
		glViewport(0, 0, width, height);
		GLfloat x = (GLfloat)width / height;
		_camera.SetAspectRatio( x );
	}

	void	
	paintGL()
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		if( _region == NULL ) {
			return;
		}
		
		if( _volumeRendering ) {
			VolumeRender();
		} else {
			SliceRender();		
		}
	}

	void	
	VolumeRender()
	{

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();

		//Set viewing parameters
		M4D::SetViewAccordingToCamera( _camera );

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		
		
		Vector< float, 3> size = _region->GetRealSize();
		Vector< float, 3> minCoord = _region->GetRealMinimum();

		if( _texName == 0 ) {
			_texName = M4D::GLPrepareTextureFromImageData( *_region, _linearInterpolation );
		}
		//Texture coordinate generation
		M4D::SetVolumeTextureCoordinateGeneration( minCoord, size );
		_transferFuncShaderConfig.dataTexture = _texName;
		glBindTexture( GL_TEXTURE_1D, 0 );
		glBindTexture( GL_TEXTURE_2D, 0 );
		glBindTexture( GL_TEXTURE_3D, 0 );
		glDisable(GL_TEXTURE_3D);
		glDisable(GL_TEXTURE_2D);
		glDisable(GL_TEXTURE_1D);



		_transferFuncShaderConfig.eyePosition = _camera.GetEyePosition();
		_transferFuncShaderConfig.lightPosition = Vector< float, 3 > ( 3000.0f, 3000.0f, -3000.0f );
		//_transferFuncShaderConfig.sliceSpacing = 1.0;
		//_transferFuncShaderConfig.sliceNormal = _camera.GetCenterDirection();
		
		//-------------	RENDERING ------------------
		//Draw bounding box
		glColor3f( 1.0f, 0.0f, 0.0f );
		M4D::GLDrawBoundingBox( _region->GetRealMinimum(), _region->GetRealMaximum() );


		//Enable right shader
		_transferFuncShaderConfig.Enable();
		//Render volume
		M4D::GLDrawVolumeSlices( _bbox, _camera, 500, _cutPlane );
		//Disable shader
		_transferFuncShaderConfig.Disable();
		//---------- RENDERING FINISHED ------------

		M4D::DisableVolumeTextureCoordinateGeneration();
		M4D::CheckForGLError( "OGL error : " );
		glFlush();		
	}

	void	
	SliceRender()
	{

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		
		
		_viewConfiguration = GetOptimalViewConfiguration( 
					VectorPurgeDimension( _region->GetRealMinimum(), _plane ),
					VectorPurgeDimension( _region->GetRealMaximum(), _plane ),
					Vector< unsigned, 2 >( this->width(), this->height() ) ,
					ztFIT
					);

		M4D::SetToViewConfiguration2D( _viewConfiguration );
		
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		if( _texName == 0 ) {
			_texName = M4D::GLPrepareTextureFromImageData( *_region, _linearInterpolation );
		}
			
		glBindTexture( GL_TEXTURE_1D, 0 );
		glBindTexture( GL_TEXTURE_2D, 0 );
		glBindTexture( GL_TEXTURE_3D, 0 );
		glDisable(GL_TEXTURE_3D);
		glDisable(GL_TEXTURE_2D);
		glDisable(GL_TEXTURE_1D);

		_brightnessContrastShaderConfig.textureName = _texName;
		_brightnessContrastShaderConfig.brightnessContrast[1] = 1.0f;
		_brightnessContrastShaderConfig.brightnessContrast[0] = .0f;

		_brightnessContrastShaderConfig.Enable();

		CheckForCgError("Check befor drawing ", _cgContext );

		M4D::GLDrawVolumeSlice( 
				_region->GetRealMinimum(), 
				_region->GetRealMaximum(),
			        _sliceCoord,
				_plane	
				);
		
		_brightnessContrastShaderConfig.Disable();
		
		glFlush();
		
	}
protected:
	void	mouseMoveEvent ( QMouseEvent * event )
	{ 
		if( _mouseDown && _volumeRendering) {
			QPoint tmp = event->globalPos(); 
			int x = (tmp - _lastPoint).x();
			int y = (tmp - _lastPoint).y();
			_lastPoint = event->globalPos();
			_camera.YawAround( x * -0.05f );
			_camera.PitchAround( y * -0.05f );
			this->update();
		}
	}


	void	mousePressEvent ( QMouseEvent * event )
	{ 	
		if( event->button() == Qt::RightButton ) {
			_volumeRendering = !_volumeRendering;
			this->update();
		} else {
			_mouseDown = true; 
			_lastPoint = event->globalPos();
		}
	}

	void	mouseReleaseEvent ( QMouseEvent * event )
	{ _mouseDown = false; }

	void	wheelEvent ( QWheelEvent * event )
	{
		if( _volumeRendering ) {
			int numDegrees = event->delta() / 8;
			int numSteps = numDegrees / 15;

			if (event->orientation() == Qt::Horizontal) {
				_cutPlane += 0.05*numSteps;
			} else {
				_cutPlane -= 0.05*numSteps;
			}
			_cutPlane = Max( 0.0f, Min( 1.0f, _cutPlane ) );
		} else {
			int numDegrees = event->delta() / 8;
			int numSteps = numDegrees / 15;

			if (event->orientation() == Qt::Horizontal) {
				_sliceCoord += 0.02*numSteps;
			} else {
				_sliceCoord -= 0.02*numSteps;
			}
			_sliceCoord = Max( 0.0f, Min( 1.0f, _sliceCoord ) );
			std::cout << _sliceCoord << std::endl;
		}
		event->accept();
		this->update();
	}

	float 			_cutPlane;
	GLuint 			_texName;
	int			_slicePos;

	bool			_mouseDown;
	QPoint			_lastPoint;

	ARegion3D		*_region;
	M4D::BoundingBox3D	_bbox;
	Camera			_camera;
	bool			_linearInterpolation;

	ViewConfiguration2D	_viewConfiguration;
	CartesianPlanes		_plane;
	float32			_sliceCoord;
	float			_brightness;
	float			_contrast;
	bool			_volumeRendering;

	CGcontext   				_cgContext;
	CgTransferFunctionShadingShaderConfig 	_transferFuncShaderConfig;
	CgBrightnessContrastShaderConfig	_brightnessContrastShaderConfig;

	/*
	CgBrightnessContrastShaderConfig	_shaderConfig;
	CgMaskBlendShaderConfig			_blendShaderConfig;*/
};

#endif /*_SIMPLE_VOLUME_VIEWER_H*/

