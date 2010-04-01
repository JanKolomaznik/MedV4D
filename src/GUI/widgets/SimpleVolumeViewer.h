#ifndef _SIMPLE_VOLUME_VIEWER_H
#define _SIMPLE_VOLUME_VIEWER_H

#include <QtGui/QWidget>
#include "Imaging/ImageRegion.h"
#include "GUI/utils/ViewConfiguration.h"
#include "GUI/utils/OGLDrawing.h"
#include "common/Vector.h"
#include "common/MathTools.h"

#include "GUI/utils/CgShaderTools.h"
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
		: SupportGLWidget( parent ), _region( NULL ), _linearInterpolation( false ),
		_camera( Vector<float,3>( 0.0f, 0.0f, -1200.0f ), Vector<float,3>( 0.0f, 0.0f, 0.0f ) )
	{
		_cutPlane = 1.0f;
		_mouseDown = false;
		_slicePos = 1900;
		_texName = 0;
		_camera.SetFieldOfView( 10.0f );
	}

	~SimpleVolumeViewer()
	{
		_transferFuncShaderConfig.Finalize();
		cgDestroyContext(_cgContext);
		glDeleteTextures( 1, &_texName );
	}

	void
	SetImageRegion( ARegion3D *region )
		{ 
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


		for( unsigned i=0; i < 256; ++i ) {
			if( i < 8 ) {
				func[i] = 0;
			} else {
				func[i] = Min(i+100, (unsigned) 255)<<24 | (i/2) << 16 | 45; //xFF0000FF;
			}
		}

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
		glClearColor(0,0,0,1);
		glShadeModel(GL_FLAT);
		glEnable(GL_DEPTH_TEST);
		//glEnable(GL_CULL_FACE);
		glEnable( GL_BLEND );
		glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
		glDepthFunc(GL_LEQUAL);

		_cgContext = cgCreateContext();
		CheckForCgError("creating context ", _cgContext );

		_transferFuncShaderConfig.Initialize( _cgContext, "SimpleTransferFunction.cg", "SimpleTransferFunction" );

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

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();

		//Set viewing parameters
		M4D::SetViewAccordingToCamera( _camera );

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		
		if( _region == NULL ) {
			return;
		}
		Vector< float, 3> size = _region->GetRealSize();
		Vector< float, 3> minCoord = _region->GetRealMinimum();

		if( _texName == 0 ) {
			_texName = M4D::GLPrepareTextureFromImageData( *_region, _linearInterpolation );
			_transferFuncShaderConfig.dataTexture = _texName;
			//Texture coordinate generation
			M4D::SetVolumeTextureCoordinateGeneration( minCoord, size );
		}
		glBindTexture( GL_TEXTURE_1D, 0 );
		glBindTexture( GL_TEXTURE_2D, 0 );
		glBindTexture( GL_TEXTURE_3D, 0 );
		glDisable(GL_TEXTURE_3D);
		glDisable(GL_TEXTURE_2D);
		glDisable(GL_TEXTURE_1D);



		
		//-------------	RENDERING ------------------
		//Draw bounding box
		glColor3f( 1.0f, 0.0f, 0.0f );
		M4D::GLDrawBoundingBox( _region->GetRealMinimum(), _region->GetRealMaximum() );


		//Enable right shader
		_transferFuncShaderConfig.Enable();
		//Render volume
		M4D::GLDrawVolumeSlices( _bbox, _camera, 300, _cutPlane );
		//Disable shader
		_transferFuncShaderConfig.Disable();
		//---------- RENDERING FINISHED ------------


		M4D::CheckForGLError( "OGL error : " );
		glFlush();		
	}
protected:
	void	mouseMoveEvent ( QMouseEvent * event )
	{ 
		if( _mouseDown ) {
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
	{ 	_mouseDown = true; 
		_lastPoint = event->globalPos();
	}

	void	mouseReleaseEvent ( QMouseEvent * event )
	{ _mouseDown = false; }

	void	wheelEvent ( QWheelEvent * event )
	{
		int numDegrees = event->delta() / 8;
		int numSteps = numDegrees / 15;

		if (event->orientation() == Qt::Horizontal) {
			_cutPlane += 0.05*numSteps;
		} else {
			_cutPlane -= 0.05*numSteps;
		}
		_cutPlane = Max( 0.0f, Min( 1.0f, _cutPlane ) );
		event->accept();
		this->update();
	}

	float 			_cutPlane;
	GLuint 			_texName;
	int			_slicePos;

	bool			_mouseDown;
	QPoint			_lastPoint;

	ARegion3D		*_region;
	ARegion2D		*_region2D;
	M4D::BoundingBox3D	_bbox;
	Camera			_camera;
	bool			_linearInterpolation;

	CGcontext   				_cgContext;
	CgSimpleTransferFunctionShaderConfig 	_transferFuncShaderConfig;
	/*
	CgBrightnessContrastShaderConfig	_shaderConfig;
	CgMaskBlendShaderConfig			_blendShaderConfig;*/
};

#endif /*_SIMPLE_VOLUME_VIEWER_H*/

