#ifndef _SAMPLE_3D_VIEWER_H
#define _SAMPLE_3D_VIEWER_H

#include <QtGui/QWidget>
#include "GUI/utils/ViewConfiguration.h"
#include "GUI/utils/OGLDrawing.h"
#include "common/Vector.h"
#include "common/MathTools.h"
#include "Imaging/Mesh.h"

#include "GUI/utils/DrawingTools.h"

#include <QtGui/QMouseEvent>
#include <QtGui/QWheelEvent>
#include <QtCore/QPoint>


extern M4D::Imaging::Geometry::SimpleMesh *
LoadMonkey();

template< typename SupportGLWidget >
class Sample3DViewer: public SupportGLWidget
{
public:
	Sample3DViewer( QWidget *parent = NULL )
		: SupportGLWidget( parent ), _camera( Vector<float,3>( 5.0f, 5.0f, 10.0f ), Vector<float,3>( 0.0f, 0.0f, 0.0f ) ), _mouseDown( false )
	{
		GetGlExtFunctions();
		_camera.SetFieldOfView( 20.0f );
		_mesh = LoadMonkey();
	}

	~Sample3DViewer()
	{
		delete _mesh;
	}


	void
	initializeGL()
	{
		glClearColor(0,0.5f,0.6,1);
		glShadeModel(GL_FLAT);
		glEnable(GL_DEPTH_TEST);
		//glEnable(GL_CULL_FACE);
		//glEnable( GL_BLEND );
		//glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
		//glDepthFunc(GL_LEQUAL);

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
		/*static const GLfloat ctrlpoints[4][4][3] = {
		   { {-1.5, -1.5, 4.0},
		     {-0.5, -1.5, 2.0},
		     {0.5, -1.5, -1.0},
		     {1.5, -1.5, 2.0}},
		   { {-1.5, -0.5, 1.0},
		     {-0.5, -0.5, 3.0},
		     {0.5, -0.5, 0.0},
		     {1.5, -0.5, -1.0}},
		   { {-1.5, 0.5, 4.0},
		     {-0.5, 0.5, 0.0},
		     {0.5, 0.5, 3.0},
		     {1.5, 0.5, 4.0}},
		   { {-1.5, 1.5, -2.0},
		     {-0.5, 1.5, -2.0},
		     {0.5, 1.5, 0.0},
		     {1.5, 1.5, -1.0}}
		};
		glMap2f(GL_MAP2_VERTEX_3, 0, 1, 3, 4, 0, 1, 12, 4, &ctrlpoints[0][0][0]);
		glEnable(GL_MAP2_VERTEX_3);
		glEnable(GL_AUTO_NORMAL);
		glMapGrid2f(20, 0.0, 1.0, 20, 0.0, 1.0);

		glEvalMesh2(GL_FILL, 0, 20, 0, 20);
		glColor3f( 1.0f, 1.0f, 0.0f );
		glEvalMesh2(GL_LINE, 0, 20, 0, 20);*/

		/*glBegin( GL_POINTS );
			//glVertex3f( 0.0f, 0.0f, 0.0f );
			for( unsigned i = 0; i < VERTEX_COUNT; ++i ) {
				glVertex3fv( coords[i] );
			}
		glEnd();*/

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		

		//Set viewing parameters
		M4D::SetViewAccordingToCamera( _camera );
		
		//-------------	RENDERING ------------------
		glColor3f( 0.5f, 0.5f, 0.5f );
		glPointSize( 5.0f );

		glPolygonOffset( 1.0, 1.0 );
		glEnable(GL_POLYGON_OFFSET_FILL);

		glPolygonMode( GL_FRONT_AND_BACK, GL_FILL);
		glBegin( GL_TRIANGLES );
			M4D::GLDrawMesh( *_mesh );
		glEnd();
		glDisable(GL_POLYGON_OFFSET_FILL);

		
		glColor3f( 0.0f, 0.0f, 0.0f );
		glPolygonMode( GL_FRONT_AND_BACK, GL_LINE);
		glBegin( GL_TRIANGLES );
			M4D::GLDrawMesh( *_mesh );
		glEnd();

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
		/*int numDegrees = event->delta() / 8;
		int numSteps = numDegrees / 15;

		if (event->orientation() == Qt::Horizontal) {
			_cutPlane += 0.05*numSteps;
		} else {
			_cutPlane -= 0.05*numSteps;
		}
		_cutPlane = Max( 0.0f, Min( 1.0f, _cutPlane ) );
		event->accept();
		this->update();*/
	}

	

	QPoint					_lastPoint;

	M4D::BoundingBox3D			_bbox;
	Camera					_camera;
	bool					_mouseDown;
	M4D::Imaging::Geometry::SimpleMesh 	*_mesh;
};

#endif /*_SAMPLE_3D_VIEWER_H*/
