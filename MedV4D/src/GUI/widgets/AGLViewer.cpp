#include <soglu/OGLDrawing.hpp>
#include <soglu/OGLTools.hpp>
//#include "MedV4D/GUI/utils/OGLDrawing.h"
#include "MedV4D/GUI/utils/QtM4DTools.h"
#include "MedV4D/Common/MathTools.h"
#include "MedV4D/GUI/widgets/AGLViewer.h"
#include "MedV4D/GUI/managers/ViewerManager.h"

namespace M4D
{
namespace GUI
{
namespace Viewer
{

struct AAAHELPER
{
	void operator()()
	{
		soglu::drawCylinder(glm::fvec3(), glm::fvec3(0.0f,0.0f,1.0f), 10, 50 );
		
		soglu::drawCylinder(glm::fvec3(200.0, 200.0, 100.0f), glm::fvec3(0.0f,0.0f,1.0f), 10, 50);
	}
};


AGLViewer::AGLViewer( QWidget *parent ): GLWidget( parent ), mSelected( false ), mLastMeasurement( 0 ), mEnableFPS( false )
{
	setMouseTracking ( true );
	setMinimumSize( 50, 50 );

	mFPSLabel = new QLabel( "0 FPS", this );
	mFPSLabel->setVisible( mEnableFPS );

}

AGLViewer::~AGLViewer()
{
	makeCurrent();
	mFrameBufferObject.Finalize();
	mPickManager.finalize();
	doneCurrent();

	deselect();
}

void
AGLViewer::getCurrentViewImageBuffer(size_t &aWidth, size_t &aHeight, boost::shared_array< uint8 > &aBuffer )
{
	makeCurrent();
	soglu::getImageBufferFromTexture(aWidth, aHeight, aBuffer, mFrameBufferObject.GetColorBuffer());	
	doneCurrent();
}

QImage
AGLViewer::getCurrentViewImage()
{
	size_t width = 0, height = 0;
	boost::shared_array< uint8 > buffer;

	getCurrentViewImageBuffer( width, height, buffer );
	QImage image( buffer.get(), width, height, QImage::Format_RGB888 );
	return image.mirrored(false,true);
}

void
AGLViewer::select()
{
	//TODO check if enabled
	mSelected = true;
	ViewerManager::getInstance()->selectViewer( this );
	update();
}

void
AGLViewer::deselect()
{
	mSelected = false;
	ViewerManager::getInstance()->deselectViewer( this );
	update();
}


void	
AGLViewer::initializeGL()
{
	soglu::initOpenGL();
	soglu::initializeCg();
	glClearColor( mViewerState->backgroundColor.redF(), mViewerState->backgroundColor.greenF(), mViewerState->backgroundColor.blueF(), mViewerState->backgroundColor.alphaF() );
	
	mFrameBufferObject.Initialize( width(), height() );

	mPickManager.initialize( 150 );D_PRINT("REMOVE THIS" );
	
	initializeRenderingEnvironment();
}

void	
AGLViewer::initializeOverlayGL()
{

}

void	
AGLViewer::paintGL()
{
	M4D::Common::Clock timer;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if ( preparedForRendering() ) {

		mFrameBufferObject.Bind();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		//****************************************
		prepareForRenderingStep();
		//updateGLViewSetupInfo();

		render();

		finalizeAfterRenderingStep();
		//****************************************

		//GLViewSetup  setup = getCurrentGLViewSetup();
		//mPickManager.render( Vector2i( tmpX,tmpY ), setup, AAAHELPER() );D_PRINT("REMOVE THIS" );//-----------------------------------------------------------------
		
		soglu::checkForGLError( "OGL error occured during rendering: " );
		
		mFrameBufferObject.Unbind();

		GL_CHECKED_CALL( glViewport(0, 0, width(), height()) );
		mFrameBufferObject.Render();
	} else {
		//D_PRINT( "Rendering not possible at the moment" );
	}
	
	
	if( mSelected ) {
		GL_CHECKED_CALL( glDisable(GL_DEPTH_TEST ) );
		GL_CHECKED_CALL( glDisable( GL_LIGHTING ) );
		//****************************************************	
		glClear( GL_DEPTH_BUFFER_BIT );
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho( 
			(double)0, 
			(double)width(), 
			(double)0, 
			(double)height(), 
			-1.0, 
			1.0
			);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glColor4f( 0.0f, 0.0f, 1.0f, 1.0f );
		soglu::drawRectangle(glm::fvec2(5.0f, 5.0f), glm::fvec2(width()-5.0f, height()-5.0f));
			/*glBegin( GL_LINE_LOOP );
				GLVertexVector( Vector2f( 5.0f, 5.0f ) );
				GLVertexVector( Vector2f( 5.0f, height()-5.0f ) );
				GLVertexVector( Vector2f( width()-5.0f, height()-5.0f ) );
				GLVertexVector( Vector2f( width()-5.0f, 5.0f ) );
			glEnd();*/
	}

	if( mEnableFPS ) {
		glFlush();
		mTimeMeasurements[mLastMeasurement++] = timer.secondsPassed();
		mLastMeasurement = mLastMeasurement % MEASUREMENT_SAMPLE_COUNT;
		double sum = 0.0;
		for( size_t i = 0; i < MEASUREMENT_SAMPLE_COUNT; ++i ) {
			sum += mTimeMeasurements[i];
		}
		sum /= double(MEASUREMENT_SAMPLE_COUNT);
		LOG( "FPS = " << 1.0 / sum );
	}
}

void	
AGLViewer::paintOverlayGL()
{
	/*glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	LOG( "Overlay" );
	if ( mSelected ) {
		glColor4f( 0.0f, 0.0f, 1.0f, 1.0f );
		glBegin( GL_LINE_LOOP );
			GLVertexVector( Vector2f( 5.0f, 5.0f ) );
			GLVertexVector( Vector2f( 5.0f, height()-5.0f ) );
			GLVertexVector( Vector2f( width()-5.0f, height()-5.0f ) );
			GLVertexVector( Vector2f( width()-5.0f, 5.0f ) );
		glEnd();
	}*/
}

void	
AGLViewer::resizeGL( int width, int height )
{
	glViewport(0, 0, width, height);
	mFrameBufferObject.Resize( width, height );

	mViewerState->mWindowSize[0] = static_cast< unsigned >( width );
	mViewerState->mWindowSize[1] = static_cast< unsigned >( height );
	mViewerState->aspectRatio = static_cast< float >(width) / height;
}

void	
AGLViewer::resizeOverlayGL( int width, int height )
{
	/*glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
	glViewport(0, 0, width, height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho( 
		(double)0, 
		(double)width, 
		(double)0, 
		(double)height, 
		-1.0, 
		1.0
		);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();*/
}

void	
AGLViewer::mouseMoveEvent ( QMouseEvent * event )
{ 
	if ( mViewerController && mViewerController->mouseMoveEvent( mViewerState, getMouseEventInfo( event ) ) ) {
		return;
	}
}

void	
AGLViewer::mouseDoubleClickEvent ( QMouseEvent * event )
{
	tmpX = event->x();
	tmpY = height() - event->y();
	update();
	ViewerManager::getInstance()->selectViewer( this );

	if ( mViewerController && mViewerController->mouseDoubleClickEvent( mViewerState, getMouseEventInfo( event ) ) ) {
		return;
	}
}

void	
AGLViewer::mousePressEvent ( QMouseEvent * event )
{ 	
	ViewerManager::getInstance()->selectViewer( this );

	if ( mViewerController && mViewerController->mousePressEvent( mViewerState,  getMouseEventInfo( event ) ) ) {
		return;
	}
}

void	
AGLViewer::mouseReleaseEvent ( QMouseEvent * event )
{ 
	ViewerManager::getInstance()->selectViewer( this );

	if ( mViewerController && mViewerController->mouseReleaseEvent( mViewerState, getMouseEventInfo( event ) ) ) {
		return;
	}
}

void	
AGLViewer::wheelEvent ( QWheelEvent * event )
{
	if ( mViewerController && mViewerController->wheelEvent( mViewerState, event ) ) {
		return;
	}
}

} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/

