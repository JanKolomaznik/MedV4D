#ifdef USE_CG

#include "GUI/utils/OGLDrawing.h"
#include "GUI/utils/QtM4DTools.h"
#include "common/MathTools.h"
#include "GUI/widgets/AGLViewer.h"
#include "GUI/utils/ViewerManager.h"

namespace M4D
{
namespace GUI
{
namespace Viewer
{


AGLViewer::AGLViewer( QWidget *parent ): GLWidget( parent ), mSelected( false )
{
	setMouseTracking ( true );
	setMinimumSize( 50, 50 );
}

AGLViewer::~AGLViewer()
{
	makeCurrent();
	mFrameBufferObject.Finalize();
	doneCurrent();

	deselect();
}

void
AGLViewer::getCurrentViewImageBuffer( uint32 &aWidth, uint32 &aHeight, boost::shared_array< uint8 > &aBuffer )
{
	makeCurrent();
	getImageBufferFromTexture( aWidth, aHeight, aBuffer, mFrameBufferObject.GetColorBuffer() );	
	doneCurrent();
}

QImage
AGLViewer::getCurrentViewImage()
{
	uint32 width = 0, height = 0;
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
	InitOpenGL();
	glClearColor( mViewerState->backgroundColor.redF(), mViewerState->backgroundColor.greenF(), mViewerState->backgroundColor.blueF(), mViewerState->backgroundColor.alphaF() );
	
	mFrameBufferObject.Initialize( width(), height() );

	initializeRenderingEnvironment();
}

void	
AGLViewer::initializeOverlayGL()
{

}

void	
AGLViewer::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if ( preparedForRendering() ) {

		mFrameBufferObject.Bind();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		//****************************************
		prepareForRenderingStep();
		updateGLViewSetupInfo();

		render();

		finalizeAfterRenderingStep();
		//****************************************

		M4D::CheckForGLError( "OGL error occured during rendering: " );
		
		mFrameBufferObject.Unbind();

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
			glBegin( GL_LINE_LOOP );
				GLVertexVector( Vector2f( 5.0f, 5.0f ) );
				GLVertexVector( Vector2f( 5.0f, height()-5.0f ) );
				GLVertexVector( Vector2f( width()-5.0f, height()-5.0f ) );
				GLVertexVector( Vector2f( width()-5.0f, 5.0f ) );
			glEnd();
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


#endif /*USE_CG*/
