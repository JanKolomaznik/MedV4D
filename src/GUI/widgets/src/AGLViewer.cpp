#ifdef USE_CG

#include "GUI/utils/OGLDrawing.h"
#include "GUI/utils/QtM4DTools.h"
#include "common/MathTools.h"
#include "GUI/widgets/AGLViewer.h"

namespace M4D
{
namespace GUI
{
namespace Viewer
{


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

	if ( !preparedForRendering() ) {
		D_PRINT( "Rendering not possible at the moment" );
		return;
	}

	mFrameBufferObject.Bind();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	//****************************************
	prepareForRenderingStep();

	render();

	finalizeAfterRenderingStep();
	//****************************************

	M4D::CheckForGLError( "OGL error occured during rendering: " );
	
	mFrameBufferObject.Unbind();

	mFrameBufferObject.Render();	

}

void	
AGLViewer::paintOverlayGL()
{

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
	glViewport(0, 0, width, height);
}

void	
AGLViewer::mouseMoveEvent ( QMouseEvent * event )
{ 
	if ( mViewerController && mViewerController->mouseMoveEvent( mViewerState, event ) ) {
		return;
	}
}

void	
AGLViewer::mouseDoubleClickEvent ( QMouseEvent * event )
{
	if ( mViewerController && mViewerController->mouseDoubleClickEvent( mViewerState, event ) ) {
		return;
	}
}

void	
AGLViewer::mousePressEvent ( QMouseEvent * event )
{ 	
	if ( mViewerController && mViewerController->mousePressEvent( mViewerState, event ) ) {
		return;
	}
}

void	
AGLViewer::mouseReleaseEvent ( QMouseEvent * event )
{ 
	if ( mViewerController && mViewerController->mouseReleaseEvent( mViewerState, event ) ) {
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
