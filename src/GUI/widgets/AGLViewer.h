#ifndef A_GL_VIEWER_H
#define A_GL_VIEWER_H

//#include "GUI/widgets/AViewer.h"
#include "GUI/utils/FrameBufferObject.h"
#include <QtGui>
#include <QtOpenGL>
#include <boost/shared_ptr.hpp>
#include <boost/cast.hpp>

namespace M4D
{
namespace GUI
{
namespace Viewer
{

enum ViewType
{
	vt2DAlignedSlices	= 1,
	vt2DGeneralSlices	= 1 << 1,
	vt3D			= 1 << 2
};

class BaseViewerState
{
public:
	typedef boost::shared_ptr< BaseViewerState > Ptr;
	virtual ~BaseViewerState(){}

	Vector2u	mWindowSize;
	float		aspectRatio;

	QWidget		*viewerWindow;

	QColor		backgroundColor;

	unsigned	availableViewTypes;
	ViewType	viewType;

	template< typename TViewerType >
	TViewerType &
	getViewerWindow()
	{
		return *boost::polymorphic_cast< TViewerType *>( viewerWindow );
	}
};

class AViewerController
{
public:
	typedef boost::shared_ptr< AViewerController > Ptr;

	virtual bool
	mouseMoveEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event ) = 0;

	virtual bool	
	mouseDoubleClickEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event ) = 0;

	virtual bool
	mousePressEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event ) = 0;

	virtual bool
	mouseReleaseEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event ) = 0;

	virtual bool
	wheelEvent ( BaseViewerState::Ptr aViewerState, QWheelEvent * event ) = 0;
};


class AGLViewer: public QGLWidget
{
	Q_OBJECT;
public:
	AGLViewer( QWidget *parent ): QGLWidget( parent )
	{
		setMouseTracking ( true );
	}

	void
	setViewerController( AViewerController::Ptr aController )
	{
		//TODO check
		mViewerController = aController;
	}
protected:
//**************************************************************
	virtual void
	initializeRenderingEnvironment() = 0;

	virtual bool
	preparedForRendering() = 0;

	virtual void
	prepareForRenderingStep() = 0;

	virtual void
	render() = 0;

	virtual void
	finalizeAfterRenderingStep() = 0;

//**************************************************************

	void	
	initializeGL ();

	void	
	initializeOverlayGL ();

	void	
	paintGL ();

	void	
	paintOverlayGL ();

	void	
	resizeGL ( int width, int height );

	void	
	resizeOverlayGL ( int width, int height );

	void
	mouseMoveEvent ( QMouseEvent * event );

	void	
	mouseDoubleClickEvent ( QMouseEvent * event );

	void
	mousePressEvent ( QMouseEvent * event );

	void
	mouseReleaseEvent ( QMouseEvent * event );

	void
	wheelEvent ( QWheelEvent * event );


	
	FrameBufferObject	mFrameBufferObject;
	BaseViewerState::Ptr	mViewerState;
	AViewerController::Ptr	mViewerController;
};


} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/

#endif /*A_GL_VIEWER_H*/

