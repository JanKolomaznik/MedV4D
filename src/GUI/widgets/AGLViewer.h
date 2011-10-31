#ifndef A_GL_VIEWER_H
#define A_GL_VIEWER_H

#include "GUI/widgets/GLWidget.h"
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

struct MouseEventInfo
{
	MouseEventInfo( QMouseEvent *aEvent, ViewType aViewType ): event( aEvent ), viewType( aViewType )
		{ }

	MouseEventInfo( QMouseEvent *aEvent, ViewType aViewType, Vector3f aPos ): event( aEvent ), viewType( aViewType ), realCoordinates( aPos )
		{ }

	MouseEventInfo( QMouseEvent *aEvent, ViewType aViewType, Point3Df aPoint, Vector3f aDirection ): event( aEvent ), viewType( aViewType ), point( aPoint ), direction( aDirection )
		{ }

	MouseEventInfo( QMouseEvent *aEvent, ViewType aViewType, Vector3f aPoint, Vector3f aDirection ): event( aEvent ), viewType( aViewType ), point( aPoint ), direction( aDirection )
		{ }

	QMouseEvent *event;
	ViewType viewType;

	//2D section
	Vector3f realCoordinates;

	//3D section
	
	Point3Df point; 
	Vector3f direction;
	//HalfAxis axis;

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

class AViewerController: public QObject
{
public:
	typedef boost::shared_ptr< AViewerController > Ptr;

	virtual bool
	mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const MouseEventInfo &aEventInfo ) = 0;

	virtual bool	
	mouseDoubleClickEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const MouseEventInfo &aEventInfo ) = 0;

	virtual bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const MouseEventInfo &aEventInfo ) = 0;

	virtual bool
	mouseReleaseEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const MouseEventInfo &aEventInfo ) = 0;

	virtual bool
	wheelEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QWheelEvent * event ) = 0;
};


class AGLViewer: public GLWidget
{
	Q_OBJECT;
public:
	AGLViewer( QWidget *parent );

	~AGLViewer();

	void
	setViewerController( AViewerController::Ptr aController )
	{
		//TODO check
		mViewerController = aController;
	}
	const GLViewSetup &
	getCurrentGLViewSetup()const
	{
		return mGLViewSetup;
	}

	void
	getCurrentViewImageBuffer( uint32 &aWidth, uint32 &aHeight, boost::shared_array< uint8 > &aBuffer );

	QImage
	getCurrentViewImage();

public slots:
	void
	select();
	void
	deselect();

	void
	setBackgroundColor( QColor aColor )
	{
		mViewerState->backgroundColor = aColor;
	}
signals:
	void
	viewerSelected();

	void
	MouseInfoUpdate( const QString &aInfo );
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

	virtual MouseEventInfo 
	getMouseEventInfo( QMouseEvent * event ) = 0;

//**************************************************************
	void
	updateGLViewSetupInfo()
	{
		getCurrentGLSetup( mGLViewSetup );
	}

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

	GLViewSetup mGLViewSetup;

	bool mSelected;

	QLabel *mFPSLabel;
};


} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/

#endif /*A_GL_VIEWER_H*/

