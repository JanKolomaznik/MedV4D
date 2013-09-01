#ifndef A_GL_VIEWER_H
#define A_GL_VIEWER_H
//Temporary workaround
#ifndef Q_MOC_RUN
#include "MedV4D/GUI/widgets/GLWidget.h"
#include <QtWidgets>
#include "soglu/FrameBufferObject.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <boost/cast.hpp>

//#include "MedV4D/GUI/utils/OGLSelection.h"
#include "MedV4D/GUI/utils/AViewerController.h"
#endif
namespace M4D
{
namespace GUI
{
namespace Viewer
{

struct MouseEventInfo
{
	MouseEventInfo( const soglu::GLViewSetup &aViewSetup, QMouseEvent *aEvent, ViewType aViewType ): viewSetup( aViewSetup ), event( aEvent ), viewType( aViewType )
		{ }

	MouseEventInfo( const soglu::GLViewSetup &aViewSetup, QMouseEvent *aEvent, ViewType aViewType, glm::fvec3 aPos ): viewSetup( aViewSetup ), event( aEvent ), viewType( aViewType ), realCoordinates( aPos )
		{ }

	//MouseEventInfo( const soglu::GLViewSetup &aViewSetup, QMouseEvent *aEvent, ViewType aViewType, Point3Df aPoint, glm::fvec3 aDirection ): viewSetup( aViewSetup ), event( aEvent ), viewType( aViewType ), point( aPoint ), direction( aDirection )
	//	{ }

	MouseEventInfo( const soglu::GLViewSetup &aViewSetup, QMouseEvent *aEvent, ViewType aViewType, glm::fvec3 aPoint, glm::fvec3 aDirection ): viewSetup( aViewSetup ), event( aEvent ), viewType( aViewType ), point( aPoint ), direction( aDirection )
		{ }

	soglu::GLViewSetup viewSetup;

	QMouseEvent *event;
	ViewType viewType;

	//2D section
	glm::fvec3 realCoordinates;

	//3D section

	glm::fvec3 point;
	glm::fvec3 direction;
	//HalfAxis axis;

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
	const soglu::GLViewSetup &
	getCurrentGLViewSetup()const
	{
		return mViewerState->glViewSetup;
	}

	void
	getCurrentViewImageBuffer(size_t &aWidth, size_t &aHeight, boost::shared_array< uint8_t > &aBuffer );

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


	void
	toggleFPS()
	{
		enableFPS( !mEnableFPS );
	}

	void
	enableFPS( bool aEnable )
	{
		mEnableFPS = aEnable;
		mFPSLabel->setVisible( mEnableFPS );
		update();
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
		soglu::getCurrentGLSetup( mViewerState->glViewSetup );
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


	soglu::FrameBufferObject	mFrameBufferObject;
	BaseViewerState::Ptr	mViewerState;
	AViewerController::Ptr	mViewerController;

	//GLViewSetup mGLViewSetup;

	bool mSelected;

	static const size_t MEASUREMENT_SAMPLE_COUNT = 10;
	double mTimeMeasurements[MEASUREMENT_SAMPLE_COUNT];
	size_t mLastMeasurement;
	bool mEnableFPS;
	QLabel *mFPSLabel;



	//M4D::PickManager mPickManager;
	int tmpX,tmpY;
};


} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/

#endif /*A_GL_VIEWER_H*/

