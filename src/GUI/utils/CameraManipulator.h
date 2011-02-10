#ifndef CAMERA_MANIPULATOR_H
#define CAMERA_MANIPULATOR_H

#include "GUI/utils/IUserEvents.h"
#include "GUI/utils/Camera.h"
#include "GUI/utils/ViewConfiguration.h"

/*class IntCheckedSignalTransition : public QSignalTransition
{
public:
	IntCheckedSignalTransition(QObject * sender, const char * signal, int value, QState * sourceState = 0)
		: QSignalTransition(sender, signal, sourceState ), mValue( value )
	{}
protected:
	bool eventTest(QEvent *e) {
		if (!QSignalTransition::eventTest(e))
			return false;
		QStateMachine::SignalEvent *se = static_cast<QStateMachine::SignalEvent*>(e);
		return (se->arguments().at(0).toInt() == mValue);
	}
	int mValue;
};*/


class CameraManipulator: public IUserEvents
{
public:
	enum InteractionMode { 
		imNONE,
		imORBIT_CAMERA 
	 };
	CameraManipulator( Camera *aCamera = NULL );

	void
	SetCamera( Camera *aCamera )
	{
		mCamera = aCamera;
	}

	bool
	mouseMoveEvent ( QSize aWinSize, QMouseEvent * event );

	//bool	
	//mouseDoubleClickEvent ( QSize aWinSize, QMouseEvent * event );

	bool
	mousePressEvent ( QSize aWinSize, QMouseEvent * event );

	bool
	mouseReleaseEvent ( QSize aWinSize, QMouseEvent * event );

	bool
	wheelEvent ( QSize aWinSize, QWheelEvent * event );

protected:
	InteractionMode mInteractionMode;
	QPoint					mClickPosition;
	QPoint					mLastPoint;
	Camera 					*mCamera;
};

class ViewConfiguration2DManipulator: public IUserEvents
{
public:
	enum InteractionMode { 
		imNONE,
		imORBIT_CAMERA 
	 };

	ViewConfiguration2DManipulator( ViewConfiguration2D *aViewConfig = NULL )
		: mViewConfig( aViewConfig )
	{ }

	void
	SetViewConfiguration2D( ViewConfiguration2D *aViewConfig )
	{
		mViewConfig = aViewConfig;
	}

	/*bool
	mouseMoveEvent ( QSize aWinSize, QMouseEvent * event );

	//bool	
	//mouseDoubleClickEvent ( QSize aWinSize, QMouseEvent * event );

	bool
	mousePressEvent ( QSize aWinSize, QMouseEvent * event );

	bool
	mouseReleaseEvent ( QSize aWinSize, QMouseEvent * event );

	bool
	wheelEvent ( QSize aWinSize, QWheelEvent * event );*/

protected:
	InteractionMode mInteractionMode;
	QPoint					mClickPosition;
	QPoint					mLastPoint;
	ViewConfiguration2D 			*mViewConfig;
};

class SliceViewConfigManipulator: public ViewConfiguration2DManipulator
{
public:
	enum InteractionMode { 
		imNONE,
		imORBIT_CAMERA 
	 };

	SliceViewConfigManipulator( ViewConfiguration2D *aViewConfig = NULL, Vector3i *aCurrentSlice = NULL, CartesianPlanes *aPlane = NULL )
		: ViewConfiguration2DManipulator( aViewConfig ), mCurrentSlice( aCurrentSlice ), mPlane( aPlane )
	{ }

	

	/*bool
	mouseMoveEvent ( QSize aWinSize, QMouseEvent * event );*/

	bool	
	mouseDoubleClickEvent ( QSize aWinSize, QMouseEvent * event )
	{
		if( event->button() == Qt::LeftButton ) {
			ASSERT( mPlane );
			*mPlane = NextCartesianPlane( *mPlane );
			return true;
		}
		return false;
	}

	/*bool
	mousePressEvent ( QSize aWinSize, QMouseEvent * event );

	bool
	mouseReleaseEvent ( QSize aWinSize, QMouseEvent * event );*/

	bool
	wheelEvent ( QSize aWinSize, QWheelEvent * event )
	{
		ASSERT( mCurrentSlice );

		int numDegrees = event->delta() / 8;
		int numSteps = numDegrees / 15;
		(*mCurrentSlice)[ *mPlane ] += numSteps;
		return true;
	}

protected:
	InteractionMode mInteractionMode;
	QPoint					mClickPosition;
	QPoint					mLastPoint;
	Vector3i				*mCurrentSlice;
	CartesianPlanes				*mPlane;
};


#endif /*CAMERA_MANIPULATOR_H*/
