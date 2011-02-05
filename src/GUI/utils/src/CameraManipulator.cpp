#include "GUI/utils/CameraManipulator.h"

CameraManipulator::CameraManipulator( Camera *aCamera )
	: mInteractionMode( imNONE ), mCamera( aCamera )
{

}

bool
CameraManipulator::mouseMoveEvent ( QSize aWinSize, QMouseEvent * event )
{
	ASSERT( mCamera != NULL );
	QPoint tmp = event->globalPos();
	switch ( mInteractionMode ) {
	case imORBIT_CAMERA: {
			QPoint diff = tmp - mLastPoint;   
			mCamera->YawAround( diff.x() * -0.05f );
			mCamera->PitchAround( diff.y() * -0.05f );
			break;
		}
	}
	mLastPoint = event->globalPos();
	return true;
}


//bool	
//CameraManipulator::mouseDoubleClickEvent ( QSize aWinSize, QMouseEvent * event );

bool
CameraManipulator::mousePressEvent ( QSize aWinSize, QMouseEvent * event )
{
	mClickPosition = event->globalPos();
	mLastPoint = mClickPosition;
	if( event->button() == Qt::LeftButton ) {
		mInteractionMode = imORBIT_CAMERA;
		return true;
	}
}

bool
CameraManipulator::mouseReleaseEvent ( QSize aWinSize, QMouseEvent * event )
{
	mInteractionMode = imNONE;
	return true;
}

bool
CameraManipulator::wheelEvent ( QSize aWinSize, QWheelEvent * event )
{
	int numDegrees = event->delta() / 8;
	int numSteps = numDegrees / 15;
	float dollyRatio = 1.1f;
	if ( event->delta() > 0 ) {
		dollyRatio = 1.0f/dollyRatio;
	}
	DollyCamera( *mCamera, dollyRatio );
	event->accept();
	return true;
}


