#include "MedV4D/GUI/utils/ViewerController.h"
#include "MedV4D/GUI/widgets/GeneralViewer.h"

namespace M4D
{
namespace GUI
{
namespace Viewer
{

ViewerController::ViewerController()
{
	mCameraOrbitButton = Qt::MidButton;
	mLUTSetMouseButton = Qt::RightButton;
	mFastSliceChangeMouseButton = Qt::MidButton;
	mCutPlaneOffsetButton = Qt::RightButton;

	mCutPlaneKeyboardModifiers = Qt::ControlModifier | Qt::MetaModifier;

	mTimer.setSingleShot( false );
	QObject::connect( &mTimer, SIGNAL(timeout()), this, SLOT( timerCall() ) );
}

bool
ViewerController::mouseMoveEvent ( BaseViewerState::Ptr aViewerState, const MouseEventInfo &aEventInfo )
{
	ViewerState &state = *(boost::polymorphic_downcast< ViewerState *>( aViewerState.get() ) );

	QPoint diff = mTrackInfo.trackUpdate( aEventInfo.event->pos(), aEventInfo.event->globalPos() );
	if ( state.viewType == vt3D && mInteractionMode == imORBIT_CAMERA ) {
		state.getViewerWindow< GeneralViewer >().cameraOrbit( Vector2f( diff.x() * -0.2f, diff.y() * -0.2f ) );
		return true;
	}
	/*
	// TODO - handle cut planes
	if ( state.viewType == vt3D && mInteractionMode == imCUT_PLANE ) {
		state.getViewerWindow< GeneralViewer >().cameraOrbit( Vector2f( diff.x() * -0.2f, diff.y() * -0.2f ) );
		glm::fvec3 dir = -1.0f*state.getViewerWindow< GeneralViewer >().getCameraTargetDirection();
		glm::fvec3 pos = state.getViewerWindow< GeneralViewer >().getCameraTargetPosition() + dir * state.mVolumeRenderConfig.cutPlaneCameraTargetOffset;
		state.getViewerWindow< GeneralViewer >().setCutPlane( soglu::Planef( pos, dir ) );
		return true;
	}*/
	if ( mInteractionMode == imLUT_SETTING ) {
		glm::fvec2 oldVal = state.getViewerWindow< GeneralViewer >().getLUTWindow();
		state.getViewerWindow< GeneralViewer >().setLUTWindow( oldVal + glm::fvec2(diff.x(), diff.y()));
		return true;
	}
	/*
	// TODO - handle cut planes
	if ( mInteractionMode == imCUT_PLANE_OFFSET ) {
		float oldVal = state.mVolumeRenderConfig.cutPlaneCameraTargetOffset;
		state.getViewerWindow< GeneralViewer >().setCutPlaneCameraTargetOffset( oldVal + 0.3*(diff.x() + diff.y()) );
		return true;
	}*/
	if ( mInteractionMode == imFAST_SLICE_CHANGE ) {
		int speed = mTrackInfo.mStartLocalPosition.y() - aEventInfo.event->pos().y();
		mTmpViewer = &(state.getViewerWindow< GeneralViewer >());
		mPositive = speed > 0;
		if( speed != 0 ) {
			float ms = 1000.0f / abs(speed);
			mTimer.setInterval( max<int>( static_cast<int>( ms ), 10 ) );
			mTimer.start();
			timerCall();
		} else {
			mTimer.stop();
		}
		return true;
	}

	if ( state.viewType == vt2DAlignedSlices ) {
		state.getViewerWindow< GeneralViewer >().updateMouseInfo( fromGLM(aEventInfo.point) );
	}
	return false;
}

bool
ViewerController::mouseDoubleClickEvent ( BaseViewerState::Ptr aViewerState, const MouseEventInfo &aEventInfo )
{
	ViewerState &state = *(boost::polymorphic_downcast< ViewerState *>( aViewerState.get() ) );
	if ( state.viewType == vt2DAlignedSlices ) {
		state.getViewerWindow< GeneralViewer >().switchToNextPlane();

		aEventInfo.event->accept();
		return true;
	}

	return false;
}

bool
ViewerController::mousePressEvent ( BaseViewerState::Ptr aViewerState, const MouseEventInfo &aEventInfo )
{
	ViewerState &state = *(boost::polymorphic_downcast< ViewerState *>( aViewerState.get() ) );

	mTrackInfo.startTracking( aEventInfo.event->pos(), aEventInfo.event->globalPos() );
	if ( state.viewType == vt3D ) {
		if( aEventInfo.event->button() == mCameraOrbitButton ) {
			//LOG( aEventInfo.event->modifiers() );
			if ( aEventInfo.event->modifiers() & mCutPlaneKeyboardModifiers /*&& state.mVolumeRenderConfig.enableCutPlane*/ ) {
				mInteractionMode = imCUT_PLANE;
				// TODO - handle cut planes
				/*glm::fvec3 dir = -1.0f*state.getViewerWindow< GeneralViewer >().getCameraTargetDirection();
				glm::fvec3 pos = state.getViewerWindow< GeneralViewer >().getCameraTargetPosition() + dir * state.mVolumeRenderConfig.cutPlaneCameraTargetOffset;
				state.getViewerWindow< GeneralViewer >().setCutPlane( soglu::Planef( pos, dir ) );
				state.getViewerWindow< GeneralViewer >().enableCutPlane( true );
				*/
				//LOG( "ENABLING CUTPLANE" );
			} else {
				mInteractionMode = imORBIT_CAMERA;
			}
			return true;
		}
		if( (aEventInfo.event->modifiers() & mCutPlaneKeyboardModifiers) && (aEventInfo.event->button() == mCutPlaneOffsetButton) ) {
			mInteractionMode = imCUT_PLANE_OFFSET;
			return true;
		}
	}
	if ( state.viewType == vt2DAlignedSlices ) {
		if( aEventInfo.event->button() == mFastSliceChangeMouseButton ) {
			mInteractionMode = imFAST_SLICE_CHANGE;
			return true;
		}
	}
	if ( state.colorTransform == M4D::GUI::Renderer::ctLUTWindow
		|| state.colorTransform == M4D::GUI::Renderer::ctMaxIntensityProjection
		|| state.colorTransform == M4D::GUI::Renderer::ctBasic 
		|| state.colorTransform == M4D::GUI::Renderer::ctEigenvalues 
		|| state.colorTransform == M4D::GUI::Renderer::ctEigenvaluesRaw)
	{
		if( aEventInfo.event->button() == mLUTSetMouseButton ) {
			mInteractionMode = imLUT_SETTING;
			return true;
		}
	}

	return false;
}

bool
ViewerController::mouseReleaseEvent ( BaseViewerState::Ptr aViewerState, const MouseEventInfo &aEventInfo )
{
	//ViewerState &state = *(boost::polymorphic_downcast< ViewerState *>( aViewerState.get() ) );
	if ( (mInteractionMode == imORBIT_CAMERA && aEventInfo.event->button() == mCameraOrbitButton)
	  || (mInteractionMode == imCUT_PLANE && aEventInfo.event->button() == mCameraOrbitButton)
	  || (mInteractionMode == imCUT_PLANE_OFFSET && aEventInfo.event->button() == mCutPlaneOffsetButton)
	  || (mInteractionMode == imLUT_SETTING && aEventInfo.event->button() == mLUTSetMouseButton) )
	{
		mInteractionMode = imNONE;
		return true;
	}
	if ( mInteractionMode == imFAST_SLICE_CHANGE && aEventInfo.event->button() == mFastSliceChangeMouseButton ) {
		mTimer.stop();
		mInteractionMode = imNONE;
		return true;
	}
	return false;
}

bool
ViewerController::wheelEvent ( BaseViewerState::Ptr aViewerState, QWheelEvent * event )
{
	ViewerState &state = *(boost::polymorphic_downcast< ViewerState *>( aViewerState.get() ) );

	//int numDegrees = event->delta() / 8;
	//int numSteps = numDegrees / 15;
	if( mInteractionMode == imORBIT_CAMERA || mInteractionMode == imCUT_PLANE ) { //prevent scale jumping during camera orbit
		return false;
	}

	if ( state.viewType == vt3D ) {
		float dollyRatio = 1.1f;
		if ( event->delta() > 0 ) {
			dollyRatio = 1.0f/dollyRatio;
		}
		state.getViewerWindow< GeneralViewer >().cameraDolly( dollyRatio );
		event->accept();
		return true;
	}
	if ( state.viewType == vt2DAlignedSlices ) {
		state.getViewerWindow< GeneralViewer >().changeCurrentSlice( event->delta() > 0 ? 1: -1 );

		event->accept();
		return true;
	}

	return false;
}


void
ViewerController::timerCall()
{
	if ( mInteractionMode == imFAST_SLICE_CHANGE ) {
		ASSERT(mTmpViewer);
		mTmpViewer->changeCurrentSlice( mPositive ? 1: -1 );
	}
}

} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/
