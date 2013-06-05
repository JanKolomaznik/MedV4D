#include "MedV4D/GUI/utils/DrawingMouseController.h"
#include "MedV4D/GUI/widgets/GeneralViewer.h"

bool
ADrawingMouseController::mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
{ 
	M4D::GUI::Viewer::ViewerState &state = *(boost::polymorphic_downcast< M4D::GUI::Viewer::ViewerState *>( aViewerState.get() ) );
	
	if ( state.viewType == M4D::GUI::Viewer::vt2DAlignedSlices && mIsDrawing) {
		Vector3f lastPos = mTrackInfo.mLastSpaceCoords;
		Vector3f diff;
		mTrackInfo.trackUpdate( aEventInfo.event->pos(), aEventInfo.event->globalPos(), Vector3f(glm::value_ptr(aEventInfo.realCoordinates)), diff );
		
		drawStep( lastPos, Vector3f(glm::value_ptr(aEventInfo.realCoordinates)));
		state.viewerWindow->update();
	}
	return false; 
}

bool	
ADrawingMouseController::mouseDoubleClickEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
{ 
	return false; 
}

bool
ADrawingMouseController::mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
{ 
	M4D::GUI::Viewer::ViewerState &state = *(boost::polymorphic_downcast< M4D::GUI::Viewer::ViewerState *>( aViewerState.get() ) );
	mTrackInfo.startTracking( aEventInfo.event->pos(), aEventInfo.event->globalPos(), Vector3f(glm::value_ptr(aEventInfo.realCoordinates)));
	if ( state.viewType == M4D::GUI::Viewer::vt2DAlignedSlices ) {
		if( aEventInfo.event->button() == mEditorInteractionButton ) {
			mIsDrawing = true;
			return true;
		}
	}
	return false; 
}

bool
ADrawingMouseController::mouseReleaseEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
{ 
	if ( aEventInfo.event->button() == mEditorInteractionButton ) 
	{
		mIsDrawing = false;
		return true;
	}
	return false; 
}

bool
ADrawingMouseController::wheelEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QWheelEvent * event ) 
{ 
	return false; 
}
