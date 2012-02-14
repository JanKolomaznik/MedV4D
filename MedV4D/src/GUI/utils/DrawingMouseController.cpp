#include "MedV4D/GUI/utils/DrawingMouseController.h"
#include "MedV4D/GUI/widgets/GeneralViewer.h"

bool
ADrawingMouseController::mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
{ 
	ViewerState &state = *(boost::polymorphic_downcast< ViewerState *>( aViewerState.get() ) );
	
	if ( state.viewType == vt2DAlignedSlices && mIsDrawing) {
		Vector3f lastPos = mTrackInfo.mLastSpaceCoords;
		Vector3f diff;
		mTrackInfo.trackUpdate( aEventInfo.event->pos(), aEventInfo.event->globalPos(), aEventInfo.realCoordinates, diff );
		
		drawStep( lastPos, realCoordinates );
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
	ViewerState &state = *(boost::polymorphic_downcast< ViewerState *>( aViewerState.get() ) );
	mTrackInfo.startTracking( aEventInfo.event->pos(), aEventInfo.event->globalPos(), aEventInfo.realCoordinates );
	if ( state.viewType == vt2DAlignedSlices ) {
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
