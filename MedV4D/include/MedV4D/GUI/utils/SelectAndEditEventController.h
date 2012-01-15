#ifndef SELECT_AND_EDIT_EVENT_CONTROLLER_H
#define SELECT_AND_EDIT_EVENT_CONTROLLER_H

#include "MedV4D/GUI/widgets/GeneralViewer.h"

class SelectEventController: public M4D::GUI::Viewer::AViewerController
{
public:
	APrimitiveCreationEventController(): mSelectInteractionButton( Qt::LeftButton ), mDeselectInteractionButton( Qt::RightButton )
	{}
	
	/*bool
	mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
	{ 
		M4D::GUI::Viewer::ViewerState &state = *(boost::polymorphic_downcast< M4D::GUI::Viewer::ViewerState *>( aViewerState.get() ) );
		if ( (aEventInfo.event->buttons() == Qt::NoButton) && (state.viewType == M4D::GUI::Viewer::vt3D) && (mCurrentStage > 0) ) 
		{
			float t1, t2;
			if ( closestPointsOnTwoLines( mPoint, mDirection, aEventInfo.point, aEventInfo.direction, t1, t2 ) ) {
				Vector3f point = mPoint + (t1 * mDirection);
				*mPrimitive = M4D::Point3Df( point );
			}

			//mPrimitive->secondPoint() = aEventInfo.realCoordinates;
			state.viewerWindow->update();
			return true;
		}
		return false; 
	}*/

	bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
	{ 
		M4D::GUI::Viewer::ViewerState &state = *(boost::polymorphic_downcast< M4D::GUI::Viewer::ViewerState *>( aViewerState.get() ) );
		if ( aEventInfo.event->button() == mSelectInteractionButton ) {
			select();
			state.viewerWindow->update();
			return true;
		} 

		if ( aEventInfo.event->button() == mDeselectInteractionButton ) {
			deselect();
			state.viewerWindow->update();
			return true;
		}
		return false;
	}
	
	virtual void
	select(){}

	virtual void
	deselect(){}
	
	Vector3f mPoint;
	Vector3f mDirection;
};


#endif /*SELECT_AND_EDIT_EVENT_CONTROLLER_H*/
